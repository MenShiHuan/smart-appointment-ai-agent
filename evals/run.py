"""
评测流水线主入口（E2）

四个子集分层评测，全部走真实 LLM（DeepSeek）与真实链路：
- intent：意图分类准确率（TaskClassifier 单轮分类）
- slots ：槽位抽取 Exact Match（InputParser 结构化抽取）
- qa    ：咨询回答正确率（RAG 检索 + 生成 + LLM-as-judge 判分）
- e2e   ：端到端成功率（生产 LangGraph 图多轮断言，含档期冲突/禁忌转人工）

用法：
  python -m evals.run                          # 全部子集
  python -m evals.run --subset intent,slots    # 指定子集
  python -m evals.run --subset qa --limit 5    # 抽样小跑
  python -m evals.run --report evals/report.md # 指定报告路径

依赖：Ollama（bge-m3）已启动；.env 已配置 DeepSeek key。
产物：evals/report.md（各子集指标 + 逐条明细 + 失败样本，供 E3 回灌）。

隔离：评测使用独立数据库（data/eval_smart_appointment.db 与 eval_checkpoints.db），
绝不读写生产库；每次运行默认清空评测库保证可复现（EVAL_KEEP_DB=1 可保留）。
"""
import argparse
import asyncio
import json
import os
import re
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

# ============ 评测环境隔离（必须在任何业务模块导入前生效） ============
_EVAL_DIR = Path(__file__).resolve().parent.parent / "data"
_EVAL_DB_FILE = _EVAL_DIR / "eval_smart_appointment.db"
_EVAL_CKPT_FILE = _EVAL_DIR / "eval_checkpoints.db"
os.environ["DATABASE_URL"] = f"sqlite:///{_EVAL_DB_FILE}"
os.environ["CHECKPOINT_DB_PATH"] = str(_EVAL_CKPT_FILE)


def _reset_eval_db():
    """清空评测库（默认每次运行重建，保证可复现；EVAL_KEEP_DB=1 保留）"""
    if os.environ.get("EVAL_KEEP_DB") == "1":
        return
    for f in (_EVAL_DB_FILE, _EVAL_CKPT_FILE):
        try:
            f.unlink()
        except FileNotFoundError:
            pass

# Windows 控制台默认 GBK，无法输出 ✅/❌ 等字符；强制 UTF-8 输出
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from langchain_core.chat_history import InMemoryChatMessageHistory

from config.model_provider import create_chat_model
from evals.time_utils import resolve_relative_time

DATASET_DIR = Path(__file__).parent / "dataset"
DEFAULT_REPORT = Path(__file__).parent / "report.md"


def load_dataset(name: str):
    with open(DATASET_DIR / f"{name}.json", encoding="utf-8") as f:
        return json.load(f)


# ============ 意图分类 ============

async def eval_intent(llm, cases, limit=None):
    from agents.task_classification.task_classifier import TaskClassifier
    classifier = TaskClassifier(llm)
    results = []
    for case in (cases[:limit] if limit else cases):
        started = time.perf_counter()
        category = await classifier.classify_task(case["input"])
        elapsed = time.perf_counter() - started
        results.append({
            "id": case["id"], "input": case["input"],
            "expected": case["expected_category"], "actual": category,
            "passed": category == case["expected_category"], "elapsed": elapsed,
        })
    return results


# ============ 槽位抽取 ============

def _compare_slots(expected: dict, ex) -> dict:
    """逐字段比对；start_time 相对表达式按运行日换算"""
    checks = {}

    if expected["start_time"] == "未知":
        checks["start_time"] = ex.start_time in (None, "未知")
    else:
        expected_dt = resolve_relative_time(expected["start_time"])
        actual = None
        if ex.start_time and ex.start_time != "未知":
            try:
                actual = datetime.strptime(ex.start_time, "%Y-%m-%d %H:%M")
            except ValueError:
                actual = None
        checks["start_time"] = (expected_dt is not None and actual == expected_dt)

    for field in ("duration", "project", "gender", "technician_name", "preference"):
        checks[field] = getattr(ex, field, None) == expected[field]

    checks["unrelated"] = bool(ex.unrelated) == bool(expected["unrelated"])
    return checks


async def eval_slots(llm, cases, limit=None):
    from agents.appointment.input_parser import InputParser
    parser = InputParser(llm)
    results = []
    for case in (cases[:limit] if limit else cases):
        started = time.perf_counter()
        ex = await parser.extract(case["input"], InMemoryChatMessageHistory())
        elapsed = time.perf_counter() - started
        checks = _compare_slots(case["expected_slots"], ex)
        results.append({
            "id": case["id"], "input": case["input"],
            "expected": case["expected_slots"],
            "actual": {f: getattr(ex, f, None) for f in
                       ("start_time", "duration", "project", "gender",
                        "technician_name", "preference", "unrelated")},
            "checks": checks,
            "passed": all(checks.values()),
            "failed_fields": [f for f, ok in checks.items() if not ok],
            "elapsed": elapsed,
        })
    return results


# ============ 咨询问答（LLM-as-judge） ============

JUDGE_PROMPT = """你是严格的评测裁判。判断助手回答是否满足参考要点。

题目：{question}
参考要点（每条都必须被回答覆盖；语义等价即算命中，如"128元"与"收费128块"等价）：
{points}
助手回答：
{answer}

只输出JSON：{{"correct": true或false, "reason": "一句话说明"}}"""


def _extract_json(text: str) -> str:
    """剥离代码块标记，取首个 JSON 片段"""
    text = re.sub(r"```(?:json)?", "", text).strip()
    match = re.search(r"\{.*\}", text, re.S)
    return match.group(0) if match else text


async def judge_answer(judge_llm, question: str, points: list, answer: str):
    prompt = JUDGE_PROMPT.format(
        question=question,
        points="\n".join(f"- {p}" for p in points),
        answer=answer[:800],
    )
    for _ in range(2):
        response = await judge_llm.ainvoke([{"role": "user", "content": prompt}])
        raw = response.content if hasattr(response, "content") else str(response)
        try:
            data = json.loads(_extract_json(raw))
            return bool(data.get("correct")), str(data.get("reason", "")), False
        except Exception:
            prompt += f"\n你上次输出无法解析：{raw[:100]}\n请直接输出纯JSON。"
    # judge 输出解析失败：与"回答错误"区分标记，避免归因误导
    return False, "judge 输出解析失败", True


async def eval_qa(llm, cases, limit=None):
    from agents.consultant.consultation_processor import ConsultationProcessor
    from agents.consultant.consultation_classifier import ConsultationClassifier
    from agents.consultant.response_generator import ResponseGenerator
    from agents.consultant.knowledge_retriever import KnowledgeRetriever

    # 咨询链路：真实检索（生产知识库单例）+ 真实生成 + 真实理疗师档案注入
    processor = ConsultationProcessor(
        KnowledgeRetriever(), ConsultationClassifier(llm), ResponseGenerator(llm),
        memory_recaller=None,
    )
    results = []
    for case in (cases[:limit] if limit else cases):
        started = time.perf_counter()
        answer = await processor.process_consultation(case["input"])
        gen_elapsed = time.perf_counter() - started
        started = time.perf_counter()
        correct, reason, judge_parse_failed = await judge_answer(
            llm, case["input"], case["reference_points"], answer)
        judge_elapsed = time.perf_counter() - started
        results.append({
            "id": case["id"], "input": case["input"],
            "reference_points": case["reference_points"],
            "answer": answer[:300], "correct": correct, "judge_reason": reason,
            "judge_parse_failed": judge_parse_failed,
            "passed": correct,
            "elapsed": gen_elapsed + judge_elapsed,
        })
    return results


# ============ 端到端 ============

def _db():
    from db.db_router import DatabaseRouter
    # 评测库由 DATABASE_URL 环境变量指向（见文件顶部隔离配置）
    return DatabaseRouter()


def _ensure_technicians():
    from services.technician_service import TechnicianService
    svc = TechnicianService()
    if svc.get_technicians_count() == 0:
        svc.initialize_default_technicians()


def _schedule_max_id(db) -> int:
    from db.models import TechnicianSchedule
    with db.session_manager.session_scope() as session:
        rows = session.query(TechnicianSchedule.id).all()
        return max((r[0] for r in rows), default=0)


def _add_busy(name: str, start_dt, end_dt) -> int:
    """预置档期冲突种子数据，返回新增排班行 id（供清理）"""
    from services.technician_service import TechnicianService
    svc = TechnicianService()
    tech = svc.get_technician_by_name(name)
    if not tech:
        return None
    return svc.db.technicians.add_schedule(
        tech["id"], start_dt, end_dt, "busy", appointment_id=-1)


def _new_busy_after(db, max_id) -> bool:
    from db.models import TechnicianSchedule
    with db.session_manager.session_scope() as session:
        rows = session.query(TechnicianSchedule).filter(
            TechnicianSchedule.id > max_id,
            TechnicianSchedule.status == "busy",
            TechnicianSchedule.appointment_id > 0,
        ).all()
        return len(rows) > 0


def _cleanup_schedules(db, max_id, seed_ids) -> None:
    """清理本用例产生的排班：种子行按 id 精确删除，其余按基线 id 删除"""
    from db.models import TechnicianSchedule
    with db.session_manager.session_scope() as session:
        if seed_ids:
            session.query(TechnicianSchedule).filter(
                TechnicianSchedule.id.in_(seed_ids)).delete(synchronize_session=False)
        session.query(TechnicianSchedule).filter(
            TechnicianSchedule.id > max_id).delete(synchronize_session=False)


def _wipe_memory_data() -> None:
    """记忆相关评测用例的确定性前置：清空行为/偏好/记忆条目"""
    db = _db()
    from db.models import UserBehavior, UserPreference, MemoryEntry
    with db.session_manager.session_scope() as session:
        session.query(UserBehavior).delete()
        session.query(UserPreference).delete()
        session.query(MemoryEntry).delete()


def _pre_setup(case_id: str, db) -> tuple:
    """用例预置数据（档期冲突等），返回 (排班表基线 max_id, 种子行 id 列表) 供断言与清理"""
    _ensure_technicians()
    seed_ids = []
    if case_id in ("e2e-002", "e2e-003"):
        start_dt = resolve_relative_time("TOMORROW 15:00")
        seed_id = _add_busy("张伟", start_dt, start_dt.replace(hour=16))
        if seed_id is not None:
            seed_ids.append(seed_id)
    if case_id in ("e2e-007", "e2e-008"):
        _wipe_memory_data()
    return _schedule_max_id(db), seed_ids


async def eval_e2e(cases, limit=None):
    from api.chat_handler import ProcessUserInput_stream

    db = _db()
    results = []
    run_stamp = str(int(time.time()))
    for case in (cases[:limit] if limit else cases):
        baseline, seed_ids = _pre_setup(case["id"], db)
        # 会话 ID 带时间戳：重跑不继承上一轮的残留槽位/图状态
        session_id = f"eval-{case['id']}-{run_stamp}"
        turns_result = []
        case_passed = True
        try:
            for turn in case["turns"]:
                started = time.perf_counter()
                tokens = []
                async for token in ProcessUserInput_stream(
                        turn["user_input"], session_id=session_id):
                    tokens.append(token)
                joined = "".join(tokens)
                elapsed = time.perf_counter() - started

                reply_ok = all(kw in joined for kw in
                               turn["expect"].get("reply_contains", []))
                reply_ok = reply_ok and not any(
                    kw in joined for kw in turn["expect"].get("reply_not_contains", []))
                saved = None
                if "appointment_saved" in turn["expect"]:
                    saved = _new_busy_after(db, baseline)
                    saved_ok = (saved == bool(turn["expect"]["appointment_saved"]))
                    reply_ok = reply_ok and saved_ok

                turns_result.append({
                    "user_input": turn["user_input"],
                    "reply": joined[:200],
                    "passed": reply_ok,
                    "appointment_saved": saved,
                    "elapsed": elapsed,
                })
                if not reply_ok:
                    case_passed = False
                    break  # 该轮失败即中止该用例后续轮次
        except Exception as e:
            turns_result.append({"error": str(e)})
            case_passed = False
        finally:
            _cleanup_schedules(db, baseline, seed_ids)

        results.append({
            "id": case["id"], "scenario": case["scenario"],
            "passed": case_passed, "turns": turns_result,
        })
    return results


# ============ 报告 ============

def _p95(elapsed_list):
    if not elapsed_list:
        return 0.0
    if len(elapsed_list) < 2:
        return elapsed_list[0]
    # 样本不足 20 时按样本数分位，避免 quantiles 抛异常
    n = min(20, len(elapsed_list))
    return statistics.quantiles(elapsed_list, n=n)[n - 2]


def generate_report(all_results: dict, report_path) -> str:
    lines = [
        "# 评测报告",
        "",
        f"> 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "> 模型：DeepSeek deepseek-chat；向量：本地 Ollama bge-m3",
        "",
    ]
    summaries = {}
    for subset, results in all_results.items():
        total = len(results)
        passed = sum(1 for r in results if r["passed"])
        if subset == "e2e":
            # e2e 的 elapsed 挂在 turn 级，按用例聚合（各轮耗时之和）
            elapsed = [sum(t.get("elapsed", 0) for t in r.get("turns", []))
                       for r in results]
        else:
            elapsed = [r.get("elapsed", 0) for r in results]
        summaries[subset] = (total, passed, elapsed)

    lines.append("## 总览")
    lines.append("")
    lines.append("| 子集 | 通过/总数 | 准确率 | P50 | P95 |")
    lines.append("|---|---|---|---|---|")
    for subset, (total, passed, elapsed) in summaries.items():
        rate = f"{passed / total * 100:.1f}%" if total else "-"
        p50 = f"{statistics.median(elapsed):.1f}s" if elapsed else "-"
        p95 = f"{_p95(elapsed):.1f}s" if elapsed else "-"
        lines.append(f"| {subset} | {passed}/{total} | {rate} | {p50} | {p95} |")
    lines.append("")

    for subset, results in all_results.items():
        lines.append(f"## {subset}")
        lines.append("")
        for r in results:
            mark = "✅" if r["passed"] else "❌"
            if subset == "intent":
                lines.append(f"- {mark} {r['id']} `{r['input'][:30]}` "
                             f"期望 {r['expected']} / 实际 {r['actual']}")
            elif subset == "slots":
                detail = "；".join(f"{f}={r['actual'].get(f)}" for f in r["failed_fields"])
                lines.append(f"- {mark} {r['id']} `{r['input'][:30]}`"
                             + (f" 失败字段: {detail}" if not r["passed"] else ""))
            elif subset == "qa":
                lines.append(f"- {mark} {r['id']} `{r['input'][:30]}` "
                             f"判定: {r['judge_reason'][:50]}")
                if not r["passed"]:
                    lines.append(f"  回答: {r['answer'][:120]}")
            elif subset == "e2e":
                lines.append(f"- {mark} {r['id']} {r['scenario']}")
                for t in r["turns"]:
                    if "error" in t:
                        lines.append(f"  - 异常: {t['error'][:100]}")
                    else:
                        lines.append(f"  - {'✅' if t['passed'] else '❌'} `{t['user_input'][:40]}`"
                                     f" -> {t['reply'][:80]}")
        lines.append("")

    text = "\n".join(lines)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(text)
    return text


# ============ 入口 ============

EVALUATORS = {"intent": eval_intent, "slots": eval_slots, "qa": eval_qa}


async def main():
    parser = argparse.ArgumentParser(description="评测流水线")
    parser.add_argument("--subset", default="intent,slots,qa,e2e",
                        choices=["intent", "slots", "qa", "e2e"],
                        help="逗号分隔的子集（intent,slots,qa,e2e）")
    parser.add_argument("--limit", type=int, default=None, help="每子集抽样条数")
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    args = parser.parse_args()

    subsets = [s.strip() for s in args.subset.split(",") if s.strip()]
    llm = create_chat_model(temperature=0)

    _reset_eval_db()
    # 理疗师种子数据：qa 的理疗师档案注入与 e2e 的档期匹配都依赖
    _ensure_technicians()

    all_results = {}
    for subset in subsets:
        print(f"== 评测 {subset} ==")
        if subset == "e2e":
            results = await eval_e2e(load_dataset(subset), args.limit)
        else:
            results = await EVALUATORS[subset](llm, load_dataset(subset), args.limit)
        total = len(results)
        passed = sum(1 for r in results if r["passed"])
        print(f"{subset}: {passed}/{total} ({passed / total * 100:.1f}%)" if total else f"{subset}: 无数据")
        all_results[subset] = results

    if all_results:
        report_text = generate_report(all_results, Path(args.report))
        print(f"\n报告已写入 {args.report}")
        print(report_text.split("## 总览")[1][:600] if "## 总览" in report_text else "")

    # 显式关闭资源，避免后台线程导致进程退出挂起
    from api.chat_handler import close_graph_checkpointer
    await close_graph_checkpointer()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    finally:
        # 评测脚本收尾：确保进程干净退出（不被客户端后台线程拖住）
        os._exit(0)
