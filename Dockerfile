# 智能健康理疗预约与服务 Agent 平台（F2：单服务一键起）
# 说明：Embedding 使用宿主机 Ollama（bge-m3），容器通过 EMBEDDING_BASE_URL 指向宿主机。
FROM python:3.11-slim

WORKDIR /app

# 先装依赖（利用镜像层缓存）
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 拷贝项目
COPY . .

# 数据与日志目录
RUN mkdir -p /app/data /app/logs

EXPOSE 8000

ENV DATABASE_URL=sqlite:////app/data/smart_appointment.db \
    CHECKPOINT_DB_PATH=/app/data/checkpoints.db

CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
