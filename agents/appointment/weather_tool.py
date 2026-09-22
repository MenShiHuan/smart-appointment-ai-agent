"""
天气查询工具（Open-Meteo）

方案：
1. 地理编码：Open-Meteo 免费 Geocoding API，城市名 → 经纬度坐标
   https://geocoding-api.open-meteo.com/v1/search?name=城市
2. 天气查询：Open-Meteo Forecast API，经纬度 → 当前天气
   https://api.open-meteo.com/v1/forecast?latitude=..&longitude=..
全程无需 API Key，免费无调用限制。

降级策略（稳定可靠优先）：
- 城市查不到 / 网络失败 / 数据缺失 → 返回通用温馨提示，不抛异常、不阻塞预约流程。
"""
import asyncio
from typing import Callable, Optional, Tuple

from langchain.tools import BaseTool

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

# WMO 天气代码 → 中文描述
WEATHER_CODE_ZH = {
    0: "晴",
    1: "基本晴朗",
    2: "多云",
    3: "阴",
    45: "有雾",
    48: "雾凇",
    51: "小毛毛雨",
    53: "毛毛雨",
    55: "大毛毛雨",
    61: "小雨",
    63: "中雨",
    65: "大雨",
    71: "小雪",
    73: "中雪",
    75: "大雪",
    80: "阵雨",
    81: "强阵雨",
    82: "强对流阵雨",
    95: "雷暴",
    96: "雷暴伴冰雹",
    99: "强雷暴伴冰雹",
}

FALLBACK_MESSAGE = "天气服务暂不可用。出行前请关注当地天气，注意适当保暖。"


class WeatherTool(BaseTool):
    """天气查询工具（Open-Meteo，无需 API Key）"""

    name: str = "get_current_weather"
    description: str = "获取指定城市的当前天气信息，返回中文描述（天气现象、气温、体感、湿度、风速）"

    def __init__(self, fetcher: Optional[Callable] = None, **kwargs):
        """
        Args:
            fetcher: 可注入的异步 HTTP 获取函数 `async def(url, params) -> dict`，
                     用于离线测试；为 None 时使用 aiohttp 真实请求。
        """
        super().__init__(**kwargs)
        self._fetcher = fetcher

    async def _request_json(self, url: str, params: dict) -> dict:
        """发起 HTTP 请求并返回 JSON"""
        if self._fetcher is not None:
            return await self._fetcher(url, params)

        import aiohttp
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as response:
                response.raise_for_status()
                return await response.json()

    async def _resolve_coordinates(self, city: str) -> Optional[Tuple[float, float, str]]:
        """地理编码：城市名 → (纬度, 经度, 规范城市名)"""
        data = await self._request_json(GEOCODING_URL, {
            "name": city,
            "count": 1,
            "language": "zh",
            "format": "json",
        })
        results = data.get("results") or []
        if not results:
            return None
        first = results[0]
        return first["latitude"], first["longitude"], first.get("name", city)

    async def _get_weather_data(self, city: str = "北京") -> str:
        """获取指定城市的当前天气（中文描述）"""
        try:
            coords = await self._resolve_coordinates(city)
            if coords is None:
                print(f"[WeatherTool] 未找到城市'{city}'，返回兜底消息")
                return FALLBACK_MESSAGE
            latitude, longitude, resolved_name = coords

            data = await self._request_json(FORECAST_URL, {
                "latitude": latitude,
                "longitude": longitude,
                "current": "temperature_2m,relative_humidity_2m,apparent_temperature,weather_code,wind_speed_10m",
                "timezone": "Asia/Shanghai",
                "forecast_days": 1,
            })
            current = data.get("current") or {}
            if "temperature_2m" not in current:
                print("[WeatherTool] 天气数据缺失，返回兜底消息")
                return FALLBACK_MESSAGE

            temp = current["temperature_2m"]
            feels = current.get("apparent_temperature", temp)
            humidity = current.get("relative_humidity_2m", 0)
            wind = current.get("wind_speed_10m", 0)
            code = current.get("weather_code", 0)
            weather_text = WEATHER_CODE_ZH.get(code, f"天气代码{code}")

            return (
                f"{resolved_name}当前天气：{weather_text}，气温{temp:.0f}°C"
                f"（体感{feels:.0f}°C），湿度{humidity}%，风速{wind}m/s"
            )
        except Exception as e:
            print(f"[WeatherTool] 天气查询失败: {e}")
            return FALLBACK_MESSAGE

    def _run(self, city: str = "北京") -> str:
        """同步版本 - 不推荐使用"""
        return asyncio.run(self._get_weather_data(city))

    async def _arun(self, city: str = "北京") -> str:
        """异步版本"""
        return await self._get_weather_data(city)
