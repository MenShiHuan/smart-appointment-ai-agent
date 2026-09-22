"""
天气工具测试（Open-Meteo 方案）

全部离线：注入 fake fetcher，验证：
1. 地理编码 + 天气查询的组合调用（城市名 → 经纬度 → 天气）
2. 城市查不到 → 兜底消息，不调天气接口
3. 网络失败 → 兜底消息，不抛异常
4. 天气数据缺失 → 兜底消息
5. WMO 天气代码 → 中文映射
6. 同步 _run 可用
"""
import asyncio
import pytest

from agents.appointment.weather_tool import WeatherTool, FALLBACK_MESSAGE, GEOCODING_URL, FORECAST_URL


class FakeFetcher:
    """按 URL 路由返回预设 JSON，并记录所有调用"""

    def __init__(self, geocoding_result, weather_result=None, raise_on=None):
        self.geocoding_result = geocoding_result
        self.weather_result = weather_result
        self.raise_on = raise_on  # 'geocoding' | 'forecast' | None
        self.calls = []

    async def __call__(self, url, params):
        self.calls.append({"url": url, "params": params})
        if self.raise_on and self.raise_on in url:
            raise RuntimeError("network down")
        if "geocoding" in url:
            return self.geocoding_result
        return self.weather_result


GEOCODING_XIAN = {
    "results": [
        {"latitude": 34.34, "longitude": 108.94, "name": "西安市", "country": "中国"}
    ]
}

WEATHER_OK = {
    "current": {
        "temperature_2m": 12.5,
        "apparent_temperature": 10.2,
        "relative_humidity_2m": 55,
        "wind_speed_10m": 3.4,
        "weather_code": 61,  # 小雨
    }
}


def test_composes_geocoding_and_forecast():
    """城市名 → 经纬度 → 天气，两步组合正确"""
    fetcher = FakeFetcher(GEOCODING_XIAN, WEATHER_OK)
    tool = WeatherTool(fetcher=fetcher)

    result = asyncio.run(tool._get_weather_data("西安"))

    assert "西安市" in result
    assert "小雨" in result
    assert "12°C" in result
    assert "体感10°C" in result
    assert "湿度55%" in result

    # 校验两步调用
    assert fetcher.calls[0]["url"] == GEOCODING_URL
    assert fetcher.calls[0]["params"]["name"] == "西安"
    assert fetcher.calls[1]["url"] == FORECAST_URL
    assert fetcher.calls[1]["params"]["latitude"] == 34.34
    assert fetcher.calls[1]["params"]["longitude"] == 108.94


def test_city_not_found_returns_fallback_without_forecast_call():
    """城市查不到 → 兜底消息，且不再请求天气接口"""
    fetcher = FakeFetcher({"results": []})
    tool = WeatherTool(fetcher=fetcher)

    result = asyncio.run(tool._get_weather_data("不存在的城市"))

    assert result == FALLBACK_MESSAGE
    assert len(fetcher.calls) == 1  # 只调了地理编码


def test_network_error_returns_fallback_not_exception():
    """网络失败 → 兜底消息，不抛异常"""
    fetcher = FakeFetcher(GEOCODING_XIAN, WEATHER_OK, raise_on="geocoding")
    tool = WeatherTool(fetcher=fetcher)

    result = asyncio.run(tool._get_weather_data("西安"))

    assert result == FALLBACK_MESSAGE


def test_missing_weather_fields_returns_fallback():
    """天气数据缺失 → 兜底消息"""
    fetcher = FakeFetcher(GEOCODING_XIAN, {"current": {}})
    tool = WeatherTool(fetcher=fetcher)

    result = asyncio.run(tool._get_weather_data("西安"))

    assert result == FALLBACK_MESSAGE


def test_weather_code_mapping_covers_common_codes():
    """WMO 代码中文映射（0 晴 / 61 小雨 / 95 雷暴）"""
    fetcher = FakeFetcher(GEOCODING_XIAN, {"current": {"temperature_2m": 20, "weather_code": 95}})
    tool = WeatherTool(fetcher=fetcher)

    result = asyncio.run(tool._get_weather_data("西安"))

    assert "雷暴" in result


def test_sync_run_returns_string():
    """同步 _run 可用"""
    fetcher = FakeFetcher(GEOCODING_XIAN, WEATHER_OK)
    tool = WeatherTool(fetcher=fetcher)

    result = tool._run("西安")

    assert isinstance(result, str)
    assert "西安市" in result
