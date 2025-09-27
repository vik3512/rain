#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# App Version: 10.3 — Explore Accuracy Patches (A/B + S/U/R/C) + Hybrid mm
# Patch: performance + robustness + small bugfixes (2025-09)

import os, json, time, math, logging
from datetime import datetime, timezone, timedelta
from typing import Tuple, Optional, List
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError

from cachetools import cached, TTLCache
import requests

# ─── Global HTTP session & connection pool tuning ─────────────────────────────
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
import threading
from threading import RLock

GLOBAL_SESSION = requests.Session()
_retry = Retry(total=3, backoff_factor=0.3, status_forcelist=(429,500,502,503,504), allowed_methods={"GET","HEAD"})
_adapter = HTTPAdapter(pool_connections=64, pool_maxsize=64, max_retries=_retry)
for _p in ("http://","https://"):
    GLOBAL_SESSION.mount(_p, _adapter)

_HOST_LIMITS = {
    "opendata.cwa.gov.tw": threading.Semaphore(6),
    "api.open-meteo.com": threading.Semaphore(8),
}
_DEFAULT_SEM = threading.Semaphore(8)

def _host_of(url: str) -> str:
    try:
        from urllib.parse import urlparse as _u
        return _u(url).netloc
    except Exception:
        return ""

def http_get_json(url: str, params=None, timeout: float = 2.5, headers=None):
    sem = _HOST_LIMITS.get(_host_of(url), _DEFAULT_SEM)
    with sem:
        try:
            r = GLOBAL_SESSION.get(url, params=params, timeout=timeout, headers=headers)
            r.raise_for_status()
        except requests.exceptions.RequestException as e:
            raise
        return r.json()

from dash import Dash, html, dcc, Input, Output, State, no_update, ctx
from dash.exceptions import PreventUpdate
import plotly.graph_objects as go

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# ===== Logging =====
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ==== Version Tag ====
APP_VERSION = "10.3"

# ==== IDs / 常數 ====
MAP_ID = "map"
SEARCH_INPUT_ID = "q"
BTN_SEARCH_ID = "btn-search"
BTN_LOCATE_ID = "btn-locate"
BTN_AREA_ID = "btn-area"

# ===== I18N / Assets =====
ASSETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
ASSETS_I18N = os.path.join(ASSETS_DIR, "i18n.json")
try:
    with open(ASSETS_I18N, "r", encoding="utf-8") as f:
        I18N = json.load(f)
except Exception as e:
    logging.warning(f"Load i18n.json failed: {e}; fallback to minimal zh placeholders.")
    I18N = {"zh": {"title":"即時雨區＋路線規劃","panel_title":"控制台","mode_explore":"探索模式","mode_route":"路線模式","placeholder_q":"輸入地點","search":"搜尋","locate":"定位","travel_mode":"交通方式","drive":"開車","scooter":"機車","walk":"步行","transit":"大眾運輸","placeholder_src":"輸入出發地","placeholder_dst":"輸入目的地","plan":"規劃路線","basemap":"底圖","low":"低飽和","osm":"標準 (OSM)","update":"更新於","legend_rain":"降雨熱度","legend_light":"較弱","legend_heavy":"較強","locate_src_title":"將起點設為目前位置","map_center":"地圖中心","toast_err":"發生錯誤","no_gmap_key":"無 Google 路線 API 金鑰","dry":"無雨","partial_result":" (部分結果)","rain":"有雨","lightrain":"小雨","heavy_rain":"大雨/雷雨","warn_thunder":"雷雨","best":"最佳路線","others":"其他路線","origin":"起點","dest":"終點","dest_now":"目的地現在","search_area":"搜尋此區域","addr_fixed":"固定地址","searching":"搜尋中…","stops_in_1h":"雨勢即將停止","stops_in_xh":"雨勢將於 {0} 小時內停止","starts_in_xh":"雨勢將於 {0} 小時內開始","stable_rain":"雨勢穩定","variable_rain":"雨勢多變","low_confidence":"置信度低（遠離測站）","thunder_reminder":"⚡ 可能有雷雨，注意閃電","skip_scan_sunny":"今日晴朗（跳過重掃）","rain_forecast":"預報有雨，詳細掃描中"}}

def t(lang: str, key: str) -> str:
    lang_key = lang or "zh"
    return I18N.get(lang_key, I18N["zh"]).get(key, I18N["zh"].get(key, key))

# ===== 全域常數 =====
GOOGLE_KEY = os.getenv("GOOGLE_MAPS_API_KEY", "").strip()
CWA_API_KEY = os.getenv("CWA_API_KEY", "").strip()
OWM_KEY = os.getenv("OPENWEATHER_API_KEY", "").strip()
HAS_GMAP = bool(GOOGLE_KEY)
UA = {"User-Agent": f"rain-route-assistant/{APP_VERSION}"}
LANG_MAP = {"zh":"zh-TW","en":"en","ja":"ja"}
MAX_SCAN_POINTS = 900

# === PATCH A: IDW parameters (explore accuracy) ===
IDW_R2_MAX = 0.35**2       # ~35km 內
IDW_EPS = 1e-12            # 避免除以 0

_EXECUTOR = ThreadPoolExecutor(max_workers=16)

# ======= B 方案：掃描冷卻 =======
SCAN_COOLDOWN_SEC = 8
LAST_SCAN_TS = 0.0
_last_scan_lock = threading.Lock()

# ===== 地圖基礎 & 新架構參數 =====
BASE_CENTER = [23.9738, 120.9820]
BASE_ZOOM   = 7
SEARCHED_ZOOM = 12
LOCATE_ZOOM = 13
DEFAULT_TILE_STYLE = "carto-positron"
FALLBACK_CENTER = [25.0375, 121.5637]
TW_BBOX = (20.5, 118.0, 26.5, 123.5)

RAIN_CIRCLE_COLORSCALE = [[0.0, "#9BE7FF"], [0.25, "#39B6FF"], [0.5, "#1C64F2"], [0.75, "#6D28D9"], [1.0, "#A21CAF"]]
COLOR_DRY  = "rgba(16,185,129,0.95)"
COLOR_WET  = "rgba(37,99,235,0.85)"

# === PATCH S1: sensitivity tweaks ===
VISUAL_MM_MIN = 0.1          # 文字/單點視覺用
RAIN_DRAW_MM_MIN = 0.05      # 地圖畫點門檻（提高敏感度）
DECISION_MM_MIN = 0.8        # 路線判定維持

# === Route accuracy knobs ===
ROUTE_SAMPLE_STEP_KM   = 0.35   # 沿線加密步長（公里）
ROUTE_BUFFER_OFFSETS_M = [0, 250, -250, 500, -500]  # 橫向緩衝帶（公尺）
DECISION_ROUTE_MM_MIN  = 0.6    # 路線濕/乾判斷門檻（比探索略敏感）
ROUTE_SMOOTH_WIN       = 3      # 多數決平滑視窗（奇數）
ROUTE_INT_CAP          = 5.0    # 風險平均強度上限（mm/h）

THUNDER_MIN, THUNDER_MAX = 95, 99
HEATMAP_MAX_MM = 8.0

# === PATCH U1: 小範圍摘要的半徑（公里） ===
AREA_SUMMARY_RADIUS_KM_CITY   = 2.5   # 市區
AREA_SUMMARY_RADIUS_KM_SUBURB = 6.0   # 近郊
AREA_SUMMARY_RADIUS_KM_RURAL  = 12.0  # 郊區/山區

# === PATCH H1: Hybrid 合成係數 ===
def dynamic_om_blend(lat: float, lon: float, nearest_dist_km: float = None) -> float:
    """動態 OM_BLEND：距最近測站越近、OM 權重越低；外海或測站稀疏處權重提高"""
    if nearest_dist_km is None:
        stations = get_cwa_stations_data()
        if stations:
            nearest_dist_sq = min((lat - s['lat'])**2 + (lon - s['lon'])**2 for s in stations)
            nearest_dist_km = math.sqrt(nearest_dist_sq) * 111  # 粗估 km
        else:
            nearest_dist_km = 50.0  # 預設遠
    blend = 0.6 + (nearest_dist_km / 100.0) * 0.8
    return min(1.4, max(0.6, blend))

def base_map_figure(center=BASE_CENTER, zoom=BASE_ZOOM, style=DEFAULT_TILE_STYLE):
    fig = go.Figure(go.Scattermapbox(lat=[center[0]], lon=[center[1]], mode='markers', marker=dict(size=0, opacity=0)))
    fig.update_layout(
        mapbox=dict(style=style, center=dict(lat=center[0], lon=center[1]), zoom=zoom),
        margin=dict(l=0, r=0, t=0, b=0), uirevision="map", dragmode="pan", showlegend=False,
    )
    return fig

# ===== 快取 =====
om_api_cache = TTLCache(maxsize=8000, ttl=3500)
api_cache = TTLCache(maxsize=256, ttl=180)  # 下修到 3 分鐘
cwa_warnings_cache = TTLCache(maxsize=1, ttl=180)
cwa_stations_cache = TTLCache(maxsize=1, ttl=1800)
cwa_temp_cache = TTLCache(maxsize=1, ttl=300)
owm_cache = TTLCache(maxsize=1024, ttl=300)
cwa_qpf_cache = TTLCache(maxsize=1, ttl=600)  # QPF 快取

# 短期黏滯快取：給 get_rain_mm_hybrid（座標量化到 3 位小數；60 秒生命）
mm_cache = TTLCache(maxsize=20000, ttl=60)

# 共用鎖（cachetools 的 @cached）
_cache_lock = RLock()
mm_cache_lock = RLock()

# ===== 核心輔助 =====
def _safe_bounds_from_relayout(relayout_data):
    try:
        if not isinstance(relayout_data, dict):
            return None
        derived = relayout_data.get("map._derived") or relayout_data.get("mapbox._derived") or {}
        if not isinstance(derived, dict):
            return None
        coords = derived.get("coordinates") or {}
        bounds = coords.get("bounds")
        if (isinstance(bounds, list) and len(bounds) == 2 and
            all(isinstance(x, (list, tuple)) and len(x) == 2 for x in bounds)):
            w, s = float(bounds[0][0]), float(bounds[0][1])
            e, n = float(bounds[1][0]), float(bounds[1][1])
            # 外擴 10%
            expand = 0.1
            s -= (n - s) * expand
            n += (n - s) * expand
            w -= (e - w) * expand
            e += (e - w) * expand
            return {"west": w, "south": s, "east": e, "north": n}
    except Exception:
        pass
    return None

def _center_zoom_from_relayout(relayout, fallback_view=None):
    fv_center = (fallback_view or {}).get("center") or FALLBACK_CENTER
    fv_zoom   = (fallback_view or {}).get("zoom") or BASE_ZOOM
    if not isinstance(relayout, dict):
        return fv_center, fv_zoom
    cen = relayout.get("mapbox.center") or relayout.get("geo.center") or {}
    lon = cen.get("lon", fv_center[1])
    lat = cen.get("lat", fv_center[0])
    zoom = relayout.get("mapbox.zoom") or relayout.get("geo.zoom") or fv_zoom
    return [lat, lon], float(zoom)

def _normalize_addr(val):
    if isinstance(val, str):
        return val
    if isinstance(val, dict):
        for k in ("formatted_address", "display_name", "name", "address"):
            v = val.get(k)
            if isinstance(v, str) and v.strip():
                return v
    return ""

def is_in_taiwan(lat: float, lon: float) -> bool:
    min_lat, min_lon, max_lat, max_lon = TW_BBOX
    return min_lat <= lat <= max_lat and min_lon <= lon <= max_lon

@cached(cwa_stations_cache, lock=_cache_lock)
def get_cwa_stations_data():
    if not CWA_API_KEY: return None
    try:
        url = "https://opendata.cwa.gov.tw/api/v1/rest/datastore/O-A0002-001"
        params = {"Authorization": CWA_API_KEY, "elementName": "RAIN,MIN_10"}
        r = GLOBAL_SESSION.get(url, params=params, timeout=10, headers=UA)
        r.raise_for_status()
        data = r.json()
        locations = data.get('records', {}).get('location', [])
        station_data = []
        for loc in locations:
            lat = float(loc.get('lat')); lon = float(loc.get('lon'))
            rain_1h, rain_10min = -1.0, -1.0
            for elem in loc.get('weatherElement', []):
                if elem.get('elementName') == 'RAIN':
                    rain_1h = float(elem.get('elementValue'))
                elif elem.get('elementName') == 'MIN_10':
                    rain_10min = float(elem.get('elementValue'))
            station_data.append({'lat': lat, 'lon': lon, 'rain_1h': rain_1h, 'rain_10min': rain_10min})
        return station_data
    except requests.exceptions.RequestException as e:
        logging.error(f"Failed to fetch CWA stations: {e}")
        return None

# === PATCH A+S2a: IDW helper（取 max(1h, 10min*6)） + 動態 k ===
def _cwa_idw_mm(lat: float, lon: float) -> Optional[float]:
    stations = get_cwa_stations_data()
    if not stations:
        return None

    rows = []
    for s in stations:
        d2 = (lat - s['lat'])**2 + (lon - s['lon'])**2
        if d2 > IDW_R2_MAX:
            continue

        mm = None
        if s['rain_1h'] >= 0 and s['rain_10min'] >= 0:
            mm = max(float(s['rain_1h']), float(s['rain_10min']) * 6.0)
        elif s['rain_1h'] >= 0:
            mm = float(s['rain_1h'])
        elif s['rain_10min'] >= 0:
            mm = float(s['rain_10min']) * 6.0
        if mm is None:
            continue

        rows.append((d2, mm))

    if not rows:
        return None

    rows.sort(key=lambda x: x[0])
    # 動態 k：依可用站數自調 (2-6)
    num_available = len(rows)
    k = max(2, min(6, int(num_available * 0.6) + 1))  # 密集 k~6, 稀疏 k~2
    rows = rows[:k]

    if rows[0][0] < 1e-10:
        return rows[0][1]

    wsum = 0.0
    vsum = 0.0
    for d2, mm in rows:
        w = 1.0 / (d2 + IDW_EPS)
        wsum += w
        vsum += w * mm
    return vsum / wsum if wsum > 0 else None

# === 最近站也取 max(1h, 10min*6) ===
def _cwa_nearest_mm(lat: float, lon: float) -> Optional[Tuple[float, float]]:
    stations = get_cwa_stations_data()
    if not stations: return None
    min_dist_sq = float('inf'); closest = None; nearest_dist_sq = None
    for s in stations:
        d2 = (lat - s['lat'])**2 + (lon - s['lon'])**2
        mm = None
        if s['rain_1h'] >= 0 and s['rain_10min'] >= 0:
            mm = max(float(s['rain_1h']), float(s['rain_10min']) * 6.0)
        elif s['rain_1h'] >= 0:
            mm = float(s['rain_1h'])
        elif s['rain_10min'] >= 0:
            mm = float(s['rain_10min']) * 6.0
        if mm is None:
            continue
        if d2 < min_dist_sq:
            min_dist_sq = d2
            closest = mm
            nearest_dist_sq = d2
    if min_dist_sq > 0.25**2:
        return None
    # 回傳 nearest_dist_km 供 blend 用
    nearest_dist_km = math.sqrt(nearest_dist_sq) * 111  # 粗估
    return closest, nearest_dist_km

# ===== Open-Meteo（逐小時） + 分段內插 ===
def _current_hour_key_utc(): return datetime.utcnow().strftime("%Y-%m-%dT%H")

@cached(om_api_cache, lock=_cache_lock)
def _om_hourly_forecast_data_cached_api(qlat: float, qlon: float, hour_key_utc: str):
    try:
        r = GLOBAL_SESSION.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": qlat, "longitude": qlon,
                "hourly": "precipitation,weather_code,temperature_2m",
                "forecast_days": 2, "timezone": "auto"
            },
            timeout=8, headers=UA
        )
        r.raise_for_status()
        js = r.json()
        h = js.get("hourly", {})
        times, precip, codes, temps = h.get("time", []), h.get("precipitation", [0.0]), h.get("weather_code", [0]), h.get("temperature_2m", [0])
        offset = js.get("utc_offset_seconds", 0)
        now_tz = datetime.now(timezone(timedelta(seconds=offset)))
        now_key = now_tz.strftime("%Y-%m-%dT%H:00")
        idx = times.index(now_key) if now_key in times else 0
        return (times, precip, codes, temps, idx, offset)
    except Exception as e:
        logging.warning(f"OM Forecast fetch throttled: {e}")
        return ([], [0.0], [0], [0], 0, 0)

def _interpolate_om_value(values: list, idx: int, frac: float) -> float:
    """分段內插：OM 小時資料內插到現在時間"""
    if len(values) < 2 or idx >= len(values) - 1:
        return float(values[idx] if idx < len(values) else 0.0)
    return values[idx] * (1 - frac) + values[idx + 1] * frac

def _om_hourly_forecast_data(lat: float, lon: float, hour_key_override: Optional[str] = None):
    data = _om_hourly_forecast_data_cached_api(round(lat, 2), round(lon, 2), hour_key_override or _current_hour_key_utc())
    times, precip, codes, temps, idx, offset = data
    if not times:
        return data
    # 內插：計算當前小時內的進度
    now_tz = datetime.now(timezone(timedelta(seconds=offset)))
    hour_start = datetime(now_tz.year, now_tz.month, now_tz.day, now_tz.hour, 0, 0, tzinfo=now_tz.tzinfo)
    frac = (now_tz - hour_start).total_seconds() / 3600.0
    precip_interp = _interpolate_om_value(precip, idx, frac)
    return (times, [precip_interp] + precip[idx+1:], codes, temps, idx, offset)  # 只內插當前

# 量化座標（路線密集點共享結果）
def _q3(lat: float, lon: float) -> Tuple[float, float]:
    return (round(float(lat), 3), round(float(lon), 3))

# === PATCH H2: Hybrid mm（回傳 (mm, code, is_visual, om_mm, nearest_dist_km)） ===
def get_rain_mm_hybrid(lat: float, lon: float, future_hour: int = 0) -> Tuple[float, int, bool, float, float]:
    """
    支援 future_hour（0 表示此刻；1 表示 1 小時後，以此類推）
    回傳: (hybrid_mm, code, is_visual_rain, om_mm_used, nearest_dist_km)
    """
    # 短期黏滯快取
    try:
        key = (_q3(lat, lon), int(future_hour))
    except Exception:
        key = ((lat, lon), int(future_hour))
    with mm_cache_lock:
        cached_val = mm_cache.get(key)
    if cached_val is not None:
        return cached_val

    om_mm = 0.0; om_code = 0
    try:
        # 注意：_om_hourly_forecast_data 已把「當前小時的內插值」放在 precip[0]
        # 之後的元素就是未來逐小時值
        _, precip, codes, _, idx, _ = _om_hourly_forecast_data(lat, lon)

        if isinstance(future_hour, int) and future_hour >= 0:
            code_idx = min(idx + future_hour, len(codes) - 1) if codes else 0
            if future_hour == 0:
                om_mm = float(precip[0]) if precip else 0.0
            else:
                p_idx = min(future_hour, len(precip) - 1) if precip else 0
                om_mm = float(precip[p_idx]) if precip else 0.0
            om_code = int(codes[code_idx]) if codes else 0
        else:
            om_mm = float(precip[0]) if precip else 0.0
            om_code = int(codes[idx]) if codes else 0
    except Exception:
        pass

    nearest_dist_km = 50.0  # 預設
    if is_in_taiwan(lat, lon):
        idw = _cwa_idw_mm(lat, lon)
        if idw is None:
            nearest = _cwa_nearest_mm(lat, lon)
            if nearest:
                idw, nearest_dist_km = nearest
        if idw is not None:
            blend = dynamic_om_blend(lat, lon, nearest_dist_km)
            mm = max(float(idw), blend * om_mm)
        else:
            mm = om_mm
        is_visual_rain = (mm >= VISUAL_MM_MIN) or (THUNDER_MIN <= om_code <= THUNDER_MAX)
        code = 80 if mm > 0.1 else om_code
        out = (mm, code, is_visual_rain, om_mm, nearest_dist_km)
        with mm_cache_lock:
            mm_cache[key] = out
        return out

    is_visual_rain = (om_mm >= VISUAL_MM_MIN) or (THUNDER_MIN <= om_code <= THUNDER_MAX)
    out = (om_mm, om_code, is_visual_rain, om_mm, nearest_dist_km)
    with mm_cache_lock:
        mm_cache[key] = out
    return out

@cached(cwa_temp_cache, lock=_cache_lock)
def get_cwa_nearby_temp(lat: float, lon: float) -> Optional[float]:
    if not CWA_API_KEY: return None
    try:
        url = "https://opendata.cwa.gov.tw/api/v1/rest/datastore/O-A0001-001"
        params = {"Authorization": CWA_API_KEY, "elementName": "TEMP"}
        r = GLOBAL_SESSION.get(url, params=params, timeout=8, headers=UA)
        r.raise_for_status()
        data = r.json()
        locations = data.get("records", {}).get("location", [])
        best, best_d = None, float("inf")
        for loc in locations:
            try:
                la = float(loc.get("lat")); lo = float(loc.get("lon"))
                d = (la - lat) ** 2 + (lo - lon) ** 2
                temp = None
                for e in loc.get("weatherElement", []):
                    if e.get("elementName") == "TEMP":
                        temp = float(e.get("elementValue"))
                        break
                if temp is not None and d < best_d:
                    best, best_d = temp, d
            except Exception:
                continue
        return best if best is not None and best_d <= (0.25 ** 2) else None
    except requests.exceptions.RequestException as e:
        logging.warning(f"CWA TEMP fetch failed: {e}")
        return None

@cached(owm_cache, lock=_cache_lock)
def get_owm_temp(lat: float, lon: float) -> Optional[float]:
    if not OWM_KEY: return None
    try:
        r = GLOBAL_SESSION.get(
            "https://api.openweathermap.org/data/2.5/weather",
            params={"lat": lat, "lon": lon, "appid": OWM_KEY, "units": "metric"},
            timeout=6, headers=UA
        )
        r.raise_for_status()
        js = r.json()
        return float(js.get("main", {}).get("temp")) if js.get("main") else None
    except requests.exceptions.RequestException as e:
        logging.warning(f"OWM temp failed: {e}")
        return None

def _clear_weather_caches_safely():
    for cache in [om_api_cache, api_cache, cwa_warnings_cache, cwa_stations_cache, cwa_temp_cache, owm_cache, cwa_qpf_cache]:
        try:
            if hasattr(cache, 'clear'): cache.clear()
        except Exception:
            pass

# === QPF 作為掃描閘門 ===
@cached(cwa_qpf_cache, lock=_cache_lock)
def get_cwa_qpf_forecast():
    if not CWA_API_KEY:
        return None
    try:
        url = "https://opendata.cwa.gov.tw/api/v1/rest/datastore/F-D0047-049"
        params = {"Authorization": CWA_API_KEY, "locationName": "全國"}
        r = GLOBAL_SESSION.get(url, params=params, timeout=10, headers=UA)
        r.raise_for_status()
        data = r.json()
        elements = data.get('records', {}).get('weatherElement', [])
        for elem in elements:
            if elem.get('elementName') == 'Wx':
                values = elem.get('time', [{}])[0].get('elementValueArray', [])
                for val in values:
                    desc = val.get('description', '')
                    if any(word in desc for word in ['雨', '雷', '陣雨']):
                        return True  # 有雨預報（大範圍）
        return False  # 無雨
    except requests.exceptions.RequestException as e:
        logging.warning(f"CWA QPF fetch failed: {e}")
        return None  # 失敗時採中立（讓後續邏輯決定是否掃描）

def _qpf_localized_msg(lang: str, thunder: bool, afternoon: bool) -> str:
    """回傳本地化 QPF 提示文字。只處理我們需要的 3 種情況。"""
    if lang == "zh":
        if thunder:
            return "預報有午後雷陣雨" if afternoon else "預報有雷雨"
        return "預報有雨"
    if lang == "ja":
        if thunder:
            return "雷雨の予報（午後の雷雨の可能性あり）" if afternoon else "雷雨の予報"
        return "雨の予報"
    # default: English
    if thunder:
        return "Thunderstorms forecast (afternoon possible)" if afternoon else "Thunderstorms forecast"
    return "Rain in the forecast"

def intersect_bounds_with_rain_areas(bounds, qpf_rain: bool):
    """若 QPF 無雨，直接回空；有雨則外縮到預測雨區（目前簡化為全掃）。"""
    if not qpf_rain:
        return None  # 跳過掃描
    # TODO: 未來整合真實雨區多邊形
    return bounds

# === 小範圍摘要 + 加權重心 ===
def weighted_rain_centroid(wx_points: List[tuple], fallback_lat: float, fallback_lon: float) -> tuple[float, float]:
    if not wx_points:
        return fallback_lat, fallback_lon
    total_mm = sum(mm for _, _, mm in wx_points)
    if total_mm == 0:
        return fallback_lat, fallback_lon
    sum_lat = sum(la * mm for la, _, mm in wx_points) / total_mm
    sum_lon = sum(lo * mm for _, lo, mm in wx_points) / total_mm
    return sum_lat, sum_lon

def _deg_span_from_km(lat: float, radius_km: float) -> tuple[float, float]:
    k = 111.0
    dlat = radius_km / k
    clon = max(0.1, math.cos(math.radians(lat)))
    dlon = radius_km / (k * clon)
    return dlat, dlon

def summarize_small_area_km(center_lat: float, center_lon: float,
                            wx_points: List[tuple], radius_km: float) -> dict:
    if not wx_points or radius_km <= 0:
        return {"core_points": 0}
    dlat, dlon = _deg_span_from_km(center_lat, radius_km)
    s_lat, w_lon = center_lat - dlat, center_lon - dlon
    n_lat, e_lon = center_lat + dlat, center_lon + dlon
    core = [(la, lo, mm) for (la, lo, mm) in wx_points if s_lat <= la <= n_lat and w_lon <= lo <= e_lon]
    if not core:
        return {"core_points": 0}
    mm_list = [mm for (_, _, mm) in core]
    max_mm = max(mm_list); avg_mm = sum(mm_list) / len(core)
    std_mm = (sum((x - avg_mm)**2 for x in mm_list) / len(core))**0.5
    variability = "stable_rain" if std_mm < 1.0 else "variable_rain"
    if max_mm >= 3.0: key_area = "heavy_rain"
    elif max_mm >= 0.6: key_area = "rain"
    else: key_area = "lightrain"
    confidence = "low_confidence" if len(core) < 3 else ""
    return {"core_points": len(core), "max_mm": round(max_mm,1), "avg_mm": round(avg_mm,1), "key_area": key_area,
            "variability": variability, "confidence": confidence}

# === 臨界區再取樣 ===
def _refine_near_threshold(seed_points: List[tuple], bounds, base_step: float,
                           mm_low: float = 0.1, mm_high: float = 4.0,
                           extra_budget: int = 160, timeout_sec: float = 1.6) -> List[tuple]:
    if not seed_points or extra_budget <= 0:
        return []

    (s_lat, w_lon), (n_lat, e_lon) = bounds
    half = max(base_step / 2.0, 0.01)

    candidates = [(la, lo, mm) for (la, lo, mm) in seed_points if mm_low <= mm <= mm_high]
    if not candidates:
        return []

    neigh = set()
    for la, lo, _ in candidates:
        for dlat in (-half, 0.0, half):
            for dlon in (-half, 0.0, half):
                if dlat == 0.0 and dlon == 0.0:
                    continue
                nla, nlo = round(la + dlat, 5), round(lo + dlon, 5)
                if s_lat <= nla <= n_lat and w_lon <= nlo <= e_lon:
                    neigh.add((nla, nlo))
                if len(neigh) >= extra_budget:
                    break
            if len(neigh) >= extra_budget:
                break
        if len(neigh) >= extra_budget:
            break

    neigh = list(neigh)
    results: List[tuple] = []
    if not neigh:
        return results

    def fetch(p):
        try:
            mm, _, _, _ ,_ = get_rain_mm_hybrid(p[0], p[1])
            if (mm is not None) and (mm >= RAIN_DRAW_MM_MIN):
                return (p[0], p[1], min(mm, HEATMAP_MAX_MM))
        except Exception:
            pass
        return None

    with ThreadPoolExecutor(max_workers=8) as ex:
        futures = [ex.submit(fetch, p) for p in neigh]
        end_at = time.time() + timeout_sec
        try:
            for f in as_completed(futures, timeout=timeout_sec):
                if time.time() > end_at:
                    break
                r = f.result()
                if r:
                    results.append(r)
                if len(results) >= extra_budget:
                    break
        except TimeoutError:
            pass

    return results

# === 輕量插值平滑 ===
def smooth_rain_points(wx_points: List[tuple], step: float) -> List[tuple]:
    if len(wx_points) < 3:
        return wx_points
    # 簡化：對鄰近點平均 (僅在此區搜尋時用，控制點數)
    smoothed = []
    points_dict = {(round(la,4), round(lo,4)): mm for la, lo, mm in wx_points}
    for la, lo, mm in wx_points:
        key = (round(la,4), round(lo,4))
        # 找鄰居平均
        neighbors = [mm]  # 自身
        for dla in [-step, 0, step]:
            for dlo in [-step, 0, step]:
                if dla == 0 and dlo == 0: continue
                nkey = (round(la + dla,4), round(lo + dlo,4))
                if nkey in points_dict:
                    neighbors.append(points_dict[nkey])
        avg_mm = sum(neighbors) / len(neighbors)
        smoothed.append((la, lo, avg_mm))
    return smoothed[:MAX_SCAN_POINTS]  # 控制點數

# === 依視窗掃描（含補掃 + QPF 閘門） ===
def get_weather_data_for_bounds(bounds, zoom, timeout_sec=8.0):
    qpf_rain = get_cwa_qpf_forecast()
    if qpf_rain is False:
        return [], True  # 跳過，標記為 partial (晴朗)

    bounds = intersect_bounds_with_rain_areas(bounds, qpf_rain) or bounds
    (s_lat, w_lon), (n_lat, e_lon) = bounds
    timed_out = False

    def get_step(z):
        if z >= 12: return 0.03
        if z >= 11: return 0.05
        if z >= 10: return 0.08
        if z >= 9:  return 0.10
        if z >= 8:  return 0.15
        return 0.20

    step = get_step(zoom)
    if qpf_rain:  # 有雨預報，加密步長
        step *= 0.7
    num_points = ((n_lat - s_lat) / step) * ((e_lon - w_lon) / step) if step > 0 else 1
    while num_points > MAX_SCAN_POINTS:
        step *= 1.4
        num_points = ((n_lat - s_lat) / step) * ((e_lon - w_lon) / step)

    points = set()
    if step > 0:
        lat_range = [s_lat + i * step for i in range(math.ceil((n_lat - s_lat) / step) + 1)]
        lon_range = [w_lon + i * step for i in range(math.ceil((e_lon - w_lon) / step) + 1)]
        for lat in lat_range:
            for lon in lon_range:
                points.add((round(lat, 5), round(lon, 5)))

    points_with_rain: List[tuple] = []

    def fetch(p):
        try:
            mm, _, _, _ ,_ = get_rain_mm_hybrid(p[0], p[1])
            if (mm is not None) and (mm >= RAIN_DRAW_MM_MIN):
                return (p[0], p[1], min(mm, HEATMAP_MAX_MM))
        except Exception:
            pass
        return None

    t_start = time.time()
    primary_budget = max(2.6, timeout_sec * 0.65)
    futures = [_EXECUTOR.submit(fetch, p) for p in points]
    try:
        for future in as_completed(futures, timeout=primary_budget):
            r = future.result()
            if r: points_with_rain.append(r)
    except TimeoutError:
        timed_out = True
        logging.warning(f"Weather scan (primary) timed out after {primary_budget:.1f} seconds.")

    remain = max(0.0, timeout_sec - (time.time() - t_start))
    if remain >= 0.6 and points_with_rain:
        extra = _refine_near_threshold(
            seed_points=points_with_rain,
            bounds=bounds,
            base_step=step,
            mm_low=0.1, mm_high=4.0,
            extra_budget=160,
            timeout_sec=min(1.6, remain)
        )
        if extra:
            points_with_rain.extend(extra)

    # 輕量插值平滑 (僅部分結果)
    if not timed_out and len(points_with_rain) > 10:
        points_with_rain = smooth_rain_points(points_with_rain, step)

    return points_with_rain, timed_out

# ===== 溫度／預報（文字） + 不確定性 + 雷雨提醒 ===
def get_best_temperature(lat: float, lon: float, ui_lang: str) -> Optional[float]:
    if is_in_taiwan(lat, lon):
        cwa_t = get_cwa_nearby_temp(lat, lon)
        if cwa_t is not None:
            return cwa_t
    try:
        _, _, _, temps, idx, _ = _om_hourly_forecast_data(lat, lon)
        if temps and 0 <= idx < len(temps):
            return float(temps[idx])
    except Exception:
        pass
    return get_owm_temp(lat, lon)

@cached(api_cache, lock=_cache_lock)
def get_point_forecast(lat: float, lon: float, lang: str) -> dict:
    try:
        times, precip, codes, temps, idx, offset = _om_hourly_forecast_data(lat, lon)
        if not times:
            raise RuntimeError("OM empty")
        key = "sunny"
        if codes[idx] in [1, 2, 3]: key = "cloudy"
        if codes[idx] > 3: key = "overcast"
        if 50 <= codes[idx] < 70: key = "lightrain"
        if codes[idx] >= 80: key = "heavy_rain"
        if 95 <= codes[idx] <= 99: key = "warn_thunder"

        forecast_str = ""
        # 此刻（內插）在 precip[0]；未來逐小時從 precip[1] 開始
        upper = min(12, len(precip))
        future_precip = precip[1:upper]
        is_raining_now = precip[0] > 0.1  # 已內插
        if is_raining_now:
            try:
                stop_h = next(i for i, p in enumerate(future_precip) if p < 0.1)
                forecast_str = t(lang, "stops_in_1h") if stop_h == 0 else t(lang, "stops_in_xh").format(stop_h + 1)
            except StopIteration:
                pass
        else:
            try:
                start_h = next(i for i, p in enumerate(future_precip) if p > 0.1)
                forecast_str = t(lang, "starts_in_xh").format(start_h + 1)
            except StopIteration:
                pass


        best_temp = get_best_temperature(lat, lon, lang)

        # 雷雨提醒
        thunder_reminder = t(lang, "thunder_reminder") if THUNDER_MIN <= codes[idx] <= THUNDER_MAX else ""

        # 不確定性：從 hybrid 取 nearest_dist_km
        _, _, _, _, nearest_dist_km = get_rain_mm_hybrid(lat, lon)
        confidence_key = "low_confidence" if nearest_dist_km > 20 else ""

        return {"key": key, "forecast": forecast_str, "temp": best_temp, "offset_sec": offset,
                "d_lvl_key": key, "d_temp": best_temp, "thunder_reminder": thunder_reminder,
                "confidence": confidence_key}
    except Exception as e:
        logging.error(f"Error in get_point_forecast: {e}")
        best_temp = get_best_temperature(lat, lon, lang)
        return {"key": "cloudy", "forecast": "", "temp": best_temp, "offset_sec": 0,
                "d_lvl_key": "cloudy", "d_temp": best_temp, "thunder_reminder": "", "confidence": ""}

# ===== Geocode / Reverse =====
@cached(api_cache, lock=_cache_lock)
def smart_geocode(q: str, lang: str = "zh-TW", bounds: Optional[dict] = None):
    if HAS_GMAP:
        try:
            params = {"address": q, "key": GOOGLE_KEY, "language": lang, "region": "tw"}
            if bounds:
                params["bounds"] = f"{bounds['south']},{bounds['west']}|{bounds['north']},{bounds['east']}"
            r = GLOBAL_SESSION.get("https://maps.googleapis.com/maps/api/geocode/json", params=params, headers=UA, timeout=5)
            r.raise_for_status()
            js = r.json()
            if js['status'] == 'OK' and js.get('results'):
                res = js['results'][0]
                loc = res['geometry']['location']
                return res['formatted_address'], (loc['lat'], loc['lng'])
        except requests.exceptions.RequestException as e:
            logging.warning(f"Google Geocode failed, falling back to OSM: {e}")
    params = {"q": q, "format": "jsonv2", "accept-language": lang, "limit": 1}
    if bounds:
        params["viewbox"] = f"{bounds['west']},{bounds['north']},{bounds['east']},{bounds['south']}"
        params["bounded"] = 1
    r = GLOBAL_SESSION.get("https://nominatim.openstreetmap.org/search", params=params, headers=UA, timeout=5)
    r.raise_for_status()
    js = r.json()
    if not js: raise ValueError("Geocoding failed for both Google and OSM")
    lat, lon = float(js[0]['lat']), float(js[0]['lon'])
    return js[0]['display_name'], (lat, lon)

@cached(api_cache, lock=_cache_lock)
def reverse_geocode(lat: float, lon: float, ui_lang: str):
    lang_code = LANG_MAP.get(ui_lang, "zh-TW")
    if HAS_GMAP:
        try:
            params = {"latlng": f"{lat},{lon}", "key": GOOGLE_KEY, "language": lang_code,
                      "result_type": "street_address|route|political"}
            r = GLOBAL_SESSION.get("https://maps.googleapis.com/maps/api/geocode/json", params=params, headers=UA, timeout=5)
            r.raise_for_status()
            js = r.json()
            if js['status'] == 'OK' and js.get('results'):
                return js['results'][0]['formatted_address']
        except requests.exceptions.RequestException as e:
            logging.warning(f"Google Reverse Geocode failed, falling back to OSM: {e}")
    r = GLOBAL_SESSION.get("https://nominatim.openstreetmap.org/reverse",
                     params={"lat": lat, "lon": lon, "format": "jsonv2", "accept-language": lang_code, "zoom": 16},
                     headers=UA, timeout=5)
    r.raise_for_status()
    return r.json().get('display_name')


# ---- Geo helpers for route densify & offsets ----
def _haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0088
    from math import radians, sin, cos, asin, sqrt
    dlat = radians(lat2 - lat1); dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1))*cos(radians(lat2))*sin(dlon/2)**2
    return 2 * R * asin(sqrt(a))

def _bearing_deg(lat1, lon1, lat2, lon2):
    from math import radians, degrees, atan2, sin, cos
    y = sin(radians(lon2 - lon1)) * cos(radians(lat2))
    x = cos(radians(lat1)) * sin(radians(lat2)) - sin(radians(lat1)) * cos(radians(lat2)) * cos(radians(lon2 - lon1))
    brng = degrees(atan2(y, x))
    return (brng + 360) % 360

def _offset_meters(lat, lon, dx_m, dy_m):
    """在當地切線座標系偏移：x=東向(經度)，y=北向(緯度)"""
    R = 6378137.0
    dlat = dy_m / R
    dlon = dx_m / (R * math.cos(math.radians(lat)))
    return lat + math.degrees(dlat), lon + math.degrees(dlon)

def _perp_offsets(lat, lon, brng_deg, offsets_m):
    """依路線方位的法向，在左右做偏移"""
    # 路線方向方位角 brng；其法向 = brng ± 90°
    perp_rad = math.radians(brng_deg + 90.0)
    ux, uy = math.cos(perp_rad), math.sin(perp_rad)  # 東向、北向單位向量
    pts = []
    for off in offsets_m:
        dx = ux * off
        dy = uy * off
        pts.append(_offset_meters(lat, lon, dx, dy))
    return pts
# ===== 路線輔助 =====
def _decode_polyline(polyline_str):
    index, lat, lng = 0, 0, 0
    coordinates = []
    changes = {'latitude': 0, 'longitude': 0}
    while index < len(polyline_str):
        for unit in ['latitude', 'longitude']:
            shift, result = 0, 0
            while True:
                byte = ord(polyline_str[index]) - 63
                index += 1
                result |= (byte & 0x1f) << shift
                shift += 5
                if not byte >= 0x20: break
            if (result & 1): changes[unit] = ~(result >> 1)
            else: changes[unit] = (result >> 1)
        lat += changes['latitude']
        lng += changes['longitude']
        coordinates.append((lat / 100000.0, lng / 100000.0))
    return coordinates

def _parse_gmap_route(route_json):
    coords = _decode_polyline(route_json['overview_polyline']['points'])
    leg = route_json['legs'][0]
    return {"lats": [c[0] for c in coords], "lons": [c[1] for c in coords],
            "duration": leg['duration']['value'], "distance": leg['distance']['value'] / 1000,  # km
            "summary": route_json.get('summary', '')}

def google_routes_with_alts(o: tuple, d: tuple, mode: str, lang: str):
    if not HAS_GMAP: return []
    try:
        gmap_mode_map = {"drive": "driving", "scooter": "two_wheeler", "walk": "walking", "transit": "transit"}
        params = {"origin": f"{o[0]},{o[1]}", "destination": f"{d[0]},{d[1]}", "key": GOOGLE_KEY,
                  "language": LANG_MAP.get(lang, "zh-TW"), "mode": gmap_mode_map.get(mode, "driving"),
                  "alternatives": "true"}
        if mode == "scooter": params["avoid"] = "highways"
        r = GLOBAL_SESSION.get("https://maps.googleapis.com/maps/api/directions/json", params=params, headers=UA, timeout=8)
        r.raise_for_status()
        return [_parse_gmap_route(route) for route in r.json().get('routes', [])]
    except requests.exceptions.RequestException as e:
        logging.error(f"Google route failed: {e}")
        return []

def osrm_route(o: tuple, d: tuple, mode: str):
    profile_map = {"drive":"driving", "walk":"walking", "scooter":"driving"}
    profile = profile_map.get(mode, "driving")
    coords = f"{o[1]},{o[0]};{d[1]},{d[0]}"
    url = f"http://router.project-osrm.org/route/v1/{profile}/{coords}?overview=full&geometries=polyline"
    if mode == "scooter": url += "&exclude=motorway"
    try:
        r = GLOBAL_SESSION.get(url, headers=UA, timeout=8)
        r.raise_for_status()
        js = r.json()
        if js.get('code') == 'Ok' and js.get('routes'):
            route = js['routes'][0]
            coords = _decode_polyline(route['geometry'])
            return [{"lats": [c[0] for c in coords], "lons": [c[1] for c in coords],
                     "duration": route['duration'], "distance": route['distance'] / 1000, "summary": "OSRM Route"}]
    except requests.exceptions.RequestException as e:
        logging.error(f"OSRM route failed: {e}")
    return []

def segments_by_flags(lats, lons, flags):
    if not flags: return []
    segments, current_segment = [], []
    current_wet_status = flags[0]
    for i in range(len(lats)):
        current_segment.append((lats[i], lons[i]))
        if i == len(lats) - 1 or flags[i] != flags[i+1]:
            segments.append({"points": current_segment, "is_wet": current_wet_status})
            current_segment = [(lats[i], lons[i])]
            if i < len(lats) - 1:
                current_wet_status = flags[i+1]
    return segments

def bbox_center(lats, lons):
    if not lats or not lons: return FALLBACK_CENTER, 10
    min_lat, max_lat, min_lon, max_lon = min(lats), max(lats), min(lons), max(lons)
    center = [(min_lat + max_lat) / 2, (min_lon + max_lon) / 2]
    lat_diff, lon_diff = max_lat - min_lat, max_lon - min_lon
    if lat_diff < 1e-6 or lon_diff < 1e-6: return center, 15
    try:
        zoom_lon = math.log2(360 / lon_diff)
        zoom_lat = math.log2(180 / lat_diff)
        zoom = min(zoom_lon, zoom_lat)
    except (ValueError, ZeroDivisionError):
        zoom = 10
    return center, max(4, min(16, zoom))

def _get_radius_for_zoom(zoom: float) -> int:
    z = float(zoom or 7.0)
    if z >= 11.0: return 95
    anchors = [(4.0, 25), (5.0, 30), (6.0, 35), (7.0, 40), (8.0, 50), (9.0, 65), (10.0, 80), (11.0, 95)]
    for (z0, r0), (z1, r1) in zip(anchors, anchors[1:]):
        if z0 <= z <= z1:
            return int(round(r0 + (z - z0) / (z1 - z0) * (r1 - r0)))
    return 40

def _bounds_from_center_zoom(lat: float, lon: float, zoom: int | float):
    z = int(zoom or 13)
    e_map = {14:0.1, 13:0.15, 12:0.25, 11:0.4, 10:0.6}
    e = e_map.get(z, 0.9)
    return [[lat - e, lon - e], [lat + e, lon + e]]


def _densify_polyline(lats: list, lons: list, step_km: float) -> tuple[list, list]:
    if not lats or not lons or len(lats) != len(lons):
        return lats, lons
    out_lat, out_lon = [lats[0]], [lons[0]]
    for i in range(1, len(lats)):
        la1, lo1, la2, lo2 = lats[i-1], lons[i-1], lats[i], lons[i]
        seg_len = _haversine_km(la1, lo1, la2, lo2)
        if seg_len <= 1e-6:
            continue
        n = max(1, int(math.floor(seg_len / step_km)))
        for j in range(1, n+1):
            t = j / n
            out_lat.append(la1 + (la2 - la1) * t)
            out_lon.append(lo1 + (lo2 - lo1) * t)
    return out_lat, out_lon

def _majority_smooth(flags: list, win: int) -> list:
    if not flags or win <= 1 or win % 2 == 0:
        return flags
    k = win // 2
    out = []
    for i in range(len(flags)):
        s = 0; c = 0
        for j in range(i-k, i+k+1):
            if 0 <= j < len(flags):
                # 加權：中心權重高，邊緣低
                weight = 2 if j == i else 1
                s += weight if flags[j] else 0
                c += weight
        out.append(s * 2 >= c)  # 調整閾值保留尖峰
    return out

def route_rain_flags_concurrent(lats: list, lons: list, mode: str):
    """回傳 ((flags, codes), mm_list)。flags 已平滑。自適應 step 和 threshold + 二階段掃描 + 黏滯快取。"""
    if not lats or not lons:
        return ([], []), []
    # 自適應步長：依總長度
    total_len = _haversine_km(lats[0], lons[0], lats[-1], lons[-1])
    step_km = max(0.2, min(ROUTE_SAMPLE_STEP_KM, total_len / 50))  # 長路徑細採樣
    if total_len >= 150:
        step_km = max(step_km, min(2.0, total_len / 120))  # 超長路線放大步長（上限 2km）
    dlats, dlons = _densify_polyline(lats, lons, step_km)

    # 自適應 threshold：步行/機車更敏感
    if mode == "walk":
        thresh = DECISION_ROUTE_MM_MIN * 0.7
    elif mode == "scooter":
        thresh = DECISION_ROUTE_MM_MIN * 0.8
    else:
        thresh = DECISION_ROUTE_MM_MIN * 1.0

    # 估計速度 (km/h) 基於模式
    avg_speed = 60 if mode == "drive" else 40 if mode == "scooter" else 5 if mode == "walk" else 30  # 預設

    cumulative_time = [0]  # 分鐘
    for i in range(1, len(dlats)):
        seg_len = _haversine_km(dlats[i-1], dlons[i-1], dlats[i], dlons[i])
        seg_time = seg_len / avg_speed * 60  # 分鐘
        cumulative_time.append(cumulative_time[-1] + seg_time)

    # ---- 二階段掃描：Coarse → Refine（降低長途效能瓶頸） ----
    coarse_idx = []
    coarse_step_km = 5.0 if total_len >= 150 else 0.0
    if coarse_step_km > 0:
        acc = 0.0
        coarse_idx.append(0)
        for i in range(1, len(dlats)):
            dkm = _haversine_km(dlats[i-1], dlons[i-1], dlats[i], dlons[i])
            acc += dkm
            if acc >= coarse_step_km:
                coarse_idx.append(i)
                acc = 0.0
        if coarse_idx[-1] != len(dlats) - 1:
            coarse_idx.append(len(dlats) - 1)

        # 快速偵測「可能有雨」區段（只看中心點；保守 0.6*thresh）
        coarse_wet = set()
        for i in coarse_idx:
            la, lo = dlats[i], dlons[i]
            future_hour = int(cumulative_time[i] / 60)
            mm, code, _is_visual = get_rain_mm_hybrid(la, lo, future_hour)[:3]
            mm = mm or 0.0
            if (mm >= 0.6 * thresh) or (THUNDER_MIN <= code <= THUNDER_MAX):
                coarse_wet.add(i)

        # 根據 coarse 濕點，建立需要精掃的索引集合（±8km 範圍）
        refine_need = set()
        if coarse_wet:
            radius_km = 8.0
            for cw in coarse_wet:
                # 往左擴
                acc = 0.0
                j = cw
                while j > 0 and acc <= radius_km:
                    refine_need.add(j)
                    acc += _haversine_km(dlats[j-1], dlons[j-1], dlats[j], dlons[j])
                    j -= 1
                # 往右擴
                acc = 0.0
                j = cw
                while j < len(dlats)-1 and acc <= radius_km:
                    refine_need.add(j)
                    acc += _haversine_km(dlats[j], dlons[j], dlats[j+1], dlons[j+1])
                    j += 1
        else:
            # coarse 全乾：直接返回全乾結果（極大幅降載）
            flags = [False] * len(dlats)
            codes = [0] * len(dlats)
            mm_list = [0.0] * len(dlats)
            flags = _majority_smooth(flags, ROUTE_SMOOTH_WIN)
            return (flags, codes), mm_list
    else:
        refine_need = set(range(len(dlats)))  # 普通路線：全部精掃

    def sample_one(idx):
        la, lo = dlats[idx], dlons[idx]
        i0 = max(0, idx-1); i1 = min(len(dlats)-1, idx+1)
        brng = _bearing_deg(dlats[i0], dlons[i0], dlats[i1], dlons[i1])
        pts = _perp_offsets(la, lo, brng, ROUTE_BUFFER_OFFSETS_M)
        best_mm = 0.0; best_code = 0
        future_hour = int(cumulative_time[idx] / 60)  # 未來小時
        # 若不在 refine 範圍，僅中心點快速判斷
        if idx not in refine_need:
            mm, code, _ = get_rain_mm_hybrid(la, lo, future_hour)[:3]
            best_mm, best_code = (mm or 0.0), (code or 0)
        else:
            for pla, plo in pts:
                mm, code, _is_rain = get_rain_mm_hybrid(pla, plo, future_hour)[:3]
                if mm is None:
                    continue
                if mm > best_mm:
                    best_mm, best_code = mm, code
        is_wet = (best_mm >= thresh)
        return (is_wet, best_code, best_mm)

    flags = []; codes = []; mm_list = []
    with ThreadPoolExecutor(max_workers=12) as ex:
        for wet, code, mm in ex.map(sample_one, range(len(dlats))):
            flags.append(wet); codes.append(code); mm_list.append(mm)

    # 3) 加權平滑
    flags = _majority_smooth(flags, ROUTE_SMOOTH_WIN)
    return (flags, codes), mm_list

# ===== App Layout & Callbacks =====
app = Dash(__name__, title="即時雨區＋路線規劃", suppress_callback_exceptions=True, assets_folder=ASSETS_DIR)
server = app.server
initial_figure = base_map_figure()

app.layout = html.Div([
    dcc.Store("lang-store", data="zh"), dcc.Store("mode-store", data="explore"),
    dcc.Store("route-store", data={}), dcc.Store("view-store", data={"center": BASE_CENTER, "zoom": BASE_ZOOM}),
    dcc.Store("geo-store"), dcc.Store("i18n-ts-prefix"), dcc.Store("status-store", data={}),
    dcc.Store("timestamp-store"), dcc.Store("rain-heatmap-store"), dcc.Store("panel-store", storage_type="local"),
    dcc.Store("explore-store", data={}), dcc.Store(id="coord-trigger-store"), dcc.Store(id="data-request-store"),
    dcc.Interval(id="background-scan-interval", interval=2000, n_intervals=0, disabled=True),  # 背景輕掃
    html.Button("≡", id="panel-toggle", n_clicks=0, className="panel-toggle-mobile"),
    html.Div(id="panel-scrim", className="panel-scrim", n_clicks=0),
    html.Div(id="panel", className="panel", children=[
        html.H2(id="ttl"), html.Button("🌐", id="btn-lang", className="globe"),
        html.Div(id="lang-menu", role="menu", className="menu hide", children=[
            html.Button("中文", id="lang-zh"), html.Button("English", id="lang-en"), html.Button("日本語", id="lang-ja"),
        ]),
        dcc.RadioItems(id="mode", value="explore", className="rad",
                       labelStyle={'display': 'inline-block', 'marginRight': '15px'}),
        html.Div(id="box-explore", children=[
            html.Div(className="input-row", children=[
                dcc.Input(id=SEARCH_INPUT_ID, className="input"),
                html.Button(id=BTN_SEARCH_ID, className="button"),
            ]),
            html.Div(className="row gap", children=[
                html.Button(id=BTN_AREA_ID, className="button link"),
                html.Button("定位", id=BTN_LOCATE_ID, className="button link"),
            ]),
        ]),
        html.Div(id="box-route", className="hide", children=[
            html.Div (className="row", children=[
                html.Span(id="lab-travel"),
                dcc.RadioItems(id="travel-mode", value="drive", className="rad",
                               labelStyle={'display': 'inline-block', 'marginRight': '10px'})
            ]),
            dcc.Input(id="src", className="input"),
            dcc.Input(id="dst", className="input"),
            html.Div(className="input-row", children=[
                html.Button(id="btn-plan", className="button", style={"flex": 1}),
                html.Button("📍", id="btn-locate-src", className="button link"),
            ]),
        ]),
        html.Hr(),
        dcc.RadioItems(id="basemap", value="low", className="rad",
                       labelStyle={'display': 'inline-block', 'marginRight': '15px'}),
        html.Div(id="addr-line", className="addr"),
        html.Div(id="alert", className="alert yellow hide"),
        html.Div(id="ts-line", className="ts"),
    ]),
    dcc.Loading(id="loading-main", type="circle",
                children=html.Div(id="loading-anchor", style={"display": "none"})),
    dcc.Graph(id=MAP_ID, style={"position":"fixed", "height":"100%", "width":"100%"},
              config={"scrollZoom": True, "displaylogo": False, "displayModeBar": False}, figure=initial_figure),
    html.Div(className="legend", id="legend-a", style={"display":"none"}, children=[
        html.Span(id="legend-title"), html.Div(id="legend-scale-container")
    ])
])

# --- 背景輕量掃描：監視 view 變化，大幅位移觸發 ---
@app.callback(
    Output("background-scan-interval", "disabled", allow_duplicate=True),
    Input("view-store", "data"),
    State("background-scan-interval", "disabled"),
    State("data-request-store", "data"),
    prevent_initial_call=True
)
def toggle_background_scan(view, disabled, last_req):
    if not view or disabled:
        return True
    # 大幅位移 (>5km) 且非最近掃描，啟動間隔
    if last_req and (time.time() - last_req.get("ts", 0)) < 10:
        return True
    # 簡化：每 2s 檢查一次，若位移大則觸發輕掃 (在 main_data_controller 中處理)
    return False

# --- 搜尋：Enter/按鈕觸發；只平移＋放標記，不掃描；並清空輸入框 ---
@app.callback(
    Output("coord-trigger-store", "data"),
    Output(SEARCH_INPUT_ID, "value", allow_duplicate=True),
    [Input(BTN_SEARCH_ID, "n_clicks"), Input(SEARCH_INPUT_ID, "n_submit")],
    [State(SEARCH_INPUT_ID, "value"), State("lang-store", "data"), State(MAP_ID, "relayoutData")],
    prevent_initial_call=True
)
def handle_search_geocoding(n_clicks, n_submit, q, lang, relayout_data):
    trig = ctx.triggered_id
    if trig not in (BTN_SEARCH_ID, SEARCH_INPUT_ID): raise PreventUpdate
    if not q: raise PreventUpdate
    try:
        bounds = _safe_bounds_from_relayout(relayout_data)
        addr, (lat, lon) = smart_geocode(q.strip(), LANG_MAP.get(lang, "zh-TW"), bounds)
        return {"coord": [lat, lon], "addr": addr, "source": "search_no_scan", "ts": time.time()}, ""
    except (TypeError, ValueError) as e:
        logging.error(f"Geocode failed for '{q}': {e}")
        return no_update, no_update

# --- 定位：只平移，不掃描 ---
app.clientside_callback(
    """
    function(n_clicks) {
        if (!n_clicks) return window.dash_clientside.no_update;
        const now = Date.now() / 1000.0;
        return new Promise(resolve => {
            const fallback = {coord: [%f, %f], addr: null, source: "locate_no_scan", is_fallback: true, ts: now};
            if (!navigator.geolocation) { resolve(fallback); return; }
            navigator.geolocation.getCurrentPosition(
                p => resolve({coord: [p.coords.latitude, p.coords.longitude], addr: null, source: "locate_no_scan", ts: now}),
                _ => resolve(fallback),
                {enableHighAccuracy: true, timeout: 3000, maximumAge: 120000}
            );
        });
    }
    """ % (FALLBACK_CENTER[0], FALLBACK_CENTER[1]),
    Output("coord-trigger-store", "data", allow_duplicate=True),
    Input(BTN_LOCATE_ID, "n_clicks"),
    prevent_initial_call=True
)

# --- 搜尋此區域（按下去才讀當前視窗；不即時同步 view-store） ---
@app.callback(
    Output("coord-trigger-store", "data", allow_duplicate=True),
    Input(BTN_AREA_ID, "n_clicks"),
    State(MAP_ID, "relayoutData"),
    State("view-store", "data"),
    State("lang-store", "data"),
    prevent_initial_call=True
)
def handle_search_area(n_clicks, relayout_data, view, lang):
    if not n_clicks:
        raise PreventUpdate

    # 冷卻
    with _last_scan_lock:
        recent = (time.time() - LAST_SCAN_TS) < SCAN_COOLDOWN_SEC
    if recent:
        raise PreventUpdate

    center, zoom = _center_zoom_from_relayout(relayout_data, view)
    derived = _safe_bounds_from_relayout(relayout_data)
    if derived:
        bounds = [[derived["south"], derived["west"]], [derived["north"], derived["east"]]]
    else:
        bounds = _bounds_from_center_zoom(center[0], center[1], zoom)

    try:
        addr = _normalize_addr(reverse_geocode(center[0], center[1], lang) or t(lang, "map_center"))
    except Exception:
        addr = t(lang, "map_center")

    return {
        "coord": center,
        "addr": addr,
        "source": "area_scan",
        "zoom": zoom,
        "bounds": bounds,
        "ts": time.time()
    }

# --- 立即平移／標記；只有 area_scan 會丟到後端掃描 ---
@app.callback(
    Output("view-store", "data", allow_duplicate=True),
    Output("explore-store", "data", allow_duplicate=True),
    Output("data-request-store", "data"),
    Input("coord-trigger-store", "data"), prevent_initial_call=True
)
def immediate_pan_and_trigger_backend(trig):
    if not trig or not trig.get("coord"): raise PreventUpdate
    (lat, lon), src = trig["coord"], trig.get("source")
    zoom = LOCATE_ZOOM if src and src.startswith("locate") else SEARCHED_ZOOM
    view_out = {"center": [lat, lon], "zoom": zoom}
    if src == "area_scan":
        explore_tmp = {"coord": [lat, lon], "addr": None, "label": t("zh", "searching"), "ts": trig.get("ts")}
        return view_out, explore_tmp, trig
    else:
        explore_tmp = {"coord": [lat, lon], "addr": trig.get("addr"), "label": None, "ts": trig.get("ts")}
        return view_out, explore_tmp, no_update

# --- 非掃描事件：用 Hybrid 覆蓋「多雲」誤差（更貼近此刻） ---
@app.callback(
    Output("status-store", "data", allow_duplicate=True),
    Output("timestamp-store", "data", allow_duplicate=True),
    Input("coord-trigger-store", "data"),
    State("lang-store", "data"),
    prevent_initial_call=True
)
def update_status_on_pan(trig, lang):
    if not trig or not trig.get("coord"): raise PreventUpdate
    if trig.get("source") == "area_scan":
        raise PreventUpdate
    lat, lon = trig["coord"]
    raw_addr = trig.get("addr") or reverse_geocode(lat, lon, lang) or f"({lat:.4f}, {lon:.4f})"
    addr = _normalize_addr(raw_addr)

    forecast = get_point_forecast(lat, lon, lang)
    off = int((forecast.get("offset_sec") if isinstance(forecast, dict) else 28800) or 28800)
    ts_out = datetime.now(timezone(timedelta(seconds=off))).strftime("%H:%M:%S")

    # 若 Hybrid 顯示有雨（≥0.1），覆蓋文字的天氣鍵
    try:
        mm_now, _, _, _ ,_ = get_rain_mm_hybrid(lat, lon)
        if (mm_now is not None) and (mm_now >= 0.1):
            key = "heavy_rain" if mm_now >= 3.0 else "lightrain"
            forecast = dict(forecast or {})
            forecast["key"] = key
            forecast["rain_mm"] = mm_now  # 加入雨量
    except Exception:
        pass

    status_out = {"type": "explore", "data": {"addr": addr, **(forecast if isinstance(forecast, dict) else {})}, "ts": trig.get("ts")}
    return status_out, ts_out

# --- 後端資料控制（僅 area_scan 會進來） + 背景輕掃 + QPF 訊息 + 加權中心 ---
@app.callback(
    [Output("status-store", "data", allow_duplicate=True),
     Output("timestamp-store", "data", allow_duplicate=True),
     Output("rain-heatmap-store", "data"),
     Output("explore-store", "data", allow_duplicate=True),
     Output("loading-anchor", "children")],
    Input("data-request-store", "data"),
    [State("lang-store", "data"), State("view-store", "data"), State("explore-store", "data")],
    prevent_initial_call=True
)
def main_data_controller(trig, lang, view, explore_data):
    if not trig or not trig.get("coord"): raise PreventUpdate
    (lat, lon), source, ts = trig["coord"], trig.get("source"), trig.get("ts")

    raw_addr = reverse_geocode(lat, lon, lang) or t(lang, "map_center")
    addr = _normalize_addr(raw_addr)

    wx_points, timed_out = [], False
    qpf_msg = ""

    if source == 'area_scan':
        zoom = trig.get("zoom", (view or {}).get("zoom", SEARCHED_ZOOM))
        tbounds = trig.get("bounds")
        if isinstance(tbounds, list) and len(tbounds) == 2 and len(tbounds[0]) == 2 and len(tbounds[1]) == 2:
            bounds = tbounds
        else:
            bounds = _bounds_from_center_zoom(lat, lon, zoom)

        qpf_rain = get_cwa_qpf_forecast()
        if qpf_rain is False:
            # —— 全國預報無雨：顯示簡化文案，不進行掃描（沿用 partial 旗標）
            qpf_msg = t(lang, "skip_scan_sunny")
            timed_out = True
        else:
            # —— 有雨預報：同時檢查本點未來幾小時是否真的可能下雨／雷雨
            try:
                # 用 Open-Meteo 逐小時，precip[1:] 是未來時段（precip[0] 已是「此刻內插」）
                _, precip, codes, _, idx, offset = _om_hourly_forecast_data(lat, lon)
                will_rain_soon = any(p > 0.1 for p in (precip[1:4] if precip else []))  # 未來 3 小時
                thunder_soon = any(
                    THUNDER_MIN <= codes[min(idx + i, len(codes) - 1)] <= THUNDER_MAX
                    for i in range(1, 4)
                ) if codes else False
                local_now = datetime.now(timezone(timedelta(seconds=offset)))
                is_afternoon = 12 <= local_now.hour <= 18

                # 先照常掃描地圖熱區（維持原有功能）
                wx_points, timed_out = get_weather_data_for_bounds(bounds, zoom)
                if wx_points:
                    lat, lon = weighted_rain_centroid(wx_points, lat, lon)
                    view["center"] = [lat, lon]

                # 只有在「此座標未來幾小時真的可能下雨」時才顯示 QPF 提示；
                # 且若偵測到雷雨碼，顯示「午後雷陣雨/雷雨」字樣。
                if will_rain_soon:
                    qpf_msg = _qpf_localized_msg(lang, thunder_soon, is_afternoon)
                    # 若掃描逾時，可在訊息後補一個（部分結果）的標註（可選）
                    if timed_out:
                        qpf_msg += t(lang, "partial_result") if t(lang, "partial_result") else " (部分結果)"
                else:
                    qpf_msg = ""  # 本點短期不太會下雨：不顯示 QPF 提示

            except Exception as e:
                logging.error(f"Scan failed: {e}")
                wx_points, timed_out = [], True
                # 掃描失敗時不要強行顯示 QPF 提示，避免誤導
                qpf_msg = ""

    elif source == 'background_light':
        zoom_now = trig.get("zoom", (view or {}).get("zoom", SEARCHED_ZOOM))
        bounds = _bounds_from_center_zoom(lat, lon, zoom_now)
        try:
            wx_points, timed_out = get_weather_data_for_bounds(bounds, zoom_now, timeout_sec=3.0)
        except Exception:
            wx_points = []

    forecast = get_point_forecast(lat, lon, lang)

    # 文字摘要的小範圍（依 zoom 挑半徑）；有點才做
    area_summary = None
    if source == 'area_scan' and wx_points:
        zoom_now = trig.get("zoom", (view or {}).get("zoom", SEARCHED_ZOOM))
        if zoom_now >= 13:   rkm = AREA_SUMMARY_RADIUS_KM_CITY
        elif zoom_now >= 11: rkm = AREA_SUMMARY_RADIUS_KM_SUBURB
        else:                rkm = AREA_SUMMARY_RADIUS_KM_RURAL
        area_summary = summarize_small_area_km(lat, lon, wx_points, rkm)

    off = int((forecast.get("offset_sec") if isinstance(forecast, dict) else 28800) or 28800)
    ts_out = datetime.now(timezone(timedelta(seconds=off))).strftime("%H:%M:%S")

    payload = {"addr": addr, **(forecast if isinstance(forecast, dict) else {})}
    if qpf_msg:
        payload["qpf_msg"] = qpf_msg
    if area_summary is not None:
        payload["area"] = area_summary
    payload["lat"] = lat
    payload["lon"] = lon

    status_out = {"type": "explore", "data": payload, "ts": ts}

    final_marker = (explore_data or {}).copy()
    final_marker.update({"coord": [lat, lon], "addr": addr, "label": None, "ts": ts})

    global LAST_SCAN_TS
    with _last_scan_lock:
        LAST_SCAN_TS = time.time()

    return status_out, ts_out, wx_points, final_marker, ""

# --- 地圖繪製 ---
@app.callback(
    Output(MAP_ID, "figure"), Output("legend-a", "style"), Output("legend-scale-container", "children"),
    Input("basemap", "value"), Input("mode-store", "data"), Input("view-store", "data"),
    Input("explore-store", "data"), Input("rain-heatmap-store", "data"),
    Input("route-store", "data"), Input("lang-store", "data"),
    prevent_initial_call=True
)
def unified_draw_map(basemap, mode, view, explore_data, rain_points, route_data, lang):
    center, zoom = (view or {}).get("center", FALLBACK_CENTER), (view or {}).get("zoom", SEARCHED_ZOOM)
    style = "carto-positron" if (basemap or "low") == "low" else "open-street-map"
    fig = base_map_figure(center=center, zoom=zoom, style=style)
    legend_style, legend_children = {"display": "none"}, []

    if mode == "explore":
        if rain_points:
            lats, lons, vals = zip(*rain_points)
            fig.add_trace(go.Scattermapbox(
                lat=lats, lon=lons, mode='markers',
                marker=dict(
                    size=max(25, _get_radius_for_zoom(zoom) * 0.8),
                    color=vals, colorscale=RAIN_CIRCLE_COLORSCALE,
                    cmin=0, cmax=HEATMAP_MAX_MM, opacity=0.65, allowoverlap=True
                ),
                hoverinfo='none', name="Rain"
            ))
            legend_style, legend_children = {"display":"flex"}, [
                html.Div(className="legend-scale", children=[html.Span(t(lang, "legend_light")),
                                                             html.Span(t(lang, "legend_heavy"))]),
                html.Div(className="legend-bar")
            ]
        if explore_data and explore_data.get("coord"):
            (lt, ln), label = explore_data["coord"], explore_data.get("label")
            fig.add_trace(go.Scattermapbox(
                lat=[lt], lon=[ln],
                mode="markers+text" if label else "markers",
                text=[label] if label else None, textposition="top center",
                marker=dict(size=14 if label else 16,
                            color="rgba(255,159,64,0.95)" if label else "rgba(239,68,68,0.95)"),
                hovertext=explore_data.get("addr"),
                hoverinfo="text" if not label else "skip", name="Marker"
            ))

    if mode == "route" and route_data:
        segments = segments_by_flags(route_data['lats'], route_data['lons'], route_data['rain_flags'])
        for seg in segments:
            if not seg['points'] or len(seg['points']) < 2: continue
            lats, lons = zip(*seg['points'])
            fig.add_trace(go.Scattermapbox(
                lat=lats, lon=lons, mode='lines',
                line=dict(width=6, color=COLOR_WET if seg['is_wet'] else COLOR_DRY),
                opacity=0.9, hoverinfo='none'
            ))
        fig.add_trace(go.Scattermapbox(
            lat=[route_data['lats'][0], route_data['lats'][-1]],
            lon=[route_data['lons'][0], route_data['lons'][-1]],
            mode='markers+text', text=[t(lang, 'origin'), t(lang, 'dest')], textposition='top right',
            marker=dict(size=14, color="#10B981")
        ))
        legend_style, legend_children = {"display":"flex"}, [
            html.Div(className="legend-scale-route", children=[
                html.Div(className="swatch", style={"backgroundColor": COLOR_DRY}),
                html.Span(t(lang, "dry")),
                html.Div(className="swatch", style={"backgroundColor": COLOR_WET}),
                html.Span(t(lang, "rain")),
            ])
        ]

    return fig, legend_style, legend_children

# --- 狀態文本 + 口語簡化 ---
@app.callback(
    Output("addr-line", "children"), Output("alert", "children"), Output("alert", "className"),
    Input("status-store", "data"), Input("lang-store", "data")
)
def update_status_text(status, lang):
    status = status or {}
    stype, data = status.get("type"), status.get("data", {})
    if not stype or not data: return "", "", "alert hide"

    if stype == "explore":
        addr_val = _normalize_addr(data.get("addr", ""))
        addr_text = f"📍 {addr_val}"
        temp_v = data.get("temp", None)
        temp_str = f"{round(temp_v)}°C" if isinstance(temp_v, (int, float)) else ""
        parts = [t(lang, data.get("key", "cloudy")), temp_str, data.get("forecast", "")]
        # 雷雨提醒
        parts.append(data.get("thunder_reminder", ""))
        # QPF 訊息
        qpf = data.get("qpf_msg", "")
        if qpf:
            parts.append(qpf)

        parts = [p for p in parts if p]
        alert_text = " | ".join(parts)
        return addr_text, alert_text, "alert yellow"

    if stype == "route":
        o_addr, d_addr = data.get('o_addr',''), data.get('d_addr','')
        addr_text = f"📍 {t(lang, 'addr_fixed')}：{o_addr} → {d_addr}"
        d_temp_v = data.get("d_temp", None)
        d_temp_str = f"{round(d_temp_v)}°C" if isinstance(d_temp_v, (int, float)) else ""
        dest_parts = [t(lang, data.get('d_lvl_key','cloudy')), d_temp_str]
        dest_str = f" // {t(lang,'dest_now')}：{' '.join(filter(None, dest_parts))}"
        dur = round(data.get('duration', 0) / 60)  # 分鐘
        dist = round(data.get('distance', 0), 1)  # km
        alert_text = f"{data.get('prefix','')}{t(lang,'best')} {data.get('risk',0)}% | 預計 {dur} 分鐘 / {dist} km {dest_str}"
        return addr_text, alert_text, "alert blue"

    return "", "", "alert hide"

@app.callback(
    Output("ts-line", "children"),
    Input("timestamp-store", "data"), Input("i18n-ts-prefix", "data")
)
def update_timestamp_text(ts, prefix):
    return f"{prefix} {ts}" if ts and prefix else ""

# --- 語言切換 ---
@app.callback(
    Output("lang-store", "data"), Output("lang-menu", "className"), Output("btn-lang", "aria-expanded"),
    Input("btn-lang", "n_clicks"), Input("lang-zh", "n_clicks"), Input("lang-en", "n_clicks"), Input("lang-ja", "n_clicks"),
    State("lang-store", "data"), State("lang-menu", "className"),
    prevent_initial_call=True
)
def lang_control(*args):
    trig = ctx.triggered_id
    cur_lang, klass = args[-2:]
    if trig == "btn-lang":
        is_opening = "hide" in (klass or "")
        return cur_lang, "menu" if is_opening else "menu hide", str(is_opening).lower()
    lang_map = {"lang-zh": "zh", "lang-en": "en", "lang-ja": "ja"}
    return lang_map.get(trig, cur_lang), "menu hide", "false"

# --- I18N 文案 ---
@app.callback(
    [Output("ttl", "children"), Output("mode", "options"), Output("q", "placeholder"), Output("btn-search", "children"),
     Output("btn-locate", "children"), Output("btn-area", "children"), Output("lab-travel", "children"),
     Output("travel-mode", "options"), Output("src", "placeholder"), Output("dst", "placeholder"),
     Output("btn-plan", "children"), Output("basemap", "options"), Output("i18n-ts-prefix", "data"),
     Output("legend-title", "children"), Output("btn-locate-src", "title")],
    Input("lang-store", "data"),
)
def update_i18n_text(lang):
    travel_opts = [{"label": t(lang, "drive"), "value": "drive"}, {"label": t(lang, "walk"), "value": "walk"}]
    if HAS_GMAP:
        travel_opts.insert(1, {"label": t(lang, "scooter"), "value": "scooter"})
        travel_opts.append({"label": t(lang, "transit"), "value": "transit"})
    return [t(lang, "panel_title"),
            [{"label": t(lang, "mode_explore"), "value": "explore"},
             {"label": t(lang, "mode_route"), "value": "route"}],
            t(lang, "placeholder_q"), t(lang, "search"), t(lang, "locate"), t(lang, "search_area"),
            t(lang, "travel_mode"), travel_opts, t(lang, "placeholder_src"), t(lang, "placeholder_dst"),
            t(lang, "plan"), [{"label": t(lang, "low"), "value": "low"},
                              {"label": t(lang, "osm"), "value": "osm"}],
            t(lang, "update"), t(lang, "legend_rain"), t(lang, "locate_src_title")]

# --- 模式切換：清空各 store ---
@app.callback(
    [Output("mode-store", "data"), Output("box-explore", "className"), Output("box-route", "className"),
     Output("rain-heatmap-store", "data", allow_duplicate=True), Output("explore-store", "data", allow_duplicate=True),
     Output("route-store", "data", allow_duplicate=True), Output("status-store", "data", allow_duplicate=True),
     Output("timestamp-store", "data", allow_duplicate=True), Output("q", "value"), Output("src", "value"), Output("dst", "value")],
    Input("mode", "value"),
    prevent_initial_call=True
)
def on_mode(m):
    clear_data = ([], {}, {}, {}, None, "", "", "")
    return (m, "" if m == "explore" else "hide", "hide" if m == "explore" else "") + clear_data

# --- 定位起點（路線） ---
app.clientside_callback(
    "n => n ? new Promise(r => navigator.geolocation.getCurrentPosition(p => r({lat:p.coords.latitude, lon:p.coords.longitude}), e => r({error:e.message}))) : window.dash_clientside.no_update",
    Output("geo-store", "data"), Input("btn-locate-src", "n_clicks"), prevent_initial_call=True
)

@app.callback(
    Output("src", "value", allow_duplicate=True),
    Input("geo-store", "data"), State("lang-store", "data"), prevent_initial_call=True
)
def fill_src_from_locate(geo, lang):
    if geo and not geo.get("error"):
        if addr := reverse_geocode(geo["lat"], geo["lon"], lang): return addr
    return no_update

# --- 側欄（手機） ---
app.clientside_callback(
    """
    function(nBtn, nScrim, ui){
      const trig = (window.dash_clientside.callback_context.triggered[0]||{}).prop_id || "";
      let open = (ui && ui.panel === "open");
      if (trig.startsWith("panel-toggle")) open = !open;
      else if (trig.startsWith("panel-scrim")) open = false;
      const storeOut = (trig) ? {panel: open ? "open" : "closed"} : window.dash_clientside.no_update;
      if (document.body) document.body.classList.toggle("lock-scroll", open && window.matchMedia("(max-width: 768px)").matches);
      return [open ? "panel" : "panel panel-hide", open ? "panel-scrim show" : "panel-scrim", storeOut];
    }
    """,
    Output("panel", "className"), Output("panel-scrim", "className"), Output("panel-store", "data"),
    Input("panel-toggle", "n_clicks"), Input("panel-scrim", "n_clicks"), State("panel-store", "data"),
)

# --- 路線控制（成功後清空 src/dst） + 自適應 mode ---
@app.callback(
    [Output("route-store", "data", allow_duplicate=True),
     Output("status-store", "data", allow_duplicate=True),
     Output("timestamp-store", "data", allow_duplicate=True),
     Output("view-store", "data", allow_duplicate=True),
     Output("src", "value", allow_duplicate=True),
     Output("dst", "value", allow_duplicate=True)],
    [Input("btn-plan", "n_clicks"), Input("dst", "n_submit"), Input("src", "n_submit")],
    [State("src", "value"), State("dst", "value"), State("travel-mode", "value"), State("lang-store", "data")],
    prevent_initial_call=True,
)
def route_controller(_, __, ___, src, dst, travel, lang):
    if not (src and dst): raise PreventUpdate
    try:
        o_addr, o_coord = smart_geocode(src, lang); d_addr, d_coord = smart_geocode(dst, lang)
    except Exception as e:
        logging.error(f"Route geocoding failed: {e}")
        return no_update, {"type": "error", "data": {"key": "no_route", "mode": "route"}}, no_update, no_update, no_update, no_update

    d_forecast = get_point_forecast(d_coord[0], d_coord[1], lang)
    routes = google_routes_with_alts(o_coord, d_coord, travel, lang) or osrm_route(o_coord, d_coord, travel)
    if not routes:
        return {}, {"type": "error", "data": {"key": "no_route", "mode": "route"}}, no_update, no_update, no_update, no_update

    min_duration = min(r.get('duration', float('inf')) for r in routes)
    
    for r in routes:
        (flags, _codes), mm_list = route_rain_flags_concurrent(r['lats'], r['lons'], travel)  # 傳 mode
        r['rain_flags'] = list(flags)
        if flags:
            wet_ratio = sum(flags) / len(flags)
            wet_mms = [min(mm, ROUTE_INT_CAP) for mm, f in zip(mm_list, flags) if f]
            avg_int = (sum(wet_mms)/len(wet_mms)) if wet_mms else 0.0
            r['risk'] = min(1.0, wet_ratio * (avg_int / ROUTE_INT_CAP))
        else:
            r['risk'] = 0.0
        # 加入時間懲罰總分
        duration_penalty = (r.get('duration', float('inf')) / min_duration) * 0.5 if min_duration > 0 else 0
        r['total_score'] = r['risk'] + duration_penalty

    routes.sort(key=lambda r: r['total_score'])
    best = routes[0]

    ts_out = datetime.now(timezone(timedelta(seconds=(d_forecast.get("offset_sec") if isinstance(d_forecast, dict) else 28800)))).strftime("%H:%M:%S")
    prefix = f"{t(lang, 'best')} ({t(lang, 'others')}: {len(routes)-1}) " if len(routes) > 1 else ""
    status_out = {"type": "route", "data": {"o_addr": o_addr, "d_addr": d_addr,
        "risk": round(best.get("risk", 0) * 100), "prefix": prefix, "duration": best.get('duration', 0), "distance": best.get('distance', 0), **(d_forecast if isinstance(d_forecast, dict) else {})}}

    center, zoom = bbox_center(best['lats'], best['lons'])
    view_out = {"center": center, "zoom": zoom}

    return best, status_out, ts_out, view_out, "", ""

# ===== 入口 =====
if __name__ == "__main__":
    port = int(os.getenv("PORT", "8050"))
    app.run(host=os.getenv("HOST", "0.0.0.0"), port=port, debug=False)
