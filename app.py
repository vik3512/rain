#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# App Version: 7.6-final (修復熱力圖與提示條邏輯不一致)

import os, json, time, threading, math, logging, re
from datetime import datetime, timezone, timedelta
from typing import Tuple, Dict, Optional, List
from concurrent.futures import ThreadPoolExecutor, as_completed

from cachetools import cached
from cachetools import TTLCache as CacheToolsTTLCache

import requests
from dash import Dash, html, dcc, Input, Output, State, no_update, ctx
from dash.exceptions import PreventUpdate
import plotly.graph_objects as go

try:
    from dotenv import load_dotenv
    load_dotenv()
except:
    pass

# ===== Logging =====
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ===== I18N / Assets =====
ASSETS_DIR = os.path.join(os.path.dirname(__file__), "assets")
ASSETS_I18N = os.path.join(ASSETS_DIR, "i18n.json")
try:
    with open(ASSETS_I18N, "r", encoding="utf-8") as f:
        I18N = json.load(f)
except Exception as e:
    logging.warning(f"Load i18n.json failed: {e}; fallback to minimal zh placeholders.")
    I18N = {"zh": {
        "title":"即時雨區＋路線規劃",
        "panel_title": "控制台",
        "mode_explore":"定點雨量","mode_route":"路線雨量",
        "placeholder_q":"輸入地點","search":"搜尋","locate":"定位",
        "travel_mode":"交通方式","drive":"開車","scooter":"機車","walk":"步行","transit":"大眾運輸",
        "placeholder_src":"輸入出發地", "placeholder_dst":"輸入目的地", "plan":"規劃路線",
        "basemap":"底圖","low":"低飽和","osm":"標準 (OSM)","update":"更新於",
        "legend_rain":"降雨熱度","legend_light":"較弱","legend_heavy":"較強",
        "locate_src_title":"將起點設為目前位置",
        "map_center":"地圖中心","toast_err":"發生錯誤","loc_fail":"定位失敗","no_route":"找不到路線",
        "best":"最佳路線","others":"其他路線","origin":"起點","dest":"終點","dest_now":"目的地現在",
        "addr_fixed":"路線", "warn_thunder":"雷雨", "warn_heavy_rain":"大雨",
        "stops_in_1h":"約 1 小時後停", "stops_in_xh":"約 {} 小時後停", "starts_in_xh":"約 {} 小時後開始",
        "rain":"有雨", "lightrain":"小雨", "heavy_rain":"大雨/雷雨", "overcast":"陰", "cloudy":"多雲", "sunny":"晴",
        "search_area": "搜尋此區域",
        "searching": "搜尋中…",
        "no_gmap_key": "無 Google 路線 API 金鑰",
        "dry": "無雨"
    }}

def t(lang: str, key: str) -> str:
    lang_key = lang or "zh"
    return I18N.get(lang_key, I18N["zh"]).get(key, I18N["zh"].get(key, key))

# ===== 常數 =====
GOOGLE_KEY = os.getenv("GOOGLE_MAPS_API_KEY", "").strip()
GOOGLE_MAPS_API_KEY = GOOGLE_KEY
OWM_KEY    = os.getenv("OPENWEATHER_API_KEY", "").strip()
HAS_GMAP = bool(GOOGLE_KEY)
HAS_OWM  = bool(OWM_KEY)
UA = {"User-Agent": "rain-route-assistant/7.6-final"}
TW_BBOX = (118.2, 20.5, 123.5, 26.5)
INTERNATIONAL_KEYWORDS = ["澳洲","美國","日本","英國","法國","德國","中國","香港","澳門","韓國","加拿大","紐西蘭","泰國"]
LANG_MAP = {"zh":"zh-TW","en":"en","ja":"ja"}

_EXECUTOR = ThreadPoolExecutor(max_workers=16)

# ===== 地圖基礎 =====
BASE_CENTER = [23.9738, 120.9820]
BASE_ZOOM   = 7
SEARCHED_ZOOM = 13
DEFAULT_TILE_STYLE = "carto-positron"

# --- 熱力圖色階 (淺藍-深藍-紫色) ---
HEATMAP_COLORSCALE = [
    [0.0, "rgba(173, 216, 230, 0.0)"],
    [0.1, "rgba(173, 216, 230, 0.5)"],
    [0.2, "rgba(70, 130, 180, 0.7)"],
    [0.5, "rgba(0, 0, 128, 0.8)"],
    [0.8, "rgba(75, 0, 130, 0.9)"],
    [1.0, "rgba(128, 0, 128, 1.0)"]
]
HEATMAP_MAX_MM = 8.0

# --- 路線圖配色 ---
COLOR_DRY  = "rgba(16,185,129,0.95)"
COLOR_WET  = "rgba(37,99,235,0.85)"

def css_gradient_from_colorscale(colorscale):
    stops = []
    for frac, rgba in colorscale:
        pct = int(round(frac * 100))
        stops.append(f"{rgba} {pct}%")
    return "linear-gradient(90deg, " + ", ".join(stops) + ")"

def base_map_figure(center=BASE_CENTER, zoom=BASE_ZOOM, style=DEFAULT_TILE_STYLE):
    fig = go.Figure(go.Scattermapbox(
        lat=[center[0]],
        lon=[center[1]],
        mode='markers',
        marker=dict(size=0, opacity=0)
    ))

    fig.update_layout(
        mapbox=dict(style=style, center=dict(lat=center[0], lon=center[1]), zoom=zoom),
        margin=dict(l=0, r=0, t=0, b=0),
        uirevision="map",
        dragmode="pan",
        showlegend=False,
    )
    return fig

def clamp_view_to_tw(center: List[float], zoom: float):
    lat0 = float(center[0] if center else BASE_CENTER[0])
    lon0 = float(center[1] if center else BASE_CENTER[1])
    z0   = float(zoom or BASE_ZOOM)
    minLon, minLat, maxLon, maxLat = TW_BBOX
    inside_tw = (minLat <= lat0 <= maxLat) and (minLon <= lon0 <= maxLon)

    if inside_tw:
        lat = max(minLat, min(maxLat, lat0))
        lon = max(minLon, min(maxLon, lon0))
        z   = max(4.5, min(14.0, z0))
        return [lat, lon], z

    lat = max(-85.0, min(85.0, lat0))
    lon = max(-179.9, min(179.9, lon0))
    z   = max(2.5, min(14.0, z0))
    return [lat, lon], z

# ===== TTL 快取 (使用自訂類) =====
class TTLCache:
    def __init__(self, ttl=300, maxsize=1024):
        self.ttl, self.maxsize = ttl, maxsize
        self.data: Dict[str, Tuple[float, object]] = {}
        self.lock = threading.Lock()
    def _purge(self):
        now = time.time()
        expired_keys = [k for k, (ts, _) in self.data.items() if now - ts > self.ttl]
        for k_exp in expired_keys:
            self.data.pop(k_exp, None)
        while len(self.data) > self.maxsize:
            try:
                oldest = min(self.data.items(), key=lambda kv: kv[1][0])[0]
                self.data.pop(oldest, None)
            except ValueError:
                break
    def get(self, k):
        with self.lock:
            v = self.data.get(k)
            if not v: return None
            ts, val = v
            if time.time() - ts > self.ttl:
                self.data.pop(k, None); return None
            return val
    def set(self, k, val):
        with self.lock:
            if len(self.data) >= self.maxsize: self._purge()
            self.data[k] = (time.time(), val)

    def clear(self):
        with self.lock:
            self.data = {}

HOURLY_CACHE = TTLCache(maxsize=8000, ttl=3600)
def _current_hour_key_utc(): return datetime.utcnow().strftime("%Y-%m-%dT%H")

# ===== 核心天氣邏輯 =====
VISUAL_MM_MIN = 0.2
DECISION_MM_MIN = 0.8
RAIN_CODES = set(range(50,70)) | set(range(80,100))
THUNDER_MIN, THUNDER_MAX = 95, 99

def _quantize(v, q=0.05): return round(round(float(v)/q)*q, 5)
def _quantize_pair(lat, lon, q=0.05): return (_quantize(lat, q), _quantize(lon, q))

om_api_cache = CacheToolsTTLCache(maxsize=8000, ttl=3500)
@cached(om_api_cache)
def _om_hourly_forecast_data_cached_api(qlat: float, qlon: float, hour_key_utc: str):
    try:
        r = requests.get("https://api.open-meteo.com/v1/forecast",
                        params={"latitude": qlat, "longitude": qlon,
                                "hourly":"precipitation,weather_code",
                                "forecast_days":2,
                                "timezone":"auto"},
                        timeout=8)
        r.raise_for_status()
        js = r.json()
        h = js.get("hourly", {})
        times = h.get("time", [])
        precip = h.get("precipitation", [0.0])
        codes = h.get("weather_code", [0])
        utc_offset_sec = js.get("utc_offset_seconds", 0)
        now_with_tz = datetime.now(timezone(timedelta(seconds=utc_offset_sec)))
        now_key = now_with_tz.strftime("%Y-%m-%dT%H:00")

        idx = 0
        if now_key in times:
            idx = times.index(now_key)
        else:
            try:
                parsed_times = [datetime.fromisoformat(ts.replace("Z","")) for ts in times]
                now_local_hourly = now_with_tz.replace(minute=0, second=0, microsecond=0)
                idx = min(range(len(parsed_times)), key=lambda i: abs(parsed_times[i] - now_local_hourly))
            except Exception as e:
                logging.error(f"Failed to find closest time index: {e}")
                idx = 0
        return (times, precip, codes, idx, utc_offset_sec)
    except Exception as e:
        logging.error(f"OM Forecast fetch error: {e}")
        return ([], [0.0], [0], 0, 0)

# ===============================================================================
# ===== BEGIN: UPDATED/NEW WEATHER SCANNING FUNCTIONS ===========================
# ===============================================================================

# 新增 hour_key_override：讓同一小時內也能繞過 HOURLY_CACHE
def _om_hourly_forecast_data(lat: float, lon: float, hour_key_override: str | None = None):
    qlat, qlon = _quantize_pair(lat, lon, q=0.05)
    hour_key = hour_key_override or _current_hour_key_utc()
    memo_key = (qlat, qlon, hour_key)
    cached_data = HOURLY_CACHE.get(memo_key)
    if cached_data:
        return cached_data
    val = _om_hourly_forecast_data_cached_api(qlat, qlon, hour_key)
    HOURLY_CACHE.set(memo_key, val)
    return val

def get_weather_at_for_scan(lat, lon, hour_key_override: str | None = None):
    try:
        _, precip, codes, idx, _ = _om_hourly_forecast_data(lat, lon, hour_key_override)
        mm_now = float(precip[idx])
        code_om = int(codes[idx])
        is_visual_rain = (mm_now >= VISUAL_MM_MIN) or (THUNDER_MIN <= code_om <= THUNDER_MAX)
        return mm_now, code_om, is_visual_rain
    except Exception:
        return 0.0, 0, False

# ---- 9 點錨點：中心 + 四角 + 四邊中點 ----
def _anchor_points(bounds):
    (south, west), (north, east) = bounds
    c_lat = (south + north) / 2.0
    c_lon = (west + east) / 2.0
    return [
        (c_lat, c_lon),                                # center
        (south, west), (south, east), (north, west), (north, east),  # corners
        (south, c_lon), (north, c_lon), (c_lat, west), (c_lat, east) # edge midpoints
    ]

# （保留同名，預設 n=3 不變；下面會在呼叫處用 n=4）
def _coarse_centers(bounds, n=3):
    (south, west), (north, east) = bounds
    lat_step = (north - south) / n
    lon_step = (east  - west)  / n
    centers, cells = [], []
    for i in range(n):
        for j in range(n):
            clat = south + (i + 0.5) * lat_step
            clon = west  + (j + 0.5) * lon_step
            centers.append((clat, clon))
            cells.append(((south + i*lat_step, west + j*lon_step),
                          (south + (i+1)*lat_step, west + (j+1)*lon_step)))
    return centers, cells

# force=True：繞過快取 + 放寬一次錨點早退；錨點=9，粗掃=4x4=16
def get_weather_data_for_bounds(bounds: Tuple[Tuple[float, float], Tuple[float, float]], zoom: float, *, force: bool = False) -> List[Tuple[float, float, float]]:
    try:
        (south_lat, west_lon), (north_lat, east_lon) = bounds
    except (TypeError, ValueError):
        return []

    fine_step = _cap_step_for_points(south_lat, north_lat, west_lon, east_lon, _get_step_for_zoom(zoom), max_points=400)

    # ★ 快取繞過 key（force 時才帶秒級種子）
    hour_key_override = None
    if force:
        hour_key_override = f"{_current_hour_key_utc()}#{int(time.time())}"

    # ---- 改：9 錨點 ----
    anchor_points = _anchor_points(((south_lat, west_lon), (north_lat, east_lon)))
    try:
        is_any_anchor_wet = False
        for lat, lon in anchor_points:
            _, _, is_visual_rain = get_weather_at_for_scan(lat, lon, hour_key_override)
            if is_visual_rain:
                is_any_anchor_wet = True
                break
        if not is_any_anchor_wet:
            # 強制模式：即使錨點沒雨，也放寬一次，避免過度保守誤空回
            if not force:
                return []
    except Exception as e:
        logging.warning(f"Heatmap anchor scan failed: {e}")

    # ---- 改：粗掃 4×4 = 16 ----
    coarse_centers, coarse_cells = _coarse_centers(((south_lat, west_lon), (north_lat, east_lon)), n=4)
    suspected_cell_indices = []
    try:
        for idx, (center_lat, center_lon) in enumerate(coarse_centers):
            _, _, is_visual_rain = get_weather_at_for_scan(center_lat, center_lon, hour_key_override)
            if is_visual_rain:
                suspected_cell_indices.append(idx)
    except Exception as e:
        logging.warning(f"Heatmap coarse scan failed: {e}")

    # 強制模式：若無可疑格，至少掃中心格一次
    if force and not suspected_cell_indices:
        suspected_cell_indices = [ (len(coarse_cells)//2) ]  # 4x4 的中間偏近格

    unique_points = set()
    for idx in suspected_cell_indices:
        cell_bounds = coarse_cells[idx]
        for (p_lat, p_lon) in _gen_fine_points_for_cell(cell_bounds, fine_step):
            unique_points.add(_quantize_pair(p_lat, p_lon, q=0.05))
    if not unique_points and suspected_cell_indices:
        for idx in suspected_cell_indices:
            center_lat, center_lon = coarse_centers[idx]
            unique_points.add(_quantize_pair(center_lat, center_lon, q=0.05))
    if not unique_points:
        return []

    points_with_rain = []
    def fetch_weather_for_point(point: Tuple[float, float]) -> Optional[Tuple[float, float, float]]:
        lat, lon = point
        try:
            mm_now, _, is_visual_rain = get_weather_at_for_scan(lat, lon, hour_key_override)
            if is_visual_rain:
                return (lat, lon, min(mm_now, HEATMAP_MAX_MM))
        except Exception as e:
            logging.error(f"Weather point fetch error for ({lat}, {lon}): {e}")
        return None

    futures = [_EXECUTOR.submit(fetch_weather_for_point, p) for p in unique_points]
    for future in as_completed(futures):
        result = future.result()
        if result:
            points_with_rain.append(result)
    return points_with_rain

# ===============================================================================
# ===== END: UPDATED/NEW WEATHER SCANNING FUNCTIONS =============================
# ===============================================================================

def _get_step_for_zoom(zoom: int | float) -> float:
    z = float(zoom or 0)
    if z >= 12: return 0.03
    if z >= 11: return 0.05
    if z >= 10: return 0.08
    if z >= 9:  return 0.10
    if z >= 8:  return 0.15
    return 0.20

# 讓熱力半徑隨 zoom 放大而變大（支援小數 zoom，做線性內插）
_RADIUS_ANCHORS = [
    (4.0, 25), (5.0, 30), (6.0, 35), (7.0, 40),
    (8.0, 50), (9.0, 65), (10.0, 80), (11.0, 95),
]

def _get_radius_for_zoom(zoom: float) -> int:
    z = float(zoom or 0)
    if z <= _RADIUS_ANCHORS[0][0]: return _RADIUS_ANCHORS[0][1]
    if z >= _RADIUS_ANCHORS[-1][0]: return _RADIUS_ANCHORS[-1][1]
    for (z0, r0), (z1, r1) in zip(_RADIUS_ANCHORS, _RADIUS_ANCHORS[1:]):
        if z0 <= z <= z1:
            if z1 == z0: return int(round(r0))
            t = (z - z0) / (z1 - z0)
            r = r0 + t * (r1 - r0)
            return int(round(r))
    return 40

def _cap_step_for_points(south, north, west, east, step, max_points=400):
    lat_span = max(0.0, float(north) - float(south))
    lon_span = max(0.0, float(east)  - float(west))
    if step <= 0: step = 0.2
    est_points = (int(lat_span/step) + 1) * (int(lon_span/step) + 1)
    while est_points > max_points:
        step *= 1.5
        est_points = (int(lat_span/step) + 1) * (int(lon_span/step) + 1)
    return step

def _bounds_from_center_zoom(lat: float, lon: float, zoom: int | float):
    z = int(zoom or 13)
    if z >= 14: e = 0.10
    elif z >= 13: e = 0.15
    elif z >= 12: e = 0.25
    elif z >= 11: e = 0.40
    elif z >= 10: e = 0.60
    else: e = 0.90
    return [[lat - e, lon - e], [lat + e, lon + e]]

def _gen_fine_points_for_cell(cell_bounds, step):
    (south, west), (north, east) = cell_bounds
    pts = []
    lat_r = south
    while lat_r <= north + 1e-9:
        lon_r = west
        while lon_r <= east + 1e-9:
            pts.append((round(lat_r,5), round(lon_r,5)))
            lon_r += step
        lat_r += step
    return pts

# ===== 詳細預報功能 (For Alert Box) =====
forecast_cache = CacheToolsTTLCache(maxsize=1024, ttl=300)
@cached(forecast_cache)
def get_point_forecast(lat: float, lon: float, lang: str) -> dict:
    try:
        _, precip, codes, idx, offset_sec = _om_hourly_forecast_data(lat, lon)
        mm_now = float(precip[idx])
        code_now = int(codes[idx])
        confirmed_now = is_rain_now(mm_now, code_now)
        if confirmed_now:
            if THUNDER_MIN <= code_now <= THUNDER_MAX: weather_key = "heavy_rain"
            elif code_now >= 80 or mm_now >= 5.0: weather_key = "heavy_rain"
            elif code_now in RAIN_CODES or mm_now >= DECISION_MM_MIN: weather_key = "rain"
            else: weather_key = "lightrain"
        else:
            weather_key = "overcast" if code_now == 3 else ("cloudy" if code_now == 2 else "sunny")
        forecast_str = ""
        if THUNDER_MIN <= code_now <= THUNDER_MAX:
            forecast_str = f"⚠️ {t(lang, 'warn_thunder')}"
        elif confirmed_now and (code_now >= 80 or mm_now >= 5.0):
             forecast_str = f"⚠️ {t(lang, 'warn_heavy_rain')}"
        future_precip, future_codes = precip[idx+1:], codes[idx+1:]
        if confirmed_now:
            stops_in = -1
            for i, mm in enumerate(future_precip):
                if (mm < VISUAL_MM_MIN) and not (THUNDER_MIN <= future_codes[i] <= THUNDER_MAX):
                    stops_in = (i + 1); break
            if stops_in == 1: forecast_str += f" ({t(lang, 'stops_in_1h')})"
            elif stops_in > 1: forecast_str += f" ({t(lang, 'stops_in_xh').format(stops_in)})"
        else:
            starts_in = -1
            for i, mm in enumerate(future_precip[:6]):
                if (mm >= VISUAL_MM_MIN) or (THUNDER_MIN <= future_codes[i] <= THUNDER_MAX):
                    starts_in = (i + 1); break
            if starts_in != -1: forecast_str += f" ({t(lang, 'starts_in_xh').format(starts_in)})"
        temp, _, _ = _get_temp_from_owm_or_om(lat, lon)
        return {"key": weather_key, "temp": temp, "forecast": forecast_str.strip(), "offset_sec": offset_sec}
    except Exception as e:
        logging.error(f"Get forecast error: {e}")
        return {"key": "cloudy", "temp": None, "forecast": "", "offset_sec": 0}

temp_cache = CacheToolsTTLCache(maxsize=2048, ttl=300)
@cached(temp_cache)
def _get_temp_from_owm_or_om(lat, lon):
    if HAS_OWM:
        try:
            r = requests.get("https://api.openweathermap.org/data/2.5/weather",
                             params={"lat":lat,"lon":lon,"appid":OWM_KEY,"units":"metric"}, timeout=8)
            r.raise_for_status()
            if (temp := r.json().get("main",{}).get("temp")) is not None: return temp, "owm", None
        except Exception: pass
    try:
        r = requests.get("https://api.open-meteo.com/v1/forecast",
                         params={"latitude":lat,"longitude":lon,"current_weather":True}, timeout=8)
        r.raise_for_status()
        if (temp := r.json().get("current_weather",{}).get("temperature")) is not None: return temp, "om", None
    except Exception: pass
    return None, None, "all failed"

# ===== 地理編碼 & 路線 (v3) =====
api_cache = CacheToolsTTLCache(maxsize=256, ttl=300)

def _looks_like_tw_place(q: str) -> bool:
    q = (q or "").strip()
    if not q: return False
    if q.isdigit() and len(q) <= 5: return True
    if any(t in q for t in "鄉鎮區市縣台臺高雄台北臺北新北桃園台中臺中台南臺南基隆新竹苗栗彰化南投雲林嘉義屏東宜蘭花蓮台東臺東澎湖金門連江"):
        return True
    return bool(re.search(r'[\u4e00-\u9fff]', q)) and len(q) <= 6

@cached(api_cache)
def _google_geocode_once(q: str, lang: str, *, region=None, components=None):
    params = {"address": q, "key": GOOGLE_MAPS_API_KEY, "language": lang}
    if region: params["region"] = region
    if components: params["components"] = components
    r = requests.get("https://maps.googleapis.com/maps/api/geocode/json", params=params, timeout=8)
    r.raise_for_status()
    return r.json()

# ===== Geocoding 強化：候選擴張 + Google Places 備援 =====
def _expand_candidates(q: str) -> List[str]:
    q = (q or "").strip()
    if not q: return []
    out = [q]
    is_digits = q.isdigit()
    is_short_zh = bool(re.search(r'^[\u4e00-\u9fff]{1,4}$', q))
    if is_digits or is_short_zh:
        extras = [f"台北{q}", f"{q} 台北", f"臺北{q}", f"{q} 臺北", f"Taipei {q}", f"{q} Taipei", f"台灣{q}", f"{q} 台灣"]
        for e in extras:
            if e not in out: out.append(e)
    return out

@cached(api_cache)
def _google_places_findplace(q: str, lang: str, *, region_tw: bool = False):
    if not HAS_GMAP: return None
    try:
        params = {"input": q, "inputtype": "textquery", "language": lang, "fields": "geometry,formatted_address,name", "key": GOOGLE_MAPS_API_KEY}
        if region_tw: params["locationbias"] = "ipbias"
        r = requests.get("https://maps.googleapis.com/maps/api/place/findplacefromtext/json", params=params, timeout=8)
        r.raise_for_status()
        js = r.json()
        if not (cands := js.get("candidates") or []): return None
        top = cands[0]
        if not (geom := top.get("geometry", {}).get("location")): return None
        addr = top.get("formatted_address") or top.get("name") or q
        lat, lon = geom.get("lat"), geom.get("lng")
        if lat is None or lon is None: return None
        return (addr, (float(lat), float(lon)), "google_places", None)
    except Exception:
        return None

def _looks_international(q: str) -> bool:
    qn = q.lower()
    if any(k in q for k in INTERNATIONAL_KEYWORDS): return True
    if re.search(r'[A-Za-z].*[ ,]', qn): return True
    return False

@cached(api_cache)
def _geocode_nominatim(q: str, lang: str = "zh-TW", tw_only: bool = True):
    params = {"q": q, "format": "json", "addressdetails": 1, "accept-language": lang, "limit": 1}
    if tw_only: params["countrycodes"] = "tw"
    try:
        r = requests.get("https://nominatim.openstreetmap.org/search", params=params, timeout=8, headers=UA)
        r.raise_for_status()
        js = r.json()
    except Exception: return None
    if isinstance(js, list) and js:
        top = js[0]
        return top.get("display_name") or q, (float(top["lat"]), float(top["lon"])), "osm", None
    return None

def smart_geocode(q: str, lang: str = "zh-TW"):
    q = (q or "").strip()
    if not q: return None
    if _looks_international(q):
        if HAS_GMAP:
            try:
                if (data := _google_geocode_once(q, lang)).get("status") == "OK" and data.get("results"):
                    r0, loc = data["results"][0], data["results"][0]["geometry"]["location"]
                    return r0.get("formatted_address", q), (loc["lat"], loc["lng"]), "google", r0["geometry"].get("viewport")
            except Exception: pass
            try:
                if r := _google_places_findplace(q, lang, region_tw=False): return r
            except Exception: pass
        try:
            if r := _geocode_nominatim(q, lang=lang, tw_only=False): return r
        except Exception: pass
        return None

    candidates = _expand_candidates(q) or [q]
    if HAS_GMAP:
        for cand in candidates:
            try:
                if (data := _google_geocode_once(cand, lang, region="tw", components="country:TW")).get("status") == "OK" and data.get("results"):
                    r0, loc = data["results"][0], data["results"][0]["geometry"]["location"]
                    return r0.get("formatted_address", cand), (loc["lat"], loc["lng"]), "google", r0["geometry"].get("viewport")
            except Exception: pass
            try:
                if r := _google_places_findplace(cand, lang, region_tw=True): return r
            except Exception: pass
        try:
            if (data := _google_geocode_once(q, lang)).get("status") == "OK" and data.get("results"):
                r0, loc = data["results"][0], data["results"][0]["geometry"]["location"]
                return r0.get("formatted_address", q), (loc["lat"], loc["lng"]), "google", r0["geometry"].get("viewport")
        except Exception: pass
    try:
        if r := _geocode_nominatim(q, lang=lang, tw_only=True): return r
    except Exception: pass
    try:
        if r := _geocode_nominatim(q, lang=lang, tw_only=False): return r
    except Exception: pass
    return None

def reverse_geocode(lat: float, lon: float, ui_lang: str, prefer_area: bool = False):
    lang = LANG_MAP.get(ui_lang, "zh-TW")
    if HAS_GMAP:
        try:
            params = {"latlng": f"{lat},{lon}", "key": GOOGLE_MAPS_API_KEY, "language": lang}
            if prefer_area: params["result_type"] = "neighborhood|sublocality|locality|postal_town|administrative_area_level_3|administrative_area_level_2|administrative_area_level_1"
            r = requests.get("https://maps.googleapis.com/maps/api/geocode/json", params=params, timeout=8)
            r.raise_for_status()
            if (js := r.json()).get("status") == "OK" and (results := js.get("results") or []):
                if not prefer_area:
                    for item in results:
                        if not ({"establishment","point_of_interest","premise"} & set(item.get("types") or [])):
                            return item.get("formatted_address")
                return results[0].get("formatted_address")
        except Exception: pass
    try:
        r = requests.get("https://nominatim.openstreetmap.org/reverse", params={"lat": lat, "lon": lon, "format": "json", "accept-language": lang, "zoom": 14 if prefer_area else 18}, headers=UA, timeout=8)
        r.raise_for_status()
        return r.json().get("display_name")
    except Exception: return None

# ===== 路線規劃邏輯 =====
def _decode_polyline(poly: str) -> List[Tuple[float,float]]:
    pts, idx, lat, lng = [], 0, 0, 0
    while idx < len(poly):
        res, shift = 0, 0
        while True:
            b = ord(poly[idx]) - 63; idx += 1
            res |= (b & 0x1f) << shift; shift += 5
            if b < 0x20: break
        dlat = ~(res >> 1) if res & 1 else (res >> 1); lat += dlat
        res, shift = 0, 0
        while True:
            b = ord(poly[idx]) - 63; idx += 1
            res |= (b & 0x1f) << shift; shift += 5
            if b < 0x20: break
        dlng = ~(res >> 1) if res & 1 else (res >> 1); lng += dlng
        pts.append((lat/1e5, lng/1e5))
    return pts

def google_routes_with_alts(o, d, mode, lang):
    if not HAS_GMAP: return []
    m={'drive':'driving','walk':'walking','transit':'transit','scooter':'driving'}.get(mode,'driving')
    p={"origin":f"{o[0]},{o[1]}","destination":f"{d[0]},{d[1]}","mode":m,"alternatives":"true", "language": LANG_MAP.get(lang, "zh-TW"), "key":GOOGLE_KEY}
    try:
        r = requests.get("https://maps.googleapis.com/maps/api/directions/json", params=p, timeout=10)
        r.raise_for_status()
        js = r.json()
        return js.get("routes",[]) if js.get("status")=="OK" else []
    except Exception as e:
        logging.error(f"Google Directions API failed: {e}")
        return []

def osrm_route(o, d, mode):
    profile_map = {"drive": "driving", "walk": "walking"}
    osrm_profile = profile_map.get(mode, "driving")
    try:
        r = requests.get(f"https://router.project-osrm.org/route/v1/{osrm_profile}/{o[1]},{o[0]};{d[1]},{d[0]}",
                          params={"overview":"full","geometries":"polyline"}, timeout=10, headers=UA)
        r.raise_for_status()
        if (js := r.json()).get("routes"):
            return [{"overview_polyline": {"points": js["routes"][0]["geometry"]}}]
    except Exception as e:
        logging.error(f"OSRM Directions API failed: {e}")
    return []

def sample_indices(n_points: int, target: int=30) -> List[int]:
    if n_points<=1: return [0]
    step=max(1, (n_points-1) // max(1, target-1))
    idx=list(range(0, n_points, step))
    if idx[-1] != n_points-1: idx.append(n_points-1)
    return idx

def route_rain_flags_concurrent(lats: List[float], lons: List[float], lang: str) -> Tuple[List[bool], List[int]]:
    idxs = sample_indices(len(lats), target=30)
    coords_to_check = [(lats[i], lons[i], lang) for i in idxs]
    def check_route_point(p):
        lat, lon, _ = p
        try:
            _, precip, codes, idx, _ = _om_hourly_forecast_data(lat, lon)
            return is_rain_now(float(precip[idx]), int(codes[idx]))
        except Exception: return False
    try:
        flags = list(_EXECUTOR.map(check_route_point, coords_to_check))
    except Exception as e:
        logging.error(f"Concurrent route rain check failed: {e}")
        flags = [False] * len(idxs)
    return flags, idxs

def segments_by_flags(lats: List[float], lons: List[float], flags: List[bool], idxs: List[int]):
    segs=[]
    if not idxs: return []
    prev_i, prev_f = idxs[0], flags[0]
    for j in range(1,len(idxs)):
        i=idxs[j]
        segs.append({"lats":lats[prev_i:i+1], "lons":lons[prev_i:i+1], "color": COLOR_WET if prev_f else COLOR_DRY})
        prev_i, prev_f = i, flags[j]
    if prev_i == idxs[-1] and len(idxs) == 1:
        segs.append({"lats":lats, "lons":lons, "color": COLOR_WET if prev_f else COLOR_DRY})
    elif prev_i != len(lats) - 1:
        segs.append({"lats":lats[prev_i:len(lats)], "lons":lons[prev_i:len(lats)], "color": COLOR_WET if flags[-1] else COLOR_DRY})
    return segs

def bbox_center(lats: List[float], lons: List[float]) -> Tuple[float,float,float]:
    if not lats: return (BASE_CENTER[0], BASE_CENTER[1], BASE_ZOOM)
    minlat,maxlat=min(lats),max(lats); minlon,maxlon=min(lons),max(lons)
    c_lat, c_lon = (minlat+maxlat)/2.0, (minlon+maxlon)/2.0
    span = max(maxlat-minlat, maxlon-minlon, 0.001)
    zoom = 12.3 - math.log2(span * 111)
    if span * 111 <= 20:
        zoom += 0.6
    zoom = max(3, min(14, zoom))
    return c_lat, c_lon, zoom

# ==== 距離→zoom 工具（只用自動演算法，無固定值） ====
def _haversine_km(lat1, lon1, lat2, lon2):
    from math import radians, sin, cos, asin, sqrt
    R = 6371.0
    dlat = radians(lat2-lat1)
    dlon = radians(lon2-lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1))*cos(radians(lat2))*sin(dlon/2)**2
    c = 2*asin(sqrt(a))
    return R*c

def _route_length_km(lats, lons, sample=50):
    """對路徑抽樣估算總長，避免對極長 polyline 全量相加造成延遲。"""
    if not lats or len(lats) != len(lons): return 0.0
    n = len(lats)
    if n <= sample: idxs = range(n)
    else:
        step = max(1, (n-1)//(sample-1))
        idxs = list(range(0, n, step))
        if idxs[-1] != n-1: idxs.append(n-1)
    dist = 0.0
    for i in range(1, len(idxs)):
        a, b = idxs[i-1], idxs[i]
        dist += _haversine_km(lats[a], lons[a], lats[b], lons[b])
    return dist

def _route_zoom_from_km(dist_km: float) -> float:
    """距離→zoom 對應表（可依體感微調）：超短程看更近；中長程逐步拉遠。"""
    if dist_km <= 3: return 13.5
    if dist_km <= 8: return 12.5
    if dist_km <= 20: return 11.5
    if dist_km <= 60: return 10.5
    if dist_km <= 150: return 9.5
    return 8.5

def _prepare_contour_data(points_with_rain: List[Tuple[float, float, float]]) -> Optional[Tuple[List, List, List]]:
    """Converts sparse point data [(lat, lon, z), ...] to a dense grid (lats, lons, Z_grid) for contours."""
    if not points_with_rain or len(points_with_rain) < 3:
        return None
    try:
        point_map = {(lat, lon): z for lat, lon, z in points_with_rain}
        lats_1d = sorted(list(set(p[0] for p in points_with_rain)))
        lons_1d = sorted(list(set(p[1] for p in points_with_rain)))
        if len(lats_1d) < 2 or len(lons_1d) < 2: return None
        z_grid = [[point_map.get((lat, lon), 0) for lon in lons_1d] for lat in lats_1d]
        return lats_1d, lons_1d, z_grid
    except Exception:
        return None

# ===== Dash App Layout =====
app = Dash(__name__, title="即時雨區＋路線規劃", suppress_callback_exceptions=True, assets_folder=ASSETS_DIR)
server = app.server
initial_figure = base_map_figure(center=BASE_CENTER, zoom=BASE_ZOOM, style=DEFAULT_TILE_STYLE)

app.layout = html.Div([
    dcc.Store("lang-store", data="zh"),
    dcc.Store("mode-store", data="explore"),
    dcc.Store("explore-store", data={}),
    dcc.Store("route-store", data={}),
    dcc.Store("view-store", data={"center": BASE_CENTER, "zoom": BASE_ZOOM}),
    dcc.Store("geo-store"),
    dcc.Store("ui-store", data={"areaBusy": False}),
    dcc.Store(id="i18n-ts-prefix"),
    dcc.Store(id="user-location-store", data=None),
    dcc.Store(id="status-store", data={"type": None, "data": {}}),
    dcc.Store(id="timestamp-store", data=None),
    dcc.Store(id="rain-heatmap-store", data=None),
    dcc.Store(id="panel-store", storage_type="local", data={"panel": "open"}),
    html.Button("≡", id="panel-toggle", n_clicks=0, className="panel-toggle-mobile", **{"aria-controls": "panel", "aria-expanded": "true", "title": "開啟/關閉控制台"}),
    html.Div(id="panel-scrim", className="panel-scrim", n_clicks=0),
    html.Div(id="panel", className="panel", children=[
        html.H2(id="ttl", children="控制台", style={'marginTop': '8px', 'marginBottom': '16px'}),
        html.Button("🌐", id="btn-lang", className="globe", **{"aria-label": "切換語言", "aria-controls": "lang-menu", "aria-expanded": "false"}),
        html.Div(id="lang-menu", role="menu", className="menu hide", children=[
            html.Button("中文", id="lang-zh", n_clicks=0, **{"role": "menuitem"}),
            html.Button("English", id="lang-en", n_clicks=0, **{"role": "menuitem"}),
            html.Button("日本語", id="lang-ja", n_clicks=0, **{"role": "menuitem"}),
        ]),
        dcc.RadioItems(id="mode", value="explore", className="rad", labelStyle={"display": "inline-block", "marginRight": "15px"}),
        html.Div(id="box-explore", children=[
            html.Div(className="input-row", children=[
                dcc.Input(id="q", className="input"),
                html.Button(id="btn-search", className="button"),
            ]),
            html.Div(className="row gap", children=[
                html.Button(id="btn-area", className="button link"),
                html.Button(id="btn-locate", className="button link"),
            ]),
        ]),
        html.Div(id="box-route", className="hide", children=[
            html.Div(className="row", children=[
                html.Span(id="lab-travel", className="lab"),
                dcc.RadioItems(id="travel-mode", value="drive", className="rad", labelStyle={"display": "inline-block", "marginRight": "10px"}),
            ]),
            dcc.Input(id="src", className="input"),
            dcc.Input(id="dst", className="input"),
            html.Div(className="input-row", children=[
                html.Button(id="btn-plan", className="button", style={"flex": 1}),
                html.Button("📍", id="btn-locate-src", className="button link btn-locate"),
            ]),
        ]),
        html.Hr(),
        dcc.RadioItems(id="basemap", value="low", className="rad", labelStyle={"display": "inline-block", "marginRight": "15px"}),
        html.Div(id="addr-line", className="addr"),
        html.Div(id="alert", className="alert yellow hide"),
        html.Div(id="ts-line", className="ts")
    ]),
    html.Div(style={'position': 'fixed', 'top': 0, 'left': 0, 'width': '100%', 'height': '100%', 'zIndex': 0}, children=[
        dcc.Graph(id="map", style={"height":"100%"}, config={"scrollZoom": True, "displaylogo": False, "displayModeBar": False}, figure=initial_figure),
    ]),
    html.Div(className="legend", id="legend-a", children=[
        html.Span(id="legend-title", className="legend-title"),
        html.Div(className="legend-scale-dynamic", id="legend-scale-container")
    ])
])

# ===== Callbacks =====
app.clientside_callback(
    """
    function(n_clicks_loc, n_clicks_loc_src) {
      const dccx = (window.dash_clientside && window.dash_clientside.callback_context) || {};
      if (!dccx.triggered || !dccx.triggered.length) return window.dash_clientside.no_update;
      const trigId = (dccx.triggered[0].prop_id || "").split(".")[0];
      if (trigId !== "btn-locate" && trigId !== "btn-locate-src") return window.dash_clientside.no_update;
      return new Promise(resolve => {
        if (!navigator.geolocation) { resolve({ error: "no-geo" }); return; }
        navigator.geolocation.getCurrentPosition(
          pos => resolve({ lat: pos.coords.latitude, lon: pos.coords.longitude }),
          err => resolve({ error: (err && (err.code === 1 ? "denied" : err.message)) || "error" }),
          { enableHighAccuracy: true, timeout: 8000, maximumAge: 10000 }
        );
      });
    }
    """,
    Output("geo-store", "data"),
    Input("btn-locate", "n_clicks"), Input("btn-locate-src", "n_clicks"),
    prevent_initial_call=True
)

@app.callback(
    Output("lang-store", "data"), Output("lang-menu", "className"), Output("btn-lang", "aria-expanded"),
    Input("btn-lang", "n_clicks"), Input("lang-zh", "n_clicks"), Input("lang-en", "n_clicks"), Input("lang-ja", "n_clicks"),
    State("lang-store", "data"), State("lang-menu", "className"),
    prevent_initial_call=True
)
def lang_control(_, __, ___, ____, cur, klass):
    trig = ctx.triggered_id
    if trig == "btn-lang":
        is_opening = "hide" in (klass or "")
        return cur, "menu" if is_opening else "menu hide", "true" if is_opening else "false"
    if trig == "lang-zh": return "zh", "menu hide", "false"
    if trig == "lang-en": return "en", "menu hide", "false"
    if trig == "lang-ja": return "ja", "menu hide", "false"
    return cur, "menu hide", "false"

@app.callback(
    Output("ttl", "children"), Output("mode", "options"), Output("q", "placeholder"), Output("btn-search", "children"),
    Output("btn-locate", "children"), Output("lab-travel", "children"), Output("travel-mode", "options"),
    Output("src", "placeholder"), Output("dst", "placeholder"), Output("btn-plan", "children"),
    Output("basemap", "options"), Output("i18n-ts-prefix", "data"),
    Output("legend-title", "children"), Output("btn-locate-src", "title"),
    Input("lang-store", "data"),
)
def update_i18n_text(lang):
    travel_opts = [{"label": t(lang, "drive"), "value": "drive"}, {"label": t(lang, "walk"), "value": "walk"}]
    if HAS_GMAP:
        travel_opts = [{"label": t(lang, "drive"), "value": "drive"}, {"label": t(lang, "scooter"), "value": "scooter"},
                       {"label": t(lang, "walk"), "value": "walk"}, {"label": t(lang, "transit"), "value": "transit"}]
    return (t(lang, "panel_title"), [{"label": t(lang, "mode_explore"), "value": "explore"}, {"label": t(lang, "mode_route"), "value": "route"}],
            t(lang, "placeholder_q"), t(lang, "search"), t(lang, "locate"), t(lang, "travel_mode"), travel_opts,
            t(lang, "placeholder_src"), t(lang, "placeholder_dst"), t(lang, "plan"),
            [{"label": t(lang, "low"), "value": "low"}, {"label": t(lang, "osm"), "value": "osm"}],
            t(lang, "update"), t(lang, "legend_rain"), t(lang, "locate_src_title"))

@app.callback(
    Output("mode-store", "data"), Output("box-explore", "className"), Output("box-route", "className"),
    Output("rain-heatmap-store", "data", allow_duplicate=True), Output("explore-store", "data", allow_duplicate=True),
    Output("route-store", "data", allow_duplicate=True), Output("status-store", "data", allow_duplicate=True),
    Output("timestamp-store", "data", allow_duplicate=True), Output("q", "value", allow_duplicate=True),
    Output("src", "value", allow_duplicate=True), Output("dst", "value", allow_duplicate=True),
    Input("mode", "value"),
    prevent_initial_call=True
)
def on_mode(m):
    clear = ([], {}, {}, {"type": None, "data": {}}, None, "", "", "")
    return (m, "" if m == "explore" else "hide", "hide" if m == "explore" else "", *clear)

@app.callback(
    Output("btn-area", "children"), Output("btn-area", "disabled"),
    Input("ui-store", "data"), Input("lang-store", "data")
)
def area_btn_ui(uistate, lang):
    busy = (uistate or {}).get("areaBusy", False)
    return t(lang,"searching") if busy else t(lang,"search_area"), busy

@app.callback(
    Output("addr-line", "children"), Output("alert", "children"), Output("alert", "className"),
    Input("status-store", "data"), Input("lang-store", "data")
)
def update_status_text(status, lang):
    status = status or {"type": None}
    stype, data = status.get("type"), status.get("data", {})
    if stype == "explore":
        temp_str = f"｜{round(data['temp'])}°C" if data.get("temp") is not None else ""
        parts = [p for p in [t(lang, data.get("weather_key", "cloudy")), temp_str, data.get("forecast", "")] if p]
        return f"📍 {data.get('addr', '')}", " | ".join(parts), "alert yellow"
    if stype == "route":
        d_temp_str = f"｜{round(data['d_temp'])}°C" if data.get("d_temp") is not None else ""
        dest_str = f" // {t(lang, 'dest_now')}：{t(lang, data.get('d_lvl_key', 'cloudy'))}{d_temp_str}"
        return f"📍 {t(lang,'addr_fixed')}：{data['o_addr']} → {data['d_addr']}", f"{data.get('prefix','')}{t(lang,'best')} {data['risk']}%{dest_str}", "alert blue"
    if stype == "error":
        alert_txt = t(lang, data.get("key", "toast_err"))
        if data.get("key") == "no_route":
            d_temp_str = f"｜{round(data['d_temp'])}°C" if data.get("d_temp") is not None else ""
            alert_txt += f" // {t(lang, 'dest_now')}：{t(lang, data.get('d_lvl_key', 'cloudy'))}{d_temp_str}"
        return "" if data.get("mode") == "explore" else no_update, alert_txt, "alert blue" if data.get("mode") == "route" else "alert yellow"
    return "", "", "alert yellow hide"

@app.callback(Output("ts-line", "children"), Input("timestamp-store", "data"), Input("i18n-ts-prefix", "data"))
def update_timestamp_text(ts, prefix):
    return f"{prefix} {ts}" if ts and prefix else ""

@app.callback(
    Output("explore-store", "data", allow_duplicate=True), Output("route-store", "data", allow_duplicate=True),
    Output("status-store", "data", allow_duplicate=True), Output("timestamp-store", "data", allow_duplicate=True),
    Output("view-store", "data"), Output("ui-store", "data", allow_duplicate=True),
    Output("user-location-store", "data"), Output("rain-heatmap-store", "data", allow_duplicate=True),
    Output("q", "value", allow_duplicate=True), Output("src", "value", allow_duplicate=True),
    Output("dst", "value", allow_duplicate=True),
    Input("btn-search", "n_clicks"), Input("btn-area", "n_clicks"), Input("q", "n_submit"),
    Input("geo-store", "data"), Input("btn-plan", "n_clicks"), Input("dst", "n_submit"), Input("src", "n_submit"),
    Input("map", "relayoutData"),
    State("q", "value"), State("src", "value"), State("dst", "value"), State("travel-mode", "value"),
    State("lang-store", "data"), State("mode-store", "data"), State("view-store", "data"),
    prevent_initial_call=True
)
def main_controller(_, __, ___, geo, ____, _____, ______, relayout, q, src, dst, travel, lang, mode, view):
    trig_id = (ctx.triggered[0]['prop_id'] or "").split('.')[0]
    
    # Default outputs
    explore_out, route_out, status_out, ts_out, view_out, ui_out, user_loc_out, heatmap_out, q_out, src_out, dst_out = [no_update] * 11
    
    if mode == "explore":
        lat, lon, zoom, addr = None, None, None, ""
        
        force_scan = (trig_id == "btn-area")

        if trig_id in ("btn-search", "q"):
            if not (res := smart_geocode(q or "", lang=LANG_MAP.get(lang, "zh-TW"))):
                status_out = {"type": "error", "data": {"key": "toast_err", "mode": "explore"}}
                ts_out = datetime.now(timezone(timedelta(hours=8))).strftime("%H:%M:%S")
                ui_out = {"areaBusy": False}
                q_out = ""
                return no_update, no_update, status_out, ts_out, no_update, ui_out, no_update, no_update, q_out, no_update, no_update
            
            addr, (lat, lon), _, vp = res
            zoom = SEARCHED_ZOOM
            if vp and (ne := vp.get("northeast")) and (sw := vp.get("southwest")):
                span = max(abs(ne["lat"] - sw["lat"]), abs(ne["lng"] - sw["lng"]), 0.001)
                zoom = max(4.5, min(14.0, 11.5 - math.log2(span * 111)))
            view_out = {"center": [lat, lon], "zoom": zoom}
            explore_out = {"coord": (lat, lon), "addr": addr}
            q_out = ""
        
        elif trig_id == "geo-store":
            if geo and geo.get("error"): return [no_update]*11
            lat, lon = geo["lat"], geo["lon"]
            addr = reverse_geocode(lat, lon, lang) or f"({lat:.4f}, {lon:.4f})"
            zoom = 14
            view_out = {"center": [lat, lon], "zoom": zoom}
            explore_out = {"coord": (lat, lon), "addr": addr}
        
        elif trig_id == "btn-area":
            api_cache.clear() # Clear geocoding cache as well
            lat, lon, zoom = view["center"][0], view["center"][1], view["zoom"]
            addr = t(lang, "map_center")
            explore_out = {} # Clear specific point marker
        
        else: raise PreventUpdate
        
        scan_bounds = _bounds_from_center_zoom(lat, lon, zoom)
        wx_points = get_weather_data_for_bounds(scan_bounds, zoom, force=force_scan)
        heatmap_out = wx_points
        
        forecast = get_point_forecast(lat, lon, lang)
        status_out = {"type": "explore", "data": {"addr": addr, **forecast}}
        ts_out = datetime.now(timezone(timedelta(seconds=forecast.get("offset_sec", 28800)))).strftime("%H:%M:%S")
        
        ui_out = {"areaBusy": False} # Reset busy state after completion
        
        return explore_out, {}, status_out, ts_out, view_out, ui_out, no_update, heatmap_out, q_out, no_update, no_update

    if mode == "route":
        if trig_id == "geo-store":
            if geo and geo.get("error"):
                status_out = {"type": "error", "data": {"key": "loc_fail", "mode": "route"}}
                ts_out = datetime.now(timezone(timedelta(hours=8))).strftime("%H:%M:%S")
                return [no_update]*2 + [status_out, ts_out] + [no_update]*7
            raise PreventUpdate
            
        if trig_id in ("btn-plan", "dst", "src"):
            if not src or not dst:
                status_out = {"type": "error", "data": {"key": "toast_err", "mode": "route"}}
                ts_out = datetime.now(timezone(timedelta(hours=8))).strftime("%H:%M:%S")
                return [no_update]*2 + [status_out, ts_out] + [no_update]*7

            g1, g2 = smart_geocode(src, lang=LANG_MAP.get(lang, "zh-TW")), smart_geocode(dst, lang=LANG_MAP.get(lang, "zh-TW"))
            if not g1 or not g2:
                status_out = {"type": "error", "data": {"key": "toast_err", "mode": "route"}}
                ts_out = datetime.now(timezone(timedelta(hours=8))).strftime("%H:%M:%S")
                return [no_update]*2 + [status_out, ts_out] + [no_update]*7

            o_addr, o_coord, _, _ = g1
            d_addr, d_coord, _, _ = g2
            d_forecast = get_point_forecast(d_coord[0], d_coord[1], lang)
            ts_out = datetime.now(timezone(timedelta(seconds=d_forecast.get("offset_sec", 28800)))).strftime("%H:%M:%S")
            raw_routes, prefix = (osrm_route(o_coord, d_coord, travel), f"[{t(lang,'no_gmap_key')}] ")
            if HAS_GMAP: raw_routes, prefix = google_routes_with_alts(o_coord, d_coord, travel, lang), ""

            if not raw_routes:
                status_out = {"type": "error", "data": {"key": "no_route", "mode": "route", **d_forecast}}
                return {}, {}, status_out, ts_out, no_update, no_update, no_update, [], no_update, "", ""
            
            scored = []
            for r in raw_routes:
                if (poly := r.get("overview_polyline", {}).get("points")) and (pts := _decode_polyline(poly)):
                    lats, lons = [p[0] for p in pts], [p[1] for p in pts]
                    flags, idxs = route_rain_flags_concurrent(lats, lons, lang)
                    risk = sum(1.0 for f in flags if f) / max(1, len(flags))
                    scored.append({"route": {"lats": lats, "lons": lons}, "risk": risk, "flags": flags, "idxs": idxs})
            
            if not scored:
                status_out = {"type": "error", "data": {"key": "no_route", "mode": "route", **d_forecast}}
                return {}, {}, status_out, ts_out, no_update, no_update, no_update, [], no_update, "", ""

            scored.sort(key=lambda x: x["risk"])
            best, others = scored[0], scored[1:] if HAS_GMAP else []
            route_out = {"origin": {"addr": o_addr, "coord": o_coord}, "dest": {"addr": d_addr, "coord": d_coord}, "best": best, "others": [{"route": x["route"], "risk": x["risk"]} for x in others]}
            
            c_lat, c_lon, _ = bbox_center(best["route"]["lats"], best["route"]["lons"])
            dist_km = _route_length_km(best["route"]["lats"], best["route"]["lons"])
            z = _route_zoom_from_km(dist_km)
            center, zoom = clamp_view_to_tw([c_lat, c_lon], z)
            view_out = {"center": center, "zoom": zoom}

            status_out = {"type": "route", "data": {"o_addr": o_addr, "d_addr": d_addr, "risk": round(best["risk"] * 100), "prefix": prefix, **d_forecast}}
            return {}, route_out, status_out, ts_out, view_out, no_update, no_update, [], no_update, "", ""
    
    if trig_id == "map" and relayout:
        center = (view or {}).get("center", BASE_CENTER)
        zoom = (view or {}).get("zoom", BASE_ZOOM)
        if mb_center := relayout.get("mapbox.center"): center = [mb_center["lat"], mb_center["lon"]]
        if mb_zoom := relayout.get("mapbox.zoom"): zoom = float(mb_zoom)
        center, zoom = clamp_view_to_tw(center, zoom)
        view_out = {"center": center, "zoom": zoom}
        return [no_update]*4 + [view_out] + [no_update]*6
    
    raise PreventUpdate

@app.callback(Output("src", "value", allow_duplicate=True), Input("geo-store", "data"), State("mode-store", "data"), State("lang-store", "data"), prevent_initial_call=True)
def fill_src_from_locate(geo, mode, lang):
    if mode == "route" and geo and not geo.get("error"):
        if addr := reverse_geocode(geo["lat"], geo["lon"], lang): return addr
    return no_update

@app.callback(
    Output("map","figure"), Output("legend-a", "style"), Output("legend-scale-container", "children"),
    Input("basemap","value"), Input("explore-store","data"), Input("route-store","data"),
    Input("view-store", "data"), Input("mode-store", "data"), Input("rain-heatmap-store", "data"), Input("lang-store", "data"),
)
def draw_map(style, explore, route, view, mode, heatmap_data, lang):
    center = (view or {}).get("center", BASE_CENTER)
    zoom = float((view or {}).get("zoom", BASE_ZOOM))
    map_style = "carto-positron" if (style or "low") == "low" else "open-street-map"
    center, zoom = clamp_view_to_tw(center, zoom)
    fig = base_map_figure(center=center, zoom=zoom, style=map_style)
    
    if mode == "explore":
        legend_children = [
            html.Div(className="legend-scale", children=[html.Span(t(lang, "legend_light")), html.Span(t(lang, "legend_heavy"))]),
            html.Div(className="legend-bar", style={"backgroundImage": css_gradient_from_colorscale(HEATMAP_COLORSCALE)}),
        ]
        if heatmap_data:
            lats_sparse = [p[0] for p in heatmap_data]
            lons_sparse = [p[1] for p in heatmap_data]
            zs_sparse = [p[2] for p in heatmap_data]
            fig.add_trace(go.Densitymapbox(
                lat=lats_sparse, lon=lons_sparse, z=zs_sparse, radius=_get_radius_for_zoom(zoom),
                colorscale=HEATMAP_COLORSCALE, zmin=VISUAL_MM_MIN, zmax=HEATMAP_MAX_MM, 
                showscale=False, opacity=0.65
            ))
            contour_data = _prepare_contour_data(heatmap_data)
            if contour_data:
                lats_1d, lons_1d, mm_grid = contour_data
                LEVELS = [2, 10, 25, 50]
                fig.add_trace(go.Contourmapbox(
                    lat=lats_1d, lon=lons_1d, z=mm_grid,
                    contours=dict(start=min(LEVELS), end=max(LEVELS), size=8, coloring="none", showlabels=False),
                    line=dict(color="rgba(0,0,0,0.6)", width=2), showscale=False, hoverinfo="skip"
                ))
                fig.add_trace(go.Contourmapbox(
                    lat=lats_1d, lon=lons_1d, z=mm_grid,
                    contours=dict(start=min(LEVELS), end=max(LEVELS), size=8, coloring="none", showlabels=False),
                    line=dict(color="rgba(255,255,255,0.8)", width=1), showscale=False, hoverinfo="skip"
                ))
        if explore and (coord := explore.get("coord")):
            fig.add_trace(go.Scattermapbox(lat=[coord[0]], lon=[coord[1]], mode="markers",
                                           marker=dict(size=16, color="rgba(239,68,68,.95)"),
                                           hovertext=explore.get("addr"), hoverinfo="text"))
        return fig, {"display": "flex"}, legend_children
        
    if mode == "route":
        legend_children = [
            html.Div(className="legend-scale-route", children=[
                html.Div(className="swatch", style={"backgroundColor": COLOR_DRY}), html.Span(t(lang, "dry")),
                html.Div(className="swatch", style={"backgroundColor": COLOR_WET}), html.Span(t(lang, "rain")),
            ]),
        ]
        if route and (best := route.get("best")):
            br, flags, idxs = best["route"], best.get("flags",[]), best.get("idxs",[])
            for seg in segments_by_flags(br["lats"], br["lons"], flags, idxs):
                fig.add_trace(go.Scattermapbox(lat=seg["lats"], lon=seg["lons"], mode="lines",
                                               line=dict(width=8, color=seg["color"]), hoverinfo="none"))
            for x in (route.get("others") or []):
                rr=x["route"]
                fig.add_trace(go.Scattermapbox(lat=rr["lats"], lon=rr["lons"], mode="lines",
                                               line=dict(width=4, color="rgba(156,163,175,0.6)"), hoverinfo="none"))
            if o := route.get("origin",{}).get("coord"):
                fig.add_trace(go.Scattermapbox(lat=[o[0]], lon=[o[1]], mode="markers", marker=dict(size=16, color="rgba(239,68,68,.95)"),
                                               hovertext=route.get("origin",{}).get("addr")))
            if d := route.get("dest",{}).get("coord"):
                fig.add_trace(go.Scattermapbox(lat=[d[0]], lon=[d[1]], mode="markers", marker=dict(size=16, color="rgba(239,68,68,.95)"),
                                               hovertext=route.get("dest",{}).get("addr")))
        return fig, {"display": "flex"}, legend_children

    return fig, {"display": "none"}, []

app.clientside_callback(
    """
    function(nBtn, nScrim, ui){
      ui = ui || {};
      const trig = (window.dash_clientside.callback_context.triggered[0]||{}).prop_id || "";
      let open = (ui.panel === "open");
      if (trig.startsWith("panel-toggle")) open = !open;
      else if (trig.startsWith("panel-scrim")) open = false;
      const storeOut = (trig.startsWith("panel-toggle") || trig.startsWith("panel-scrim"))
                       ? {panel: open ? "open" : "closed"}
                       : window.dash_clientside.no_update;
      try {
        const btn = document.getElementById("panel-toggle");
        if (btn) btn.setAttribute("aria-expanded", open ? "true" : "false");
        document.body.classList.toggle("lock-scroll", open && window.matchMedia("(max-width: 768px)").matches);
      } catch(e){}
      return [open ? "panel" : "panel panel-hide", open ? "panel-scrim show" : "panel-scrim", storeOut];
    }
    """,
    Output("panel", "className"), Output("panel-scrim", "className"), Output("panel-store", "data"),
    Input("panel-toggle", "n_clicks"), Input("panel-scrim", "n_clicks"), State("panel-store", "data"),
)

app.clientside_callback(
    """
    function(n) {
      const dccx = (window.dash_clientside && window.dash_clientside.callback_context) || {};
      if (!dccx.triggered || !dccx.triggered.length) return window.dash_clientside.no_update;
      return {areaBusy: true};
    }
    """,
    Output("ui-store", "data", allow_duplicate=True),
    Input("btn-area", "n_clicks"),
    prevent_initial_call=True
)

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8050"))
    # 本地測試使用 127.0.0.1；Render 部署用 0.0.0.0
    app.run(host=os.getenv("HOST", "0.0.0.0"), port=port, debug=False)