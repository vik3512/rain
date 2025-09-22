#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# App Version: 10.2 (merged: CWA SSL compat + scoped mount + OM retry + scan timeout + geocode cache key)

import os, json, time, math, logging
from datetime import datetime, timezone, timedelta
from typing import Tuple, Optional, List
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError

from cachetools import cached, TTLCache
from cachetools.keys import hashkey
import requests
from dash import Dash, html, dcc, Input, Output, State, no_update, ctx
from dash.exceptions import PreventUpdate
import plotly.graph_objects as go

# === SSL / HTTP tooling ===
import ssl
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from requests.exceptions import SSLError
try:
    # for broader compatibility (urllib3 v1.x / v2.x)
    from urllib3 import PoolManager
except Exception:
    from urllib3.poolmanager import PoolManager  # type: ignore

# Silence only when insecure fallback is used
try:
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
except Exception:
    pass

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# ===== Logging =====
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ==== Version Tag ====
APP_VERSION = "10.2"

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
    I18N = {"zh": {"title":"即時雨區＋路線規劃","panel_title":"控制台","mode_explore":"定點雨量","mode_route":"路線雨量","placeholder_q":"輸入地點","search":"搜尋","locate":"定位","travel_mode":"交通方式","drive":"開車","scooter":"機車","walk":"步行","transit":"大眾運輸","placeholder_src":"輸入出發地","placeholder_dst":"輸入目的地","plan":"規劃路線","basemap":"底圖","low":"低飽和","osm":"標準 (OSM)","update":"更新於","legend_rain":"降雨熱度","legend_light":"較弱","legend_heavy":"較強","locate_src_title":"將起點設為目前位置","map_center":"地圖中心","toast_err":"發生錯誤","loc_fail":"定位失敗","no_route":"找不到路線","best":"最佳路線","others":"其他路線","origin":"起點","dest":"終點","dest_now":"目的地現在","addr_fixed":"路線","warn_thunder":"雷雨","warn_heavy_rain":"大雨","stops_in_1h":"約 1 小時後停","stops_in_xh":"約 {} 小時後停","starts_in_xh":"約 {} 小時後開始","rain":"有雨","lightrain":"小雨","heavy_rain":"大雨/雷雨","overcast":"陰","cloudy":"多雲","sunny":"晴","search_area":"搜尋此區域","searching":"搜尋中…","no_gmap_key":"無 Google 路線 API 金鑰","dry":"無雨","partial_result":" (部分結果)"}}

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

_EXECUTOR = ThreadPoolExecutor(max_workers=16)

# === Sessions (general + CWA with scoped relaxed strict) ===
def _make_retry_session(total=2, backoff=0.2) -> requests.Session:
    s = requests.Session()
    retries = Retry(
        total=total,
        backoff_factor=backoff,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        raise_on_status=False,
    )
    s.mount("https://", HTTPAdapter(max_retries=retries))
    return s

def _make_cwa_session() -> requests.Session:
    """Create a session that relaxes VERIFY_X509_STRICT, scoped to CWA only, and compatible with older requests."""
    ctx = ssl.create_default_context()
    try:
        ctx.verify_flags &= ~ssl.VERIFY_X509_STRICT  # type: ignore[attr-defined]
    except Exception:
        pass

    class SSLContextAdapter(HTTPAdapter):
        # For requests < 2.31 / urllib3 < 2 compatible injection
        def init_poolmanager(self, *args, **kwargs):
            kwargs['ssl_context'] = ctx
            return super().init_poolmanager(*args, **kwargs)
        def proxy_manager_for(self, *args, **kwargs):
            kwargs['ssl_context'] = ctx
            return super().proxy_manager_for(*args, **kwargs)

    s = requests.Session()
    retries = Retry(
        total=2,
        backoff_factor=0.2,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        raise_on_status=False,
    )
    # Scope adapter to exact host prefix to avoid leaking relaxed rules elsewhere
    s.mount("https://opendata.cwa.gov.tw/", SSLContextAdapter(max_retries=retries))
    return s

_CWA = _make_cwa_session()   # only for opendata.cwa.gov.tw
_OM  = _make_retry_session() # for api.open-meteo.com

# ===== 地圖基礎 & 新架構參數 =====
BASE_CENTER = [23.9738, 120.9820]
BASE_ZOOM   = 7
SEARCHED_ZOOM = 12
LOCATE_ZOOM = 13
DEFAULT_TILE_STYLE = "carto-positron"
FALLBACK_CENTER = [25.0375, 121.5637]
TW_BBOX = (21.5, 119.5, 25.5, 122.5)

RAIN_CIRCLE_COLORSCALE = [[0.0, "#9BE7FF"], [0.25, "#39B6FF"], [0.5, "#1C64F2"], [0.75, "#6D28D9"], [1.0, "#A21CAF"]]
COLOR_DRY  = "rgba(16,185,129,0.95)"
COLOR_WET  = "rgba(37,99,235,0.85)"

VISUAL_MM_MIN = 0.2
DECISION_MM_MIN = 0.8
THUNDER_MIN, THUNDER_MAX = 95, 99
HEATMAP_MAX_MM = 8.0

# ✅ 改動 1：新增 uirevision 參數；預設用視角字串，讓地圖在程式性移動時穩定套用新視角
def base_map_figure(center=BASE_CENTER, zoom=BASE_ZOOM, style=DEFAULT_TILE_STYLE, uirevision=None):
    fig = go.Figure(go.Scattermapbox(lat=[center[0]], lon=[center[1]], mode='markers', marker=dict(size=0, opacity=0)))
    fig.update_layout(
        mapbox=dict(style=style, center=dict(lat=center[0], lon=center[1]), zoom=zoom),
        margin=dict(l=0, r=0, t=0, b=0),
        uirevision=uirevision or f"view-{center[0]:.4f}-{center[1]:.4f}-{float(zoom):.2f}",
        dragmode="pan",
        showlegend=False,
    )
    return fig

# ===== 快取 =====
om_api_cache = TTLCache(maxsize=8000, ttl=3500)
api_cache = TTLCache(maxsize=256, ttl=300)
cwa_warnings_cache = TTLCache(maxsize=1, ttl=300)
cwa_stations_cache = TTLCache(maxsize=1, ttl=3600)
cwa_temp_cache = TTLCache(maxsize=1, ttl=600)
owm_cache = TTLCache(maxsize=1024, ttl=600)

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
            return {"west": w, "south": s, "east": e, "north": n}
    except Exception:
        pass
    return None

# === 新增：按鈕點擊時才讀一次視角 ===
def _center_zoom_from_relayout(relayout, fallback_view=None):
    """僅在按鈕點擊時讀一次視角；若 relayout 缺值就用 fallback_view。"""
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

# ====== CWA: 雨量 ======
@cached(cwa_stations_cache)
def get_cwa_stations_data():
    if not CWA_API_KEY: return None
    url = "https://opendata.cwa.gov.tw/api/v1/rest/datastore/O-A0002-001"
    params = {"Authorization": CWA_API_KEY, "elementName": "RAIN,MIN_10"}
    try:
        r = _CWA.get(url, params=params, timeout=6, headers=UA)
        r.raise_for_status()
    except SSLError:
        # 單次退而求其次（只限 CWA）
        try:
            r = requests.get(url, params=params, timeout=6, headers=UA, verify=False)
            r.raise_for_status()
        except Exception as e:
            logging.error(f"Failed to fetch CWA stations (insecure fallback failed): {e}")
            return None
    except Exception as e:
        logging.error(f"Failed to fetch CWA stations: {e}")
        return None

    try:
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
    except Exception as e:
        logging.error(f"CWA stations parse error: {e}")
        return None

def get_cwa_realtime_rain(lat: float, lon: float) -> Optional[float]:
    stations = get_cwa_stations_data()
    if not stations: return None
    min_dist_sq = float('inf'); closest_station_rain = None
    for station in stations:
        dist_sq = (lat - station['lat'])**2 + (lon - station['lon'])**2
        if dist_sq < min_dist_sq:
            min_dist_sq = dist_sq
            if station['rain_1h'] >= 0:
                closest_station_rain = station['rain_1h']
            elif station['rain_10min'] >= 0:
                closest_station_rain = station['rain_10min'] * 6.0
            else:
                closest_station_rain = None
    if min_dist_sq > 0.25**2: return None
    return closest_station_rain

# ====== CWA: 溫度 ======
@cached(cwa_temp_cache)
def get_cwa_nearby_temp(lat: float, lon: float) -> Optional[float]:
    if not CWA_API_KEY: return None
    url = "https://opendata.cwa.gov.tw/api/v1/rest/datastore/O-A0001-001"
    params = {"Authorization": CWA_API_KEY, "elementName": "TEMP"}
    try:
        r = _CWA.get(url, params=params, timeout=6, headers=UA)
        r.raise_for_status()
    except SSLError:
        try:
            r = requests.get(url, params=params, timeout=6, headers=UA, verify=False)
            r.raise_for_status()
        except Exception as e:
            logging.warning(f"CWA TEMP insecure fallback failed: {e}")
            return None
    except Exception as e:
        logging.warning(f"CWA TEMP fetch failed: {e}")
        return None

    try:
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
    except Exception as e:
        logging.warning(f"CWA TEMP parse failed: {e}")
        return None

# ====== OWM: 溫度後援 ======
@cached(owm_cache)
def get_owm_temp(lat: float, lon: float) -> Optional[float]:
    if not OWM_KEY: return None
    try:
        r = requests.get(
            "https://api.openweathermap.org/data/2.5/weather",
            params={"lat": lat, "lon": lon, "appid": OWM_KEY, "units": "metric"},
            timeout=6, headers=UA
        )
        r.raise_for_status()
        js = r.json()
        return float(js.get("main", {}).get("temp")) if js.get("main") else None
    except Exception as e:
        logging.warning(f"OWM temp failed: {e}")
        return None

def _clear_weather_caches_safely():
    for cache in [om_api_cache, api_cache, cwa_warnings_cache, cwa_stations_cache, cwa_temp_cache, owm_cache]:
        try:
            if hasattr(cache, 'clear'): cache.clear()
        except Exception:
            pass

def _current_hour_key_utc(): return datetime.utcnow().strftime("%Y-%m-%dT%H")

# ====== Open-Meteo: 小時預報（加重試 + timeout 12s） ======
@cached(om_api_cache)
def _om_hourly_forecast_data_cached_api(qlat: float, qlon: float, hour_key_utc: str):
    try:
        r = _OM.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": qlat, "longitude": qlon,
                "hourly": "precipitation,weather_code,temperature_2m",
                "forecast_days": 2, "timezone": "auto"
            },
            timeout=12, headers=UA
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

def _om_hourly_forecast_data(lat: float, lon: float, hour_key_override: Optional[str] = None):
    return _om_hourly_forecast_data_cached_api(round(lat, 2), round(lon, 2), hour_key_override or _current_hour_key_utc())

def get_rain_mm_hybrid(lat: float, lon: float) -> Tuple[float, int, bool]:
    if is_in_taiwan(lat, lon):
        cwa_rain_mm_hr = get_cwa_realtime_rain(lat, lon)
        if cwa_rain_mm_hr is not None:
            code = 80 if cwa_rain_mm_hr > 0.1 else 0
            is_visual_rain = (cwa_rain_mm_hr >= VISUAL_MM_MIN)
            return cwa_rain_mm_hr, code, is_visual_rain
    try:
        _, precip, codes, _, idx, _ = _om_hourly_forecast_data(lat, lon)
        mm_now, code_om = float(precip[idx]), int(codes[idx])
        is_visual_rain = (mm_now >= VISUAL_MM_MIN) or (THUNDER_MIN <= code_om <= THUNDER_MAX)
        return mm_now, code_om, is_visual_rain
    except Exception:
        return 0.0, 0, False

# === 掃描（預設逾時由 4s -> 6s） ===
def get_weather_data_for_bounds(bounds, zoom, timeout_sec=6.0):
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

    points_with_rain = []

    def fetch(p):
        try:
            if not is_in_taiwan(p[0], p[1]):
                return None
            mm, _, is_rain = get_rain_mm_hybrid(p[0], p[1])
            if is_rain: return (p[0], p[1], min(mm, HEATMAP_MAX_MM))
        except Exception:
            pass
        return None

    futures = [_EXECUTOR.submit(fetch, p) for p in points]
    try:
        for future in as_completed(futures, timeout=timeout_sec):
            if result := future.result(): points_with_rain.append(result)
    except TimeoutError:
        timed_out = True
        logging.warning(f"Weather scan timed out after {timeout_sec} seconds.")

    return points_with_rain, timed_out

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

@cached(api_cache)
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
        future_precip = precip[idx+1:idx+12]
        is_raining_now = precip[idx] > 0.1
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

        return {"key": key, "forecast": forecast_str, "temp": best_temp, "offset_sec": offset,
                "d_lvl_key": key, "d_temp": best_temp}
    except Exception as e:
        logging.error(f"Error in get_point_forecast: {e}")
        best_temp = get_best_temperature(lat, lon, lang)
        return {"key": "cloudy", "forecast": "", "temp": best_temp, "offset_sec": 0,
                "d_lvl_key": "cloudy", "d_temp": best_temp}

# ====== Geocoding（修正 cache key：避免 bounds 是 dict 造成 unhashable） ======
def _smart_geocode_key(q, lang, bounds=None):
    # 簡單版：忽略 bounds，提升命中率並避免 dict 當 key
    return hashkey(q, lang)

@cached(api_cache, key=_smart_geocode_key)
def smart_geocode(q: str, lang: str = "zh-TW", bounds: Optional[dict] = None):
    if HAS_GMAP:
        try:
            params = {"address": q, "key": GOOGLE_KEY, "language": lang, "region": "tw"}
            if bounds:
                params["bounds"] = f"{bounds['south']},{bounds['west']}|{bounds['north']},{bounds['east']}"
            r = requests.get("https://maps.googleapis.com/maps/api/geocode/json", params=params, headers=UA, timeout=5)
            r.raise_for_status()
            js = r.json()
            if js['status'] == 'OK' and js.get('results'):
                res = js['results'][0]
                loc = res['geometry']['location']
                return res['formatted_address'], (loc['lat'], loc['lng'])
        except Exception as e:
            logging.warning(f"Google Geocode failed, falling back to OSM: {e}")
    params = {"q": q, "format": "jsonv2", "accept-language": lang, "limit": 1}
    if bounds:
        params["viewbox"] = f"{bounds['west']},{bounds['north']},{bounds['east']},{bounds['south']}"
        params["bounded"] = 1
    r = requests.get("https://nominatim.openstreetmap.org/search", params=params, headers=UA, timeout=5)
    r.raise_for_status()
    js = r.json()
    if not js: raise ValueError("Geocoding failed for both Google and OSM")
    lat, lon = float(js[0]['lat']), float(js[0]['lon'])
    return js[0]['display_name'], (lat, lon)

@cached(api_cache)
def reverse_geocode(lat: float, lon: float, ui_lang: str):
    lang_code = LANG_MAP.get(ui_lang, "zh-TW")
    if HAS_GMAP:
        try:
            params = {"latlng": f"{lat},{lon}", "key": GOOGLE_KEY, "language": lang_code,
                      "result_type": "street_address|route|political"}
            r = requests.get("https://maps.googleapis.com/maps/api/geocode/json", params=params, headers=UA, timeout=5)
            r.raise_for_status()
            js = r.json()
            if js['status'] == 'OK' and js.get('results'):
                return js['results'][0]['formatted_address']
        except Exception as e:
            logging.warning(f"Google Reverse Geocode failed, falling back to OSM: {e}")
    r = requests.get("https://nominatim.openstreetmap.org/reverse",
                     params={"lat": lat, "lon": lon, "format": "jsonv2", "accept-language": lang_code, "zoom": 16},
                     headers=UA, timeout=5)
    r.raise_for_status()
    return r.json().get('display_name')

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
    return {"lats": [c[0] for c in coords], "lons": [c[1] for c in coords],
            "duration": route_json['legs'][0]['duration']['value'],
            "summary": route_json.get('summary', '')}

def google_routes_with_alts(o: tuple, d: tuple, mode: str, lang: str):
    if not HAS_GMAP: return []
    try:
        gmap_mode_map = {"drive": "driving", "scooter": "two_wheeler", "walk": "walking", "transit": "transit"}
        params = {"origin": f"{o[0]},{o[1]}", "destination": f"{d[0]},{d[1]}", "key": GOOGLE_KEY,
                  "language": LANG_MAP.get(lang, "zh-TW"), "mode": gmap_mode_map.get(mode, "driving"),
                  "alternatives": "true"}
        if mode == "scooter": params["avoid"] = "highways"
        r = requests.get("https://maps.googleapis.com/maps/api/directions/json", params=params, headers=UA, timeout=8)
        r.raise_for_status()
        return [_parse_gmap_route(route) for route in r.json().get('routes', [])]
    except Exception as e:
        logging.error(f"Google route failed: {e}")
        return []

def osrm_route(o: tuple, d: tuple, mode: str):
    profile_map = {"drive":"driving", "walk":"walking", "scooter":"driving"}
    profile = profile_map.get(mode, "driving")
    coords = f"{o[1]},{o[0]};{d[1]},{d[0]}"
    url = f"http://router.project-osrm.org/route/v1/{profile}/{coords}?overview=full&geometries=polyline"
    if mode == "scooter": url += "&exclude=motorway"
    try:
        r = requests.get(url, headers=UA, timeout=8)
        r.raise_for_status()
        js = r.json()
        if js.get('code') == 'Ok' and js.get('routes'):
            route = js['routes'][0]
            coords = _decode_polyline(route['geometry'])
            return [{"lats": [c[0] for c in coords], "lons": [c[1] for c in coords],
                     "duration": route['duration'], "summary": "OSRM Route"}]
    except Exception as e:
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

def route_rain_flags_concurrent(lats: List[float], lons: List[float]):
    def fetch(p):
        mm, code, _ = get_rain_mm_hybrid(p[0], p[1])
        return mm >= DECISION_MM_MIN, code
    with ThreadPoolExecutor(max_workers=10) as executor:
        results = list(executor.map(fetch, zip(lats, lons)))
    return zip(*results) if results else ([], [])

# ========= 小範圍摘要 =========
def summarize_small_area(center_lat: float, center_lon: float, wx_points: List[tuple], zoom: float) -> Optional[dict]:
    if not wx_points: return {"core_points": 0}
    z = max(4.0, min(16.0, float(zoom or 12.0)))
    radius = max(0.05, min(0.10, 0.07 * (12.0 / z)))
    core = [(la, lo, mm) for (la, lo, mm) in wx_points if abs(la - center_lat) <= radius and abs(lo - center_lon) <= radius]
    core_points = len(core)
    if core_points == 0:
        return {"core_points": 0}
    mm_list = [mm for (_, _, mm) in core]
    max_mm = max(mm_list)
    avg_mm = sum(mm_list) / core_points
    if max_mm >= 3.0: key_area = "heavy_rain"
    elif max_mm >= 0.6: key_area = "rain"
    else: key_area = "lightrain"
    return {"core_points": core_points, "max_mm": round(max_mm, 1), "avg_mm": round(avg_mm, 1), "key_area": key_area}

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
            html.Div(className="row", children=[
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

# --- 非掃描事件也要更新下方地址（與天氣/溫度） ---
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
    status_out = {"type": "explore", "data": {"addr": addr, **(forecast if isinstance(forecast, dict) else {})}, "ts": trig.get("ts")}
    return status_out, ts_out

# --- 後端資料控制（僅 area_scan 會進來） ---
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
    if source == 'area_scan':
        # 優先用觸發物件自帶的 zoom/bounds；缺少時再退回 view-store 或估算
        zoom = trig.get("zoom", (view or {}).get("zoom", SEARCHED_ZOOM))
        tbounds = trig.get("bounds")
        if isinstance(tbounds, list) and len(tbounds) == 2 and len(tbounds[0]) == 2 and len(tbounds[1]) == 2:
            bounds = tbounds  # [[s_lat, w_lon], [n_lat, e_lon]]
        else:
            bounds = _bounds_from_center_zoom(lat, lon, zoom)
        try:
            wx_points, timed_out = get_weather_data_for_bounds(bounds, zoom)
        except Exception as e:
            logging.error(f"Scan failed: {e}")
            wx_points, timed_out = [], True

    forecast = get_point_forecast(lat, lon, lang)

    area_summary = None
    if source == 'area_scan':
        zoom_now = trig.get("zoom", (view or {}).get("zoom", SEARCHED_ZOOM))
        area_summary = summarize_small_area(lat, lon, wx_points, zoom_now)

    off = int((forecast.get("offset_sec") if isinstance(forecast, dict) else 28800) or 28800)
    ts_out = datetime.now(timezone(timedelta(seconds=off))).strftime("%H:%M:%S")

    payload = {"addr": addr, **(forecast if isinstance(forecast, dict) else {})}
    if area_summary is not None:
        payload["area"] = area_summary

    status_out = {"type": "explore", "data": payload, "ts": ts}

    final_marker = (explore_data or {}).copy()
    final_marker.update({"coord": [lat, lon], "addr": addr, "label": None, "ts": ts})

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

    # ✅ 改動 1（配合）：動態 uirevision，確保程式性平移/縮放立即生效
    uirev = f"view-{center[0]:.4f}-{center[1]:.4f}-{float(zoom):.2f}"
    fig = base_map_figure(center=center, zoom=zoom, style=style, uirevision=uirev)

    legend_style, legend_children = {"display": "none"}, []

    if mode == "explore":
        if rain_points:
            lats, lons, vals = zip(*rain_points)
            fig.add_trace(go.Scattermapbox(
                lat=lats, lon=lons, mode='markers',
                marker=dict(
                    size=max(25, _get_radius_for_zoom(zoom) * 0.8),
                    color=vals, colorscale=RAIN_CIRCLE_COLORSCALE,
                    cmin=0, cmax=HEATMAP_MAX_MM, opacity=0.65
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

# --- 狀態文本 ---
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
        parts = [p for p in [t(lang, data.get("key", "cloudy")), temp_str, data.get("forecast", "")] if p]

        area = data.get("area")
        if isinstance(area, dict):
            if area.get("core_points", 0) > 0:
                lvl = t(lang, area.get("key_area", "rain"))
                mmax = data.get("area", {}).get("max_mm")
                parts.append(f"此區域：{lvl}（{area['core_points']} 點、最大 {mmax} mm/h）")
            else:
                parts.append(f"此區域：{t(lang,'dry')}")

        alert_text = " | ".join(parts)
        return addr_text, alert_text, "alert yellow"

    if stype == "route":
        o_addr, d_addr = data.get('o_addr',''), data.get('d_addr','')
        addr_text = f"📍 {t(lang, 'addr_fixed')}：{o_addr} → {d_addr}"
        d_temp_v = data.get("d_temp", None)
        d_temp_str = f"{round(d_temp_v)}°C" if isinstance(d_temp_v, (int, float)) else ""
        dest_parts = [t(lang, data.get('d_lvl_key','cloudy')), d_temp_str]
        dest_str = f" // {t(lang,'dest_now')}：{' '.join(filter(None, dest_parts))}"
        alert_text = f"{data.get('prefix','')}{t(lang,'best')} {data.get('risk',0)}%{dest_str}"
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

# --- 路線控制（成功後清空 src/dst） ---
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

    for r in routes:
        flags, _ = route_rain_flags_concurrent(r['lats'], r['lons'])
        r['rain_flags'] = list(flags)
        r['risk'] = sum(flags) / len(flags) if flags else 0

    routes.sort(key=lambda r: (r['risk'], r.get('duration', float('inf'))))
    best = routes[0]

    ts_out = datetime.now(timezone(timedelta(seconds=(d_forecast.get("offset_sec") if isinstance(d_forecast, dict) else 28800)))).strftime("%H:%M:%S")
    prefix = f"{t(lang, 'best')} ({t(lang, 'others')}: {len(routes)-1}) " if len(routes) > 1 else ""
    status_out = {"type": "route", "data": {"o_addr": o_addr, "d_addr": d_addr,
        "risk": round(best.get("risk", 0) * 100), "prefix": prefix, **(d_forecast if isinstance(d_forecast, dict) else {})}}

    center, zoom = bbox_center(best['lats'], best['lons'])
    view_out = {"center": center, "zoom": zoom}

    # 成功規劃後清空輸入框
    return best, status_out, ts_out, view_out, "", ""

# ===== 入口 =====
if __name__ == "__main__":
    port = int(os.getenv("PORT", "8050"))
    app.run(host=os.getenv("HOST", "0.0.0.0"), port=port, debug=False)
