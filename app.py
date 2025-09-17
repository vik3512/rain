#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# App Version: 7.6-final (修復熱力圖與提示條邏輯不一致)

import os, json, time, threading, math, logging
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

def _om_hourly_forecast_data(lat: float, lon: float):
    qlat, qlon = _quantize_pair(lat, lon, q=0.05)
    hour_key = _current_hour_key_utc()
    memo_key = (qlat, qlon, hour_key)
    cached_data = HOURLY_CACHE.get(memo_key)
    if cached_data:
        return cached_data
    val = _om_hourly_forecast_data_cached_api(qlat, qlon, hour_key)
    HOURLY_CACHE.set(memo_key, val)
    return val

def is_rain_now(mm_now: float, code_om: int) -> bool:
    if mm_now >= VISUAL_MM_MIN:
        return True
    if THUNDER_MIN <= code_om <= THUNDER_MAX:
        return True
    if (code_om in RAIN_CODES) and (mm_now >= 0.3):
        return True
    return False

def _get_step_for_zoom(zoom: int | float) -> float:
    z = float(zoom or 0)
    if z >= 12: return 0.03
    if z >= 11: return 0.05
    if z >= 10: return 0.08
    if z >= 9:  return 0.10
    if z >= 8:  return 0.15
    return 0.20
    
# 讓熱力半徑隨 zoom 放大而變大（支援小數 zoom，做線性內插）
# 對照表（你指定的值）：
# 4→25, 5→30, 6→35, 7→40, 8→50, 9→65, 10→80, 11→95
_RADIUS_ANCHORS = [
    (4.0, 25),
    (5.0, 30),
    (6.0, 35),
    (7.0, 40),
    (8.0, 50),
    (9.0, 65),
    (10.0, 80),
    (11.0, 95),
]

def _get_radius_for_zoom(zoom: float) -> int:
    z = float(zoom or 0)

    # 邊界外直接 clamp 到最接近的端點
    if z <= _RADIUS_ANCHORS[0][0]:
        return _RADIUS_ANCHORS[0][1]
    if z >= _RADIUS_ANCHORS[-1][0]:
        return _RADIUS_ANCHORS[-1][1]

    # 線性內插：找到 z 所在的兩個錨點，依比例算半徑
    for (z0, r0), (z1, r1) in zip(_RADIUS_ANCHORS, _RADIUS_ANCHORS[1:]):
        if z0 <= z <= z1:
            if z1 == z0:
                return int(round(r0))
            t = (z - z0) / (z1 - z0)
            r = r0 + t * (r1 - r0)
            return int(round(r))

    # 理論上不會到這裡
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

def _anchors_from_bounds(bounds):
    (south, west), (north, east) = bounds
    cy = (south + north) / 2.0
    cx = (west  + east)  / 2.0
    return [(cy, cx), (south, west), (south, east), (north, west), (north, east)]

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

def get_weather_at_for_scan(lat, lon):
    try:
        _, precip, codes, idx, _ = _om_hourly_forecast_data(lat, lon)
        mm_now = float(precip[idx])
        code_om = int(codes[idx])
        is_visual_rain = (mm_now >= VISUAL_MM_MIN) or (THUNDER_MIN <= code_om <= THUNDER_MAX)
        return mm_now, code_om, is_visual_rain
    except Exception:
        return 0.0, 0, False

def get_weather_data_for_bounds(bounds, zoom) -> List[Tuple[float, float, float]]:
    try:
        (south_lat, west_lon), (north_lat, east_lon) = bounds
    except Exception:
        return []
    fine_step = _cap_step_for_points(south_lat, north_lat, west_lon, east_lon, _get_step_for_zoom(zoom), max_points=400)
    anchor_pts = _anchors_from_bounds(((south_lat, west_lon), (north_lat, east_lon)))
    all_dry = True
    try:
        for (ay, ax) in anchor_pts:
            _, _, is_visual_rain = get_weather_at_for_scan(ay, ax)
            if is_visual_rain: 
                all_dry = False; break
        if all_dry: return []
    except Exception as e:
        logging.error(f"Anchor check error: {e}")
    centers, cells = _coarse_centers(((south_lat, west_lon), (north_lat, east_lon)), n=3)
    suspect_cells_idx = []
    try:
        for idx, (cy, cx) in enumerate(centers):
            _, _, is_visual_rain = get_weather_at_for_scan(cy, cx)
            if is_visual_rain: 
                suspect_cells_idx.append(idx)
    except Exception as e:
        logging.error(f"Coarse scan error: {e}")
    uniq_points = set()
    for idx in suspect_cells_idx:
        cell = cells[idx]
        for (py, px) in _gen_fine_points_for_cell(cell, fine_step):
            uniq_points.add(_quantize_pair(py, px, q=0.05))
    if not uniq_points and suspect_cells_idx:
        for idx in suspect_cells_idx:
            cy, cx = centers[idx]
            uniq_points.add(_quantize_pair(cy, cx, q=0.05))
    if not uniq_points:
        return []
    points_with_rain = []
    def work(pt):
        glat, glon = pt
        mm_now, code_om, is_visual_rain = get_weather_at_for_scan(glat, glon)
        if is_visual_rain:
            return (glat, glon, min(mm_now, HEATMAP_MAX_MM))
        return None
    futures = [_EXECUTOR.submit(work, p) for p in uniq_points]
    for f in as_completed(futures):
        try:
            res = f.result()
            if res: points_with_rain.append(res)
        except Exception as e:
            logging.error(f"Weather point fetch error: {e}")
    return points_with_rain

# ===== 詳細預報功能 (For Alert Box) =====
forecast_cache = CacheToolsTTLCache(maxsize=1024, ttl=300)
@cached(forecast_cache) 
def get_point_forecast(lat: float, lon: float, lang: str) -> dict:
    try:
        times, precip, codes, idx, offset_sec = _om_hourly_forecast_data(lat, lon)
        mm_now = float(precip[idx])
        code_now = int(codes[idx])
        confirmed_now = is_rain_now(mm_now, code_now)
        
        if confirmed_now:
            if THUNDER_MIN <= code_now <= THUNDER_MAX:
                 weather_key = "heavy_rain"
            elif code_now >= 80 or mm_now >= 5.0:
                 weather_key = "heavy_rain"
            elif code_now in RAIN_CODES or mm_now >= DECISION_MM_MIN:
                 weather_key = "rain"
            else:
                 weather_key = "lightrain"
        else:
            weather_key = "overcast" if code_now == 3 else ("cloudy" if code_now == 2 else "sunny")
        is_raining_now = confirmed_now
        forecast_str = ""
        if THUNDER_MIN <= code_now <= THUNDER_MAX:
            forecast_str = f"⚠️ {t(lang, 'warn_thunder')}"
        elif confirmed_now and (code_now >= 80 or mm_now >= 5.0):
             forecast_str = f"⚠️ {t(lang, 'warn_heavy_rain')}"
        future_precip = precip[idx+1:]
        future_codes = codes[idx+1:]
        if is_raining_now:
            stops_in = -1
            for i in range(len(future_precip)):
                if (future_precip[i] < VISUAL_MM_MIN) and not (THUNDER_MIN <= future_codes[i] <= THUNDER_MAX):
                    stops_in = (i + 1)
                    break
            if stops_in == 1:
                forecast_str += f" ({t(lang, 'stops_in_1h')})"
            elif stops_in > 1:
                forecast_str += f" ({t(lang, 'stops_in_xh').format(stops_in)})"
        else:
            starts_in = -1
            for i in range(min(len(future_precip), 6)):
                if (future_precip[i] >= VISUAL_MM_MIN) or (THUNDER_MIN <= future_codes[i] <= THUNDER_MAX):
                    starts_in = (i + 1)
                    break
            if starts_in != -1:
                 forecast_str += f" ({t(lang, 'starts_in_xh').format(starts_in)})"
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
                             params={"lat":lat,"lon":lon,"appid":OWM_KEY,"units":"metric"},
                             timeout=8)
            r.raise_for_status()
            temp = r.json().get("main",{}).get("temp")
            if temp is not None:
                return temp, "owm", None
        except Exception:
            pass
    try:
        r = requests.get("https://api.open-meteo.com/v1/forecast",
                         params={"latitude":lat,"longitude":lon,"current_weather":True},
                         timeout=8)
        r.raise_for_status()
        temp = r.json().get("current_weather",{}).get("temperature")
        if temp is not None:
             return temp, "om", None
    except Exception:
        pass
    return None, None, "all failed"

# ===== 地理編碼 & 路線 =====
def _expand_candidates(q: str) -> List[str]:
    q = (q or "").strip()
    base = [q] if q else []
    if q == "101": 
        base = ["台北101", "Taipei 101"] + base
    return list(dict.fromkeys(base))

api_cache = CacheToolsTTLCache(maxsize=256, ttl=300)

@cached(api_cache)
def _geocode_google(q: str, scope: str = "tw", lang: str = "zh-TW"):
    if not HAS_GMAP: return None, (None, None), "NO_KEY", None
    params = {"address": q, "key": GOOGLE_MAPS_API_KEY, "language": lang}
    if scope == "tw": 
        params.update({"region": "tw", "components": "country:TW"})
    try:
        r = requests.get("https://maps.googleapis.com/maps/api/geocode/json", params=params, timeout=8)
        r.raise_for_status()
        js = r.json()
    except Exception:
        return None, (None, None), "REQUEST_ERROR", None
    if js.get("status") == "OK" and js.get("results"):
        top = js["results"][0]
        loc = top["geometry"]["location"]
        vp = top["geometry"].get("viewport")
        return top.get("formatted_address", q), (loc.get("lat"), loc.get("lng")), None, vp
    return None, (None, None), js.get("status") or "NO_RESULT", None

@cached(api_cache)
def _places_findplace(q: str, lang: str = "zh-TW"):
    if not HAS_GMAP: return None, (None, None), "NO_KEY", None
    params = {"input": q, "inputtype": "textquery",
              "fields":"geometry,formatted_address,name,viewport","language":lang,"key":GOOGLE_MAPS_API_KEY}
    try:
        r = requests.get("https://maps.googleapis.com/maps/api/place/findplacefromtext/json", params=params, timeout=8)
        r.raise_for_status()
        js = r.json()
    except Exception:
        return None, (None, None), "REQUEST_ERROR", None
    c = js.get("candidates") or []
    if js.get("status") == "OK" and c:
        loc = c[0]["geometry"]["location"]
        vp = c[0]["geometry"].get("viewport")
        addr = c[0].get("formatted_address") or c[0].get("name") or q
        return addr, (float(loc.get("lat")), float(loc.get("lng"))), None, vp
    return None, (None, None), js.get("status") or "NO_RESULT", None

@cached(api_cache)
def _geocode_nominatim(q: str, lang: str = "zh-TW", tw_only: bool = True):
    params = {"q": q, "format": "json", "addressdetails": 1, "accept-language": lang, "limit": 1}
    if tw_only: params["countrycodes"] = "tw"
    try:
        r = requests.get("https://nominatim.openstreetmap.org/search", params=params, timeout=8, headers=UA)
        r.raise_for_status()
        js = r.json()
    except Exception:
        return None, (None, None), "REQUEST_ERROR", None
    if isinstance(js, list) and js:
        top = js[0]
        return top.get("display_name") or q, (float(top["lat"]), float(top["lon"])), None, None
    return None, (None, None), js.get("status") or "NO_RESULT", None

def smart_geocode(q: str, lang_code: str):
    lang = LANG_MAP.get(lang_code, "zh-TW")
    if not q: return None
    is_intl = any(k in q for k in INTERNATIONAL_KEYWORDS)
    candidates = _expand_candidates(q) if not is_intl else [q]
    if not is_intl:
        for cand in candidates:
            addr,(lat,lon),_,vp = _geocode_google(cand, scope="tw", lang=lang)
            if lat and lon: return (lat, lon, addr, vp)
        for cand in candidates:
            addr,(lat,lon),_,vp = _places_findplace(cand, lang=lang)
            if lat and lon: return (lat, lon, addr, vp)
    addr,(lat,lon),_,vp = _geocode_google(q, scope="global", lang=lang)
    if lat and lon: return (lat, lon, addr, vp)
    addr,(lat,lon),_,vp = _geocode_nominatim(q, lang=lang, tw_only=False)
    if lat and lon: return (lat, lon, addr, vp)
    return None

@cached(api_cache)
def _reverse_geocode_google(lat: float, lon: float, lang: str = "zh-TW", prefer_area: bool = False):
    if not HAS_GMAP: return None, "NO_KEY"
    params = {"latlng": f"{lat},{lon}", "key": GOOGLE_MAPS_API_KEY, "language": lang}
    if prefer_area:
        params["result_type"] = ("neighborhood|sublocality|locality|postal_town|"
            "administrative_area_level_3|administrative_area_level_2|administrative_area_level_1")
    try:
        r = requests.get("https://maps.googleapis.com/maps/api/geocode/json", params=params, timeout=8)
        r.raise_for_status()
        js = r.json()
    except Exception:
        return None, "REQUEST_ERROR"
    results = js.get("results") or []
    if js.get("status") == "OK" and results:
        if not prefer_area:
            for item in results:
                types = set(item.get("types") or [])
                if not ({"establishment","point_of_interest","premise"} & types):
                    return item.get("formatted_address"), None
        return results[0].get("formatted_address"), None
    return None, js.get("status") or "NO_RESULT"

def reverse_geocode(lat: float, lon: float, lang_code: str, prefer_area: bool = False):
    lang = LANG_MAP.get(lang_code, "zh-TW")
    addr, _ = _reverse_geocode_google(lat, lon, lang=lang, prefer_area=prefer_area)
    if addr: return addr
    try:
        r = requests.get("https://nominatim.openstreetmap.org/reverse",
                        params={"lat": lat, "lon": lon, "format": "json",
                                "accept-language": lang, "zoom": (14 if prefer_area else 18)},
                        headers=UA, timeout=8)
        r.raise_for_status()
        js = r.json()
        return js.get("display_name")
    except Exception:
        return None

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
    p={"origin":f"{o[0]},{o[1]}","destination":f"{d[0]},{d[1]}","mode":m,"alternatives":"true",
       "language": LANG_MAP.get(lang, "zh-TW"), "key":GOOGLE_KEY}
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
        js = r.json()
        if (js.get("routes") or []):
            poly = js["routes"][0]["geometry"]
            return [{"overview_polyline": {"points": poly}}]
    except Exception as e:
        logging.error(f"OSRM Directions API failed: {e}")
        return []
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
            mm_now = float(precip[idx])
            code_om = int(codes[idx])
            return is_rain_now(mm_now, code_om)
        except Exception:
            return False
    try:
        flags = list(_EXECUTOR.map(check_route_point, coords_to_check))
    except Exception as e:
        logging.error(f"Concurrent route rain check failed: {e}")
        flags = [False] * len(idxs)
    return flags, idxs

def segments_by_flags(lats: List[float], lons: List[float], flags: List[bool], idxs: List[int]):
    segs=[]
    # --- 修正：用常數確保顏色一致 ---
    if not idxs: return []
    prev_i=idxs[0]; prev_f=flags[0]
    for j in range(1,len(idxs)):
        i=idxs[j]
        seg_color = COLOR_WET if prev_f else COLOR_DRY
        segs.append({"lats":lats[prev_i:i+1], "lons":lons[prev_i:i+1], "color":seg_color})
        prev_i=i; prev_f=flags[j]
    if prev_i == idxs[-1] and len(idxs) == 1:
        i = idxs[0]
        segs.append({"lats":lats, "lons":lons, "color": (COLOR_WET if prev_f else COLOR_DRY)})
    elif prev_i != len(lats) - 1:
        last_color = COLOR_WET if flags[-1] else COLOR_DRY
        i = len(lats) - 1
        segs.append({"lats":lats[prev_i:i+1], "lons":lons[prev_i:i+1], "color": last_color})
    return segs

def bbox_center(lats: List[float], lons: List[float]) -> Tuple[float,float,float]:
    if not lats: return (BASE_CENTER[0], BASE_CENTER[1], BASE_ZOOM)
    minlat,maxlat=min(lats),max(lats); minlon,maxlon=min(lons),max(lons)
    c_lat, c_lon = (minlat+maxlat)/2.0, (minlon+maxlon)/2.0
    span = max(maxlat-minlat, maxlon-minlon, 0.001)
    zoom = 11.5 - math.log2(span * 111)
    zoom = max(3, min(zoom, 14))
    return c_lat,c_lon,zoom

# ===== Dash App Layout =====
app = Dash(__name__, title="即時雨區＋路線規劃", suppress_callback_exceptions=True, assets_folder=ASSETS_DIR)
server = app.server
initial_figure = base_map_figure(center=BASE_CENTER, zoom=BASE_ZOOM, style=DEFAULT_TILE_STYLE)

app.layout = html.Div([
    # ===== Stores =====
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

    # ===== Panel =====
    html.Div([
        # ── 頂部列：控制台標題 + 模式切換 + 語言切換 ──
        html.Div(className="top-bar", children=[
            html.H2(id="ttl", children="控制台"),
            dcc.RadioItems(
                id="mode",
                value="explore",
                className="rad",
                labelStyle={"display": "inline-block", "marginRight": "10px"},
                style={"overflowX": "auto"}  # 手機：過長時可水平滑動
            ),
            html.Button("🌐", id="btn-lang", className="btn globe"),
        ]),

        # 語言選單
        html.Div(id="lang-menu", className="menu hide", children=[
            html.Button("中文", id="lang-zh", n_clicks=0),
            html.Button("English", id="lang-en", n_clicks=0),
            html.Button("日本語", id="lang-ja", n_clicks=0),
        ]),

        # ── Explore 模式 ──
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

        # ── Route 模式 ──
        html.Div(id="box-route", className="hide", children=[
            html.Div(className="row", children=[
                html.Span(id="lab-travel", className="lab"),
                dcc.RadioItems(
                    id="travel-mode",
                    value="drive",
                    className="rad",
                    labelStyle={"display": "inline-block", "marginRight": "10px"}
                ),
            ]),
            dcc.Input(id="src", className="input"),
            dcc.Input(id="dst", className="input"),
            html.Div(className="input-row", children=[
                html.Button(id="btn-plan", className="button", style={"flex": 1}),
                html.Button("📍", id="btn-locate-src", className="button link btn-locate"),
            ]),
        ]),

        # 分隔線
        html.Hr(),

        # 底圖切換
        html.Div(className="row", children=[
            html.Span(id="lab-basemap", className="lab"),
            dcc.RadioItems(
                id="basemap",
                value="low",
                className="rad",
                labelStyle={"display": "inline-block", "marginRight": "15px"}
            ),
        ]),

        # 狀態/提示
        html.Div(id="addr-line", className="addr"),
        html.Div(id="alert", className="alert yellow hide"),
        html.Div(id="ts-line", className="ts"),
    ], className="panel"),

    # ===== Map =====
    html.Div(
        style={'position': 'fixed', 'top': 0, 'left': 0, 'width': '100%', 'height': '100%', 'zIndex': 0},
        children=[
            dcc.Graph(
                id="map",
                style={"height": "100%"},
                config={"scrollZoom": True, "displaylogo": False, "displayModeBar": False},
                figure=initial_figure
            ),
        ]
    ),

    # ===== Legend =====
    html.Div(className="legend", id="legend-a", children=[
        html.Span(id="legend-title", className="legend-title"),
        html.Div(className="legend-scale-dynamic", id="legend-scale-container"),
    ]),
])

# ===== Callbacks =====
app.clientside_callback(
    """
    function(n_clicks_loc, n_clicks_loc_src) {
      const dccx = (typeof dash_clientside !== "undefined" && dash_clientside.callback_context) || null;
      if (!dccx || !dccx.triggered || dccx.triggered.length === 0) {
        return (dash_clientside && dash_clientside.no_update) || null;
      }
      const trigId = (dccx.triggered[0].prop_id || "").split(".")[0];
      if (trigId !== "btn-locate" && trigId !== "btn-locate-src") {
        return (dash_clientside && dash_clientside.no_update) || null;
      }
      return new Promise((resolve) => {
        if (!navigator.geolocation) {
          resolve({ error: "no-geo" });
          return;
        }
        navigator.geolocation.getCurrentPosition(
          (pos) => resolve({ lat: pos.coords.latitude, lon: pos.coords.longitude }),
          (err) => {
            const msg = (err && (err.code === 1 ? "denied" : err.message)) || "error";
            resolve({ error: msg });
          },
          { enableHighAccuracy: true, timeout: 8000, maximumAge: 10000 }
        );
      });
    }
    """,
    Output("geo-store", "data"),
    Input("btn-locate", "n_clicks"),
    Input("btn-locate-src", "n_clicks"),
    prevent_initial_call=True
)

@app.callback(
    Output("lang-store","data"),
    Output("lang-menu","className"),
    Input("btn-lang","n_clicks"),
    Input("lang-zh","n_clicks"), Input("lang-en","n_clicks"), Input("lang-ja","n_clicks"),
    State("lang-store","data"), State("lang-menu","className"), prevent_initial_call=True
)
def lang_control(b, nz, ne, nj, cur, klass):
    trig = ctx.triggered_id
    if trig == "btn-lang":
        return cur, ("menu" if "hide" in (klass or "") else "menu hide")
    if trig == "lang-zh": return "zh", "menu hide"
    if trig == "lang-en": return "en", "menu hide"
    if trig == "lang-ja": return "ja", "menu hide"
    return cur, "menu hide"

@app.callback(
    Output("ttl", "children"),
    Output("mode", "options"),
    Output("q", "placeholder"),
    Output("btn-search", "children"),
    Output("btn-locate", "children"),
    Output("lab-travel", "children"),
    Output("travel-mode", "options"),
    Output("src", "placeholder"),
    Output("dst", "placeholder"),
    Output("btn-plan", "children"),
    Output("lab-basemap", "children"),
    Output("basemap", "options"),
    Output("i18n-ts-prefix", "data"), 
    Output("legend-title", "children"),
    Output("btn-locate-src", "title"),
    Input("lang-store", "data"),
)
def update_i18n_text(lang):
    mode_opts = [{"label": t(lang, "mode_explore"), "value": "explore"},
                 {"label": t(lang, "mode_route"), "value": "route"}]
    base_opts = [{"label": t(lang, "low"), "value": "low"},
                 {"label": t(lang, "osm"), "value": "osm"}]
    if HAS_GMAP:
        travel_opts = [{"label": t(lang, "drive"), "value": "drive"},
                       {"label": t(lang, "scooter"), "value": "scooter"},
                       {"label": t(lang, "walk"), "value": "walk"},
                       {"label": t(lang, "transit"), "value": "transit"}]
    else:
        travel_opts = [{"label": t(lang, "drive"), "value": "drive"},
                       {"label": t(lang, "walk"), "value": "walk"}]
    return (
        t(lang, "panel_title"),
        mode_opts,
        t(lang, "placeholder_q"),
        t(lang, "search"),
        t(lang, "locate"),
        t(lang, "travel_mode"),
        travel_opts,
        t(lang, "placeholder_src"),
        t(lang, "placeholder_dst"),
        t(lang, "plan"),
        t(lang, "basemap"),
        base_opts,
        t(lang, "update"),
        t(lang, "legend_rain"),
        t(lang, "locate_src_title"),
    )

@app.callback(
    Output("mode-store", "data"),
    Output("box-explore", "className"),
    Output("box-route", "className"),
    Output("rain-heatmap-store", "data", allow_duplicate=True),
    Output("explore-store", "data", allow_duplicate=True),
    Output("route-store", "data", allow_duplicate=True),
    Output("status-store", "data", allow_duplicate=True),
    Output("timestamp-store", "data", allow_duplicate=True),
    Output("q", "value", allow_duplicate=True),
    Output("src", "value", allow_duplicate=True),
    Output("dst", "value", allow_duplicate=True),
    Input("mode", "value"),
    prevent_initial_call=True
)
def on_mode(m):
    clear_data = ([], {}, {}, {"type": None, "data": {}}, None, "", "", "")
    if m == "explore":
        return (m, "", "hide", *clear_data)
    else:
        return (m, "hide", "", *clear_data)

@app.callback(
    Output("btn-area", "children"),
    Output("btn-area", "disabled"),
    Input("ui-store", "data"),
    Input("lang-store", "data")
)
def area_btn_ui(uistate, lang):
    busy = (uistate or {}).get("areaBusy", False)
    return (t(lang,"searching") if busy else t(lang,"search_area"), busy)

@app.callback(
    Output("addr-line", "children"),
    Output("alert", "children"),
    Output("alert", "className"),
    Input("status-store", "data"),
    Input("lang-store", "data")
)
def update_status_text(status_data, lang):
    status_data = status_data or {"type": None, "data": {}}
    status_type = status_data.get("type")
    data = status_data.get("data", {})
    
    if status_type == "explore":
        addr = data.get("addr", "")
        lvl_key = data.get("weather_key", "cloudy")
        temp = data.get("temp")
        forecast_str = data.get("forecast", "") 
        addr_line = f"📍 {addr}"
        temp_str = f"｜{round(temp)}°C" if temp is not None else ""
        weather_str = t(lang, lvl_key)
        parts = [p for p in [weather_str, temp_str, forecast_str] if p]
        alert_txt = " | ".join(parts)
        alert_class = "alert yellow"
        return addr_line, alert_txt, alert_class
    elif status_type == "route":
        o_addr = data.get("o_addr", "")
        d_addr = data.get("d_addr", "")
        d_lvl_key = data.get("d_lvl_key", "cloudy")
        d_temp = data.get("d_temp")
        risk = data.get("risk")
        prefix = data.get("prefix", "")
        d_temp_str = f"｜{round(d_temp)}°C" if d_temp is not None else ""
        dest_weather_str = f" // {t(lang, 'dest_now')}：{t(lang, d_lvl_key)}{d_temp_str}"
        addr_line = f"📍 {t(lang,'addr_fixed')}：{o_addr} → {d_addr}"
        alert_txt = f"{prefix}{t(lang,'best')} {risk}%{dest_weather_str}"
        alert_class = "alert blue"
        return addr_line, alert_txt, alert_class
    elif status_type == "error":
        key = data.get("key", "toast_err")
        mode = data.get("mode", "explore")
        alert_txt = t(lang, key)
        if key == "no_route":
            d_lvl_key = data.get("d_lvl_key", "cloudy")
            d_temp = data.get("d_temp")
            d_temp_str = f"｜{round(d_temp)}°C" if d_temp is not None else ""
            dest_weather_str = f" // {t(lang, 'dest_now')}：{t(lang, d_lvl_key)}{d_temp_str}"
            alert_txt += dest_weather_str
        alert_class = "alert blue" if mode == "route" else "alert yellow"
        addr_line_out = "" if mode == "explore" else no_update
        return addr_line_out, alert_txt, alert_class
    return "", "", "alert yellow hide"

@app.callback(
    Output("ts-line", "children"),
    Input("timestamp-store", "data"),
    Input("i18n-ts-prefix", "data")
)
def update_timestamp_text(ts_data, prefix):
    if not ts_data or not prefix:
        return ""
    return f"{prefix} {ts_data}"

@app.callback(
    Output("explore-store", "data", allow_duplicate=True),
    Output("route-store", "data", allow_duplicate=True),
    Output("status-store", "data", allow_duplicate=True),
    Output("timestamp-store", "data", allow_duplicate=True),
    Output("view-store", "data"),
    Output("ui-store", "data"),
    Output("user-location-store", "data"),
    Output("rain-heatmap-store", "data", allow_duplicate=True),
    Output("q", "value"),
    Output("src", "value", allow_duplicate=True),
    Output("dst", "value"),
    Input("btn-search", "n_clicks"),
    Input("btn-area", "n_clicks"),
    Input("q", "n_submit"),
    Input("geo-store", "data"),
    Input("btn-plan", "n_clicks"),
    Input("dst", "n_submit"),
    Input("src", "n_submit"),
    Input("map", "relayoutData"),
    State("q", "value"),
    State("src", "value"),
    State("dst", "value"),
    State("travel-mode", "value"),
    State("lang-store", "data"),
    State("mode-store", "data"),
    State("view-store", "data"),
    State("user-location-store", "data"),
    prevent_initial_call=True
)
def main_controller(n_search, n_area, n_submit, geo_data, n_plan, n_dst_submit, n_src_submit, relayout_data,
                    q_val, src_val, dst_val, travel_mode, lang, mode, current_view, user_loc):
    if not ctx.triggered:
        raise PreventUpdate
    trig_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    lang_code = LANG_MAP.get(lang, "zh-TW")
    explore_out, route_out, status_out, ts_out, view_out, ui_out, user_loc_out, heatmap_out, q_out, src_out, dst_out = (no_update,) * 11
    fallback_tz = timezone(timedelta(hours=8)) 
    def make_return():
        return (explore_out, route_out, status_out, ts_out, view_out, ui_out, user_loc_out, heatmap_out, q_out, src_out, dst_out)
    
    if mode == "route" and trig_id in ("btn-plan", "dst", "src", "geo-store"):
        if trig_id == "geo-store":
            if geo_data and geo_data.get("error"):
                status_out = {"type": "error", "data": {"key": "loc_fail", "mode": "route"}}
                return make_return()
            
            src_out = reverse_geocode(geo_data["lat"], geo_data["lon"], lang_code)
            return make_return()
    
    if (mode == "explore" and trig_id in ("btn-search", "q", "btn-area")) or \
       (mode == "explore" and trig_id == "geo-store" and geo_data and not geo_data.get("error")):
        
        route_out = {}
        ui_out = {"areaBusy": True}
        lat, lon, zoom = None, None, None
        addr = ""
        try:
            if trig_id == "btn-area":
                HOURLY_CACHE.clear()
                om_api_cache.clear()
                forecast_cache.clear()
                temp_cache.clear()
                api_cache.clear()
            
            if trig_id in ("btn-search", "q"):
                g = smart_geocode(q_val or "", lang_code)
                if not g:
                    status_out = {"type": "error", "data": {"key": "toast_err", "mode": "explore"}}
                    ui_out = {"areaBusy": False}
                    ts_out = datetime.now(fallback_tz).strftime("%H:%M:%S")
                    return make_return()
                lat, lon, addr, vp = g
                zoom = SEARCHED_ZOOM
                if vp:
                    try:
                        ne, sw = vp.get("northeast"), vp.get("southwest")
                        if ne and sw:
                            span_lat = abs(float(ne["lat"]) - float(sw["lat"]))
                            span_lon = abs(float(ne["lng"]) - float(sw["lng"]))
                            span = max(span_lat, span_lon, 0.001)
                            zoom = 11.5 - math.log2(span * 111)
                            zoom = max(4.5, min(14.0, zoom))
                    except Exception:
                        pass
                view_out = {"center": [lat, lon], "zoom": zoom}
                explore_out = {"coord": (lat, lon), "addr": addr}
                q_out = "" 
            elif trig_id == "geo-store":
                lat, lon = geo_data["lat"], geo_data["lon"]
                user_loc_out = (lat, lon)
                addr = reverse_geocode(lat, lon, lang_code) or f"({lat:.4f}, {lon:.4f})"
                zoom = 14
                view_out = {"center": [lat, lon], "zoom": zoom}
                explore_out = {"coord": (lat, lon), "addr": addr}
            elif trig_id == "btn-area":
                c = (current_view or {}).get("center", BASE_CENTER)
                z = (current_view or {}).get("zoom", BASE_ZOOM)
                lat, lon, zoom = c[0], c[1], z
                addr = t(lang, "map_center")
                explore_out = {} 
            scan_bounds = _bounds_from_center_zoom(lat, lon, zoom)
            wx_points = get_weather_data_for_bounds(scan_bounds, zoom)
            heatmap_out = wx_points
            forecast_data = get_point_forecast(lat, lon, lang)
            status_out = {"type": "explore", "data": {
                "addr": addr,
                "weather_key": forecast_data.get("key"),
                "temp": forecast_data.get("temp"),
                "forecast": forecast_data.get("forecast")
            }}
            offset_sec = forecast_data.get("offset_sec", 0)
            ts_timezone = timezone(timedelta(seconds=offset_sec))
            ts_out = datetime.now(ts_timezone).strftime("%H:%M:%S")
            ui_out = {"areaBusy": False}
            return make_return()
        except Exception as e:
            logging.error(f"Explore mode error: {e}", exc_info=True)
            status_out = {"type": "error", "data": {"key": "toast_err", "mode": "explore"}}
            ts_out = datetime.now(fallback_tz).strftime("%H:%M:%S")
            ui_out = {"areaBusy": False}
            return make_return()

    if mode == "route" and trig_id in ("btn-plan", "dst", "src", "btn-locate-src", "geo-store"):
        if trig_id in ("btn-plan", "dst", "src"):
            if not src_val or not dst_val:
                status_out = {"type": "error", "data": {"key": "toast_err", "mode": "route"}}
                ts_out = datetime.now(fallback_tz).strftime("%H:%M:%S")
                return make_return()
            
            heatmap_out = [] 
            route_out = {} 
            explore_out = {}
            
            g1 = smart_geocode(src_val, lang_code)
            g2 = smart_geocode(dst_val, lang_code)
            
            if not g1 or not g2:
                status_out = {"type": "error", "data": {"key": "toast_err", "mode": "route"}}
                ts_out = datetime.now(fallback_tz).strftime("%H:%M:%S")
                return make_return()

            o_coord, d_coord = (g1[0],g1[1]), (g2[0],g2[1])
            o_addr, d_addr = g1[2], g2[2] 
            d_lat, d_lon = d_coord
            
            d_forecast = get_point_forecast(d_lat, d_lon, lang)
            o_forecast = get_point_forecast(o_coord[0], o_coord[1], lang)
            offset_sec = o_forecast.get("offset_sec", d_forecast.get("offset_sec", 28800)) 
            
            if HAS_GMAP:
                raw_routes = google_routes_with_alts(o_coord, d_coord, travel_mode, lang)
                alert_prefix = ""
            else:
                raw_routes = osrm_route(o_coord, d_coord, travel_mode)
                alert_prefix = f"[{t(lang,'no_gmap_key')}] "

            ts_timezone = timezone(timedelta(seconds=offset_sec))
            ts_out = datetime.now(ts_timezone).strftime("%H:%M:%S")

            if not raw_routes:
                status_out = {"type": "error", "data": {"key": "no_route", "mode": "route", 
                              "d_lvl_key": d_forecast.get("key"), "d_temp": d_forecast.get("temp")}}
                return make_return()
            
            scored = []
            for r in raw_routes:
                poly = r.get("overview_polyline", {}).get("points")
                if not poly: continue
                pts = _decode_polyline(poly)
                if not pts: continue
                lats = [p[0] for p in pts]; lons = [p[1] for p in pts]
                flags, idxs = route_rain_flags_concurrent(lats, lons, lang)
                denom = max(1, len(flags))
                risk = sum(1.0 for f in flags if f) / denom
                scored.append({"route": {"lats": lats, "lons": lons}, "risk": risk, "flags": flags, "idxs": idxs})
            if not scored:
                status_out = {"type": "error", "data": {"key": "no_route", "mode": "route", 
                              "d_lvl_key": d_forecast.get("key"), "d_temp": d_forecast.get("temp")}}
                return make_return()

            scored.sort(key=lambda x: x["risk"])
            best = scored[0]; others = scored[1:] if HAS_GMAP else []
            route_out = {
                "origin": {"addr": o_addr, "coord": o_coord},
                "dest": {"addr": d_addr, "coord": d_coord},
                "best": best,
                "others": [{"route": x["route"], "risk": x["risk"]} for x in others]
            }
            risk_percent = round(best["risk"] * 100)
            status_out = {"type": "route", "data": {
                "o_addr": o_addr, "d_addr": d_addr, 
                "d_lvl_key": d_forecast.get("key"), "d_temp": d_forecast.get("temp"), 
                "risk": risk_percent, "prefix": alert_prefix
            }}
            c_lat, c_lon, zoom = bbox_center(best["route"]["lats"], best["route"]["lons"])
            view_out = {"center": [c_lat, c_lon], "zoom": zoom}
            src_out = "" 
            dst_out = "" 
            return make_return()

    if trig_id == "map":
        if not relayout_data: raise PreventUpdate
        center = current_view.get("center", BASE_CENTER)
        zoom = current_view.get("zoom", BASE_ZOOM)
        mb_center = relayout_data.get("mapbox.center")
        mb_zoom = relayout_data.get("mapbox.zoom")
        if mb_center:
            center = [mb_center["lat"], mb_center["lon"]]
        if mb_zoom:
            zoom = float(mb_zoom)
        center, zoom = clamp_view_to_tw(center, zoom)
        view_out = {"center": center, "zoom": zoom}
        return make_return()

    raise PreventUpdate

@app.callback(
    Output("src", "value", allow_duplicate=True),
    Input("geo-store", "data"),
    State("mode-store", "data"),
    State("lang-store", "data"),
    prevent_initial_call=True
)
def fill_src_from_locate(geo_data, mode, lang):
    if mode == "route" and geo_data and not geo_data.get("error"):
        lat, lon = geo_data["lat"], geo_data["lon"]
        lang_code = LANG_MAP.get(lang, "zh-TW")
        addr = reverse_geocode(lat, lon, lang_code)
        if addr:
            return addr
    return no_update

@app.callback(
    Output("map","figure"),
    Output("legend-a", "style"),
    Output("legend-scale-container", "children"),
    Input("basemap","value"),
    Input("explore-store","data"),
    Input("route-store","data"),
    Input("view-store", "data"),
    Input("mode-store", "data"),
    Input("rain-heatmap-store", "data"),
    Input("lang-store", "data"),
)
def draw_map(style, explore, route, view, mode, heatmap_data, lang):
    center = (view or {}).get("center", BASE_CENTER)
    zoom   = float((view or {}).get("zoom", BASE_ZOOM))
    style_val = (style or "low")
    map_style = "carto-positron" if style_val == "low" else "open-street-map"
    center, zoom = clamp_view_to_tw(center, zoom)
    fig = base_map_figure(center=center, zoom=zoom, style=map_style)
    
    if mode == "explore":
        legend_style = {"display": "flex"}
        legend_children = [
            html.Div(className="legend-scale", children=[
                html.Span(t(lang, "legend_light")),
                html.Span(t(lang, "legend_heavy"))
            ]),
            html.Div(className="legend-bar", style={"backgroundImage": css_gradient_from_colorscale(HEATMAP_COLORSCALE)}),
        ]
        heatmap_data = heatmap_data or []
        if heatmap_data:
            lats = [p[0] for p in heatmap_data]
            lons = [p[1] for p in heatmap_data]
            zs = [p[2] for p in heatmap_data]
            
            fig.add_trace(go.Densitymapbox(
                lat=lats, lon=lons, z=zs,
                radius=_get_radius_for_zoom(zoom),
                colorscale=HEATMAP_COLORSCALE,
                zmin=VISUAL_MM_MIN,
                zmax=HEATMAP_MAX_MM,
                showscale=False,
                opacity=0.65        
            ))
        if explore and explore.get("coord"):
            lat,lon = explore["coord"]
            fig.add_trace(go.Scattermapbox(
                lat=[lat], lon=[lon], mode="markers",
                marker=dict(size=16, color="rgba(239,68,68,.95)", symbol="circle"),
                name="point", hovertext=explore.get("addr"), hoverinfo="text"
            ))
    elif mode == "route":
        legend_style = {"display": "flex"}
        legend_children = [
            html.Div(className="legend-scale-route", children=[
                html.Div(className="swatch", style={"background-color": COLOR_DRY}),
                html.Span(t(lang, "dry")),
                html.Div(className="swatch", style={"background-color": COLOR_WET}),
                html.Span(t(lang, "rain")),
            ]),
        ]
        if route and route.get("best"):
            best = route["best"]
            br = best["route"]; flags = best.get("flags",[]); idxs = best.get("idxs",[])
            segments = segments_by_flags(br["lats"], br["lons"], flags, idxs)
            for seg in segments:
                fig.add_trace(go.Scattermapbox(
                    lat=seg["lats"], lon=seg["lons"], mode="lines",
                    line=dict(width=8, color=seg["color"]),
                    name=t(lang, "best"), showlegend=False, hoverinfo="none"
                ))
            for x in (route.get("others") or []):
                rr=x["route"]
                fig.add_trace(go.Scattermapbox(
                    lat=rr["lats"], lon=rr["lons"], mode="lines",
                    line=dict(width=4, color="rgba(156,163,175,0.6)"),
                    name=t(lang, "others"), showlegend=False, hoverinfo="none"
                ))
            try:
                o=route.get("origin",{}).get("coord"); d=route.get("dest",{}).get("coord")
                if o:
                    fig.add_trace(go.Scattermapbox(lat=[o[0]], lon=[o[1]], mode="markers",
                                           marker=dict(size=16, color="rgba(239,68,68,.95)", symbol="circle"),
                                           name=t(lang, "origin"), hovertext=route.get("origin",{}).get("addr"),
                                           showlegend=False))
                if d:
                    fig.add_trace(go.Scattermapbox(lat=[d[0]], lon=[d[1]], mode="markers",
                                           marker=dict(size=16, color="rgba(239,68,68,.95)", symbol="circle"),
                                           name=t(lang, "dest"), hovertext=route.get("dest",{}).get("addr"),
                                           showlegend=False))
            except Exception as e:
                logging.error(f"Failed to draw O/D markers: {e}")
    else:
        legend_style = {"display": "none"}
        legend_children = []
        
    return fig, legend_style, legend_children

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8050"))
    # 本地測試使用 127.0.0.1；Render 部署用 0.0.0.0
    app.run(host=os.getenv("HOST", "0.0.0.0"), port=port, debug=False)