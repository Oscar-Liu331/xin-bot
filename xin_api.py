import json
import re
from pathlib import Path
from typing import List, Dict, Any

import requests
from math import radians, sin, cos, asin, sqrt

from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse


CITY_PATTERN = (
    r"(台北市|臺北市|新北市|桃園市|臺中市|台中市|臺南市|台南市|高雄市|"
    r"基隆市|新竹市|嘉義市|新竹縣|苗栗縣|彰化縣|南投縣|雲林縣|嘉義縣|"
    r"屏東縣|宜蘭縣|花蓮縣|臺東縣|台東縣|澎湖縣|金門縣|連江縣)"
)

ADDR_HEAD_RE = re.compile(rf"^{CITY_PATTERN}(.*?(區|鄉|鎮|市))")

SECTION = "pro"
DATASET_PATTERN = f"elearn_{SECTION}_*_*_dataset.json"
TOP_K = 5  

XIN_POINTS_FILE = Path("xin_points.json")
UNITS_FILE = Path("wellbeing_elearn_pro_all_with_articles.json")

MENTAL_KEYWORDS = [
    "憂鬱", "情緒低落", "心情不好", "心情低落",
    "心情",      
    "低落",      
    "難過",     
    "沮喪", "沒動力",
    "焦慮", "緊張", "恐慌", "壓力", "職場壓力", "睡不著", "失眠",
    "孤單", "寂寞",
    "婆媳", "婆婆", "公婆", "家庭衝突", "家庭關係",
    "夫妻", "婚姻", "親子衝突", "親子關係",

    "小孩", "孩子", "幼兒", "青少年",
    "教養", "親子", "親子衝突", "親子關係",
    "吵架", "頂嘴", "哭鬧", "情緒失控", "脾氣",
]

STOP_WORDS = [
    "我", "你", "他", "她", "它", "我們", "你們", "他們",
    "最近", "一直", "覺得", "有點", "有一點",
    "如果", "好像", "是不是",
    "該怎麼辦", "怎麼辦", "怎麼做", "該怎麼做",
    "可以", "覺得", "自己",
    "的", "了", "呢", "嗎", "吧"
]
def build_nearby_points_response(address: str, results):
    """
    把 find_nearby_points 的結果轉成 JSON-friendly 結構
    """
    if not results:
        return {
            "type": "xin_points",
            "address": address,
            "points": [],
            "message": f"在「{address}」5 公里內沒有找到心據點"
        }

    points = []
    for p, d in results:
        points.append({
            "title": p.get("title"),
            "address": p.get("address"),
            "tel": p.get("tel"),
            "distance_km": round(d, 2),
        })

    return {
        "type": "xin_points",
        "address": address,
        "points": points
    }


def build_recommendations_response(query: str, results: List[Dict[str, Any]]):
    """
    把 search_units 的結果轉成 JSON-friendly 結構
    """
    if not results:
        return {
            "type": "course_recommendation",
            "query": query,
            "results": [],
            "message": "目前找不到很符合的課程，可以試著用：婆媳、壓力、憂鬱、失眠… 等詞再試試看。"
        }

    items = []
    for r in results[:TOP_K]:
        title = r.get("title") or "(無標題)"
        section_title = r.get("section_title") or "(未分類小節)"
        score = r.get("_score", 0.0)

        is_article = bool(r.get("is_article"))
        youtube_url = r.get("youtube_url")
        entry: Dict[str, Any] = {
            "section_title": section_title,
            "title": title,
            "score": score,
        }

        if is_article:
            article_url = r.get("article_url") or r.get("url")
            content_text = (r.get("content_text") or "").replace("\n", " ")
            snippet = content_text[:100] + ("..." if len(content_text) > 100 else "")

            entry["type"] = "article"
            entry["article_url"] = article_url
            entry["snippet"] = snippet

        else:
            seg = r.get("_best_segment")
            if seg:
                start_sec = seg.get("start_sec", 0.0)
                start_str = format_time(start_sec)
                seg_text = seg.get("text", "") or ""
                hint = f"該單元在 {start_str} 有提到：「{seg_text[:30]}...」"
            else:
                hint = "字幕裡沒有特別命中關鍵句，可以從頭開始看。"

            entry["type"] = "video"
            entry["youtube_url"] = youtube_url
            entry["hint"] = hint

        items.append(entry)

    return {
        "type": "course_recommendation",
        "query": query,
        "results": items
    }

def load_xin_points() -> List[Dict[str, Any]]:
    try:
        data = json.loads(XIN_POINTS_FILE.read_text("utf-8"))
        return data.get("data", [])
    except Exception as e:
        print(f"[xin] ⚠️ 心據點載入失敗：{e}")
        return []
    
def haversine_km(lon1, lat1, lon2, lat2) -> float:
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    return 6371 * c  # km

def geocode_address(address: str):
    if not address:
        return None

    def try_geocode(addr: str):
        url = "https://nominatim.openstreetmap.org/search"
        params = {"q": addr, "format": "json", "limit": 1}
        headers = {"User-Agent": "xin-bot/1.0"}
        try:
            r = requests.get(url, params=params, headers=headers, timeout=5)
            r.raise_for_status()
            data = r.json()
            if data:
                lat = float(data[0]["lat"])
                lon = float(data[0]["lon"])
                print(f"[geocode] 命中：'{addr}' -> lat={lat}, lon={lon}")
                return lat, lon
        except Exception as e:
            print(f"[geocode] 錯誤：{e}")
        return None

    print(f"[geocode] 嘗試：{address}")
    result = try_geocode(address)
    if result:
        return result

    if "臺" in address:
        addr2 = address.replace("臺", "台")
        print(f"[geocode] 嘗試：{addr2}")
        result = try_geocode(addr2)
        if result:
            return result

    addr3 = re.sub(r"\d+號.*", "", address)
    if addr3 != address:
        print(f"[geocode] 嘗試（去號）：{addr3}")
        result = try_geocode(addr3)
        if result:
            return result

    addr4 = re.sub(r"\d+弄.*", "", address)
    if addr4 != address:
        print(f"[geocode] 嘗試（去弄）：{addr4}")
        result = try_geocode(addr4)
        if result:
            return result

    addr5 = re.sub(r"\d+巷.*", "", address)
    if addr5 != address:
        print(f"[geocode] 嘗試（去巷）：{addr5}")
        result = try_geocode(addr5)
        if result:
            return result

    m = re.match(
        r"(台北市|臺北市|新北市|桃園市|臺中市|台中市|臺南市|台南市|高雄市|"
        r"基隆市|新竹市|嘉義市|新竹縣|苗栗縣|彰化縣|南投縣|雲林縣|嘉義縣|"
        r"屏東縣|宜蘭縣|花蓮縣|臺東縣|台東縣|澎湖縣|金門縣|連江縣)"
        r"(.+?(區|市|鎮|鄉))",
        address
    )
    if m:
        addr6 = m.group(1) + m.group(2)
        print(f"[geocode] 嘗試（市+區/鄉/鎮/市）：{addr6}")
        result = try_geocode(addr6)
        if result:
            return result

    print(f"[geocode] 完全查不到：{address}")
    return None
    
def find_nearby_points(lat, lon, max_km=5, top_k=5):
    points = load_xin_points()
    results = []

    for p in points:
        if p.get("lat") and p.get("lon"):
            d = haversine_km(lon, lat, p["lon"], p["lat"])
            if d <= max_km:
                results.append((p, d))

    results.sort(key=lambda x: x[1])
    return results[:top_k]

def load_all_units() -> List[Dict[str, Any]]:
    data = json.loads(UNITS_FILE.read_text("utf-8"))
    raw_units = data.get("units", [])

    units: List[Dict[str, Any]] = []

    for u in raw_units:
        u = dict(u)  

        section_title = u.get("section_title") or ""

        subtitle_texts = " ".join(
            seg.get("text", "") for seg in u.get("subtitles", []) or []
        )

        content_text = u.get("content_text", "") or ""

        search_text = " ".join(
            s for s in [
                section_title,
                u.get("title") or "",
                content_text,
                subtitle_texts,
            ] if s
        )

        u["_search_text"] = search_text

        units.append(u)

    print(f"[load] ✅ 共載入 {len(units)} 個單元（含影片 + 文章）")
    return units

def extract_address_from_query(q: str) -> str:
    original = q

    if "附近" in q:
        q = q.split("附近")[0]

    for kw in ["心據點", "門診", "看診"]:
        if kw in q:
            q = q.split(kw)[0]

    prefixes = ["我住在", "我住", "家在", "家住", "住在", "住", "在"]
    q = q.strip()
    for p in prefixes:
        if q.startswith(p):
            q = q[len(p):].strip()
            break

    tail_words = ["有沒有", "有嗎", "嗎", "呢", "啊", "啦"]
    for t in tail_words:
        if q.endswith(t):
            q = q[: -len(t)].strip()

    q = q.strip(" ?？!")

    if len(q) < 4:
        return ""

    print(f"[debug] extract_address_from_query: '{original}' -> '{q}'")
    return q

def normalize_query(q: str) -> List[str]:
    q = q.strip()
    if not q:
        return []

    terms: List[str] = []

    for kw in MENTAL_KEYWORDS:
        if kw in q and kw not in terms:
            terms.append(kw)

    parts = re.split(r"[，。！!？?\s、；;:：]+", q)
    for part in parts:
        part = part.strip()
        if not part or part in STOP_WORDS:
            continue

        if re.fullmatch(r"[\u4e00-\u9fff]+", part):
            n = len(part)
            for size in range(2, min(4, n) + 1):
                for i in range(0, n - size + 1):
                    gram = part[i:i+size]
                    if gram not in STOP_WORDS and gram not in terms:
                        terms.append(gram)
        else:
            if len(part) >= 2 and part not in terms and part not in STOP_WORDS:
                terms.append(part)

    if not terms:
        terms = [q]

    return terms

def score_unit(unit, query_terms, core_terms):
    text = unit.get("_search_text", "") or ""
    if not text:
        return 0.0, None

    title = (unit.get("section_title") or "") + (unit.get("title") or "")
    subtitles = unit.get("subtitles", [])

    title_core = any(c in title for c in core_terms)

    subtitle_core = False
    for seg in subtitles:
        seg_text = seg.get("text", "") or ""
        if any(c in seg_text for c in core_terms):
            subtitle_core = True
            break

    if not (title_core or subtitle_core):
        return 0.0, None

    total_hits = sum(text.count(term) for term in query_terms)
    if total_hits < 3:
        return 0.0, None

    total_core_hits = sum(text.count(c) for c in core_terms)

    if not title_core and total_core_hits < 3:
        return 0.0, None

    best_seg = None
    best_seg_score = 0
    for seg in subtitles:
        seg_text = seg.get("text", "") or ""
        seg_score = sum(seg_text.count(t) for t in query_terms)
        if seg_score > best_seg_score:
            best_seg_score = seg_score
            best_seg = seg

    title_core_hits = sum(1 for c in core_terms if c in title)

    final_score = (
        total_core_hits * 4 +      # 核心詞在全文出現越多，越相關
        title_core_hits * 6 +      # 標題有核心詞，加大權重
        total_hits * 1 +           # 其他詞也略加分
        best_seg_score * 2         # 有一段字幕特別集中，也加分
    )

    return final_score, best_seg

def format_time(seconds: float) -> str:
    s = int(seconds)
    h = s // 3600
    m = (s % 3600) // 60
    sec = s % 60
    if h > 0:
        return f"{h:02d}:{m:02d}:{sec:02d}"
    return f"{m:02d}:{sec:02d}"

def search_units(units: List[Dict[str, Any]], query: str, top_k: int = TOP_K):
    terms = normalize_query(query)
    if not terms:
        return []
    
    core_terms: List[str] = [t for t in terms if t in MENTAL_KEYWORDS]

    if not core_terms:
        long_terms = sorted([t for t in terms if len(t) >= 2],
                            key=lambda x: len(x),
                            reverse=True)
        core_terms = long_terms[:2]

    print(f"[debug] query={query} → terms={terms} | core_terms={core_terms}")

    results = []
    for u in units:
        score, best_seg = score_unit(u, terms, core_terms)
        if score > 0:
            r = dict(u)
            r["_score"] = score
            r["_best_segment"] = best_seg
            results.append(r)

    results.sort(key=lambda x: x["_score"], reverse=True)
    return results

# ---------- 互動主迴圈 ----------
app = FastAPI(title="心快活課程推薦 API")

# 若前端網頁會跨網域呼叫，可開 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 上線時建議改成你的網域
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ▼▼ 新增：讓 /static 底下可以直接抓檔案 ▼▼
app.mount("/static", StaticFiles(directory="static"), name="static")


# ▼▼ 新增：讓 "/" 直接回 index.html ▼▼
@app.get("/", include_in_schema=False)
def serve_index():
    # static/index.html
    return FileResponse("static/index.html")

# 啟動時就先載入所有單元
UNITS_CACHE: List[Dict[str, Any]] = load_all_units()

# 簡單放在記憶體的聊天紀錄（server 重啟會清空）
HISTORY: List[Dict[str, Any]] = []


class ChatRequest(BaseModel):
    query: str


class NearbyRequest(BaseModel):
    address: str


class RecommendRequest(BaseModel):
    query: str


@app.get("/ping")
def ping():
    return {"status": "ok"}


@app.post("/chat")
def chat(req: ChatRequest):
    q = req.query.strip()

    resp: Dict[str, Any]

    # 1) 判斷是否為「心據點」/「看診」詢問
    if ("附近" in q) and ("心據點" in q or "看診" in q or "門診" in q):
        addr = extract_address_from_query(q)
        if not addr:
            resp = {
                "type": "xin_points",
                "address": None,
                "points": [],
                "message": "我有點抓不到地址，請嘗試輸入完整地址，例如：台南市東區大學路1號"
            }
        else:
            geo = geocode_address(addr)
            if not geo:
                resp = {
                    "type": "xin_points",
                    "address": addr,
                    "points": [],
                    "message": f"查不到「{addr}」這個地址，請改成更正式的寫法試試看"
                }
            else:
                lat, lon = geo
                results = find_nearby_points(lat, lon, max_km=5, top_k=TOP_K)
                resp = build_nearby_points_response(addr, results)

    # 2) 直接輸入完整地址（開頭就是「台南市xxx」之類）
    elif ADDR_HEAD_RE.match(q):
        addr = q
        geo = geocode_address(addr)
        if not geo:
            resp = {
                "type": "xin_points",
                "address": addr,
                "points": [],
                "message": f"查不到「{addr}」這個地址，請改成更正式的寫法試試看"
            }
        else:
            lat, lon = geo
            results = find_nearby_points(lat, lon, max_km=5, top_k=TOP_K)
            resp = build_nearby_points_response(addr, results)

    # 3) 其他情況：當作課程推薦查詢
    else:
        results = search_units(UNITS_CACHE, q, top_k=TOP_K)
        resp = build_recommendations_response(q, results)

    # --- 在這裡記錄歷史 ---
    HISTORY.append({
        "query": q,
        "response": resp,
    })
    # 最多保留 50 筆，太舊的丟掉
    if len(HISTORY) > 50:
        HISTORY.pop(0)

    return resp

@app.get("/history")
def get_history():
    return {
        "items": HISTORY
    }

@app.post("/nearby")
def nearby(req: NearbyRequest):
    addr = req.address.strip()
    if not addr:
        return {
            "type": "xin_points",
            "address": None,
            "points": [],
            "message": "請提供完整地址，例如：台南市東區大學路1號"
        }

    geo = geocode_address(addr)
    if not geo:
        return {
            "type": "xin_points",
            "address": addr,
            "points": [],
            "message": f"查不到「{addr}」這個地址，請改成更正式的寫法試試看"
        }

    lat, lon = geo
    results = find_nearby_points(lat, lon, max_km=5, top_k=TOP_K)
    return build_nearby_points_response(addr, results)


@app.post("/recommend")
def recommend(req: RecommendRequest):
    q = req.query.strip()
    results = search_units(UNITS_CACHE, q, top_k=TOP_K)
    return build_recommendations_response(q, results)
