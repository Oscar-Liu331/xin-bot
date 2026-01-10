import time
import json
import re
import os
import requests
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
from math import radians, sin, cos, asin, sqrt
import urllib.parse

from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from typing import Optional

from langdetect import detect, LangDetectException
from deep_translator import GoogleTranslator

# --- 常數設定 ---
CITY_PATTERN = (
    r"(台北市|臺北市|新北市|桃園市|臺中市|台中市|臺南市|台南市|高雄市|"
    r"基隆市|新竹市|嘉義市|新竹縣|苗栗縣|彰化縣|南投縣|雲林縣|嘉義縣|"
    r"屏東縣|宜蘭縣|花蓮縣|臺東縣|台東縣|澎湖縣|金門縣|連江縣)"
)
ADDR_HEAD_RE = re.compile(rf"^{CITY_PATTERN}(.*?(區|鄉|鎮|市))")
TOP_K = 5  

XIN_POINTS_FILE = Path("xin_points.json")
UNITS_FILE = Path("wellbeing_elearn_pro_all_with_articles.json")

CORPUS_VECTORS = None 

JINA_API_URL = "https://api.jina.ai/v1/embeddings"
JINA_API_KEY = None

KEYWORDS_FILE = Path("keywords.json")
KEYWORDS_DATA = {} 
MENTAL_KEYWORDS = [] 
STOP_WORDS = []

# 翻譯用快取
TRANSLATION_CACHE = {}

MODEL_CONFIGS = {
    "v4": {
        "api_model_name": "jina-embeddings-v4", # 最新發布的版本
        "vector_filename": "vectors_v4.json",
        "dimensions": 2048 # ⚠️ 注意：v4 預設維度是 2048
    },
    "v3": {
        "api_model_name": "jina-embeddings-v3",
        "vector_filename": "vectors_v3.json",
        "dimensions": 1024
    },
    "v2-zh": {
        "api_model_name": "jina-embeddings-v2-base-zh",
        "vector_filename": "vectors_v2_zh.json",
        "dimensions": 768
    }
}

CURRENT_MODEL_KEY = "v3"

CURRENT_CONFIG = MODEL_CONFIGS[CURRENT_MODEL_KEY]

VECTORS_FILE = Path(CURRENT_CONFIG["vector_filename"])

# --- 核心工具函式 ---

def detect_language(text: str) -> str:
    """
    語言偵測最終版
    """
    if not text: return "zh-TW"
    
    # 1. [絕對優先] 檢查常見日文特徵字
    if re.search(r'[のはですがますくださいてにを気]', text):
        return "ja"
    if re.search(r'[\u3040-\u309f\u30a0-\u30ff]', text):
        return "ja"

    # 2. 檢查韓文
    if re.search(r'[\uac00-\ud7af]', text):
        return "ko"

    # 3. 檢查純英文
    clean_text = re.sub(r'[0-9\s,.?!:;\'"()\[\]]', '', text)
    if clean_text and all(ord(c) < 128 for c in clean_text):
        return "en"

    # 4. 檢查中文
    if re.search(r'[\u4e00-\u9fa5]', text):
        return "zh-TW"

    try:
        lang = detect(text)
        if lang.startswith("zh"): return "zh-TW"
        if lang == 'ja': return 'ja'
        return lang
    except LangDetectException:
        return "zh-TW"

def translate_text(text: str, target: str) -> str:
    if not text: return ""
    if target == "zh-TW" and detect_language(text) == "zh-TW":
        return text
    
    cache_key = f"{text}_{target}"
    if cache_key in TRANSLATION_CACHE:
        return TRANSLATION_CACHE[cache_key]
    
    try:
        translator = GoogleTranslator(source='auto', target=target)
        result = translator.translate(text)
        
        # 防呆
        if result == text and len(text) > 5 and target != "zh-TW":
             clean = re.sub(r"[【】《》「」]", " ", text).strip()
             if clean != text:
                 retry = translator.translate(clean)
                 if retry != clean:
                     result = retry

        TRANSLATION_CACHE[cache_key] = result
        return result
    except Exception as e:
        print(f"!!! [Translate Error] Text: {text[:10]}... | Error: {e}")
        return text 

def load_keywords_from_json():
    global KEYWORDS_DATA, MENTAL_KEYWORDS, STOP_WORDS
    try:
        if KEYWORDS_FILE.exists():
            with open(KEYWORDS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                KEYWORDS_DATA = data.get("mental_keywords", {})
                all_kws = []
                for category_list in KEYWORDS_DATA.values():
                    all_kws.extend(category_list)
                MENTAL_KEYWORDS = list(set(all_kws))
                STOP_WORDS = data.get("stop_words", [])
            print(f"[load] ✅ 分類載入成功。共 {len(KEYWORDS_DATA)} 個類別。")
    except Exception as e:
        print(f"[load] ❌ 分類載入失敗: {e}")

load_keywords_from_json()

# 請確保全域變數宣告包含 VECTOR_CACHE
# VECTOR_CACHE = {} 

def init_vector_model():
    global VECTOR_CACHE, JINA_API_KEY
    
    # 你的 API KEY (建議之後還是換成環境變數比較安全)
    JINA_API_KEY = os.environ.get("JINA_API_KEY")

    if not JINA_API_KEY:
        print("[init] ⚠️ 警告：找不到 JINA_API_KEY，語意搜尋將無法運作！")
    else:
        print("[init] ✅ Jina API Key 已設定")

    print("[init] 🚀 正在初始化多模型系統...")
    
    # 初始化快取字典
    VECTOR_CACHE = {} 

    # 迴圈讀取 MODEL_CONFIGS 裡面的每一組設定
    for key, config in MODEL_CONFIGS.items():
        fname = Path(config["vector_filename"])
        expected_dim = config["dimensions"]
        
        if fname.exists():
            try:
                print(f"   Using > 正在載入 [{key}] 向量檔: {fname} ...")
                
                with open(fname, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    matrix = np.array(data, dtype="float32")
                
                # 防呆檢查：檢查維度是否正確
                current_dim = matrix.shape[1] if len(matrix) > 0 else 0
                if current_dim != expected_dim:
                    print(f"   ⚠️ 警告：[{key}] 檔案維度 ({current_dim}) 與設定 ({expected_dim}) 不符！可能需要重新生成。")
                
                # 存入快取
                VECTOR_CACHE[key] = matrix
                print(f"   ✅ [{key}] 載入成功 (共 {len(matrix)} 筆, 維度 {current_dim})")
                
            except Exception as e:
                print(f"   ❌ [{key}] 讀取失敗: {e}")
        else:
            print(f"   ⚠️ [{key}] 找不到檔案 {fname}，跳過此版本。")
            
    print(f"[init] 完成！共載入 {len(VECTOR_CACHE)} 個模型版本。\n")

def get_jina_embedding(text, model_name):
    if not JINA_API_KEY:
        raise Exception("JINA_API_KEY not set")
    
    headers = { "Content-Type": "application/json", "Authorization": f"Bearer {JINA_API_KEY}" }
    
    payload = { 
        "model": model_name, 
        "input": [text] 
    }
    
    # v3 和 v4 建議加上 task 參數
    if "v3" in model_name or "v4" in model_name:
        payload["task"] = "retrieval.passage"

    try:
        resp = requests.post(JINA_API_URL, headers=headers, json=payload, timeout=10)
        resp.raise_for_status()
        return resp.json()["data"][0]["embedding"]
    except Exception as e:
        print(f"[Jina API Error] {e}")
        return None

def search_units_semantic(query: str, model_key: str, top_k: int = 5):
    # 1. 從全域快取中取得對應版本的向量矩陣
    # 請確保你有宣告 global VECTOR_CACHE
    corpus = VECTOR_CACHE.get(model_key)
    
    if corpus is None:
        print(f"[search] 錯誤：找不到版本 {model_key} 的向量資料")
        return []
    
    config = MODEL_CONFIGS.get(model_key)
    if not config: return []

    try:
        # 2. 呼叫指定版本的 API 取得 Query Vector
        # 注意：這裡會呼叫 get_jina_embedding，傳入對應的模型名稱 (例如 jina-embeddings-v3)
        query_vec_list = get_jina_embedding(query, config["api_model_name"])
        
        if not query_vec_list: return []
        
        query_vec = np.array(query_vec_list, dtype="float32")
        
        # 3. 計算相似度 (矩陣運算)
        scores = np.dot(corpus, query_vec)
        top_indices = np.argsort(scores)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            score = float(scores[idx])
            # 門檻值可以自己微調
            if score > 0.25: 
                r = dict(UNITS_CACHE[idx])
                r["_score"] = score
                r["_best_segment"] = None
                results.append(r)
        return results
    except Exception as e:
        print(f"[search] 向量搜尋發生錯誤: {e}")
        return []
    
def detect_pagination_intent(q: str) -> bool:
    q = q.lower().strip()
    keywords = [
        "給我後五個", "給我下五個", "後五個", "下五個", "下一頁", "更多推薦", 
        "next 5", "show me more", "more results", 
        "次の5件", "もっと見る", "続き", "最後の5つ", "最後の5つをください"
    ]
    return any(kw in q for kw in keywords)

def extract_address_from_query(q: str) -> str:
    original = q
    if "附近" in q: q = q.split("附近")[0]
    for kw in ["心據點", "門診", "看診"]:
        if kw in q: q = q.split(kw)[0]
    prefixes = ["我住在", "我住", "家在", "家住", "住在", "住", "在"]
    q = q.strip()
    for p in prefixes:
        if q.startswith(p):
            q = q[len(p):].strip()
            break
    tail_words = ["有沒有", "有嗎", "嗎", "呢", "啊", "啦"]
    for t in tail_words:
        if q.endswith(t): q = q[: -len(t)].strip()
    q = q.strip(" ?？!")
    if len(q) < 4: return ""
    return q

def normalize_query(q: str):
    q = q.strip().lower()
    if not q: return [], [], []
    functional_words = ["文章", "影片", "想看", "給我", "只有", "只想看", "推薦", "影音", "播放", "查詢", "找", "有哪些", "介紹"]
    user_input_core = []
    category_expanded = []
    other_terms = []
    found_categories = set()
    for category, kws in KEYWORDS_DATA.items():
        for kw in kws:
            if kw in q:
                if kw not in user_input_core: user_input_core.append(kw)
                found_categories.add(category)
    for cat in found_categories:
        group_kws = KEYWORDS_DATA[cat]
        for kw in group_kws:
            if kw not in user_input_core and kw not in category_expanded: category_expanded.append(kw)
    temp_q = q
    for kw in user_input_core: temp_q = temp_q.replace(kw, " ") 
    for fw in functional_words: temp_q = temp_q.replace(fw, " ")
    parts = re.split(r"[，。！!？?\s、；;:：]+", temp_q)
    for part in parts:
        if len(part) >= 2 and part not in STOP_WORDS:
            if part not in other_terms: other_terms.append(part)
    return user_input_core, category_expanded, other_terms

def score_unit(unit, user_core, expanded_core, other_terms):
    title = (unit.get("section_title") or "") + (unit.get("title") or "")
    content = unit.get("content_text", "") or "" 
    if not title and not content: return 0.0, None
    score = 0.0
    for kw in user_core:
        if kw in title: score += 10.0
        cnt = content.count(kw)
        if cnt > 0: score += cnt * 4.0
    for kw in expanded_core:
        if kw in title: score += 5.0
        cnt = content.count(kw)
        if cnt > 0: score += cnt * 2.0
    for kw in other_terms:
        if kw in title: score += 1.0
        cnt = content.count(kw)
        if cnt > 0: score += cnt * 0.5
    subtitles = unit.get("subtitles", [])
    best_seg = None
    best_seg_score = 0
    has_core_list = []
    for seg in subtitles:
        seg_text = seg.get("text", "")
        hits = sum(1 for kw in user_core if kw in seg_text)
        if hits == 0: hits = sum(1 for kw in expanded_core if kw in seg_text) * 0.5 
        has_core = (hits > 0)
        has_core_list.append(has_core)
        if hits > best_seg_score:
            best_seg_score = hits
            best_seg = seg
    count_continuous_hits = 0
    if len(has_core_list) >= 3:
        for i in range(len(has_core_list) - 2):
            if has_core_list[i] and has_core_list[i+1] and has_core_list[i+2]: count_continuous_hits += 1
    score += count_continuous_hits * 2.0
    return score, best_seg

EP_TAG_RE = re.compile(r"(（上）|（下）|\(上\)|\(下\)|上篇|下篇|上集|下集)")
def get_episode_tag(title: str) -> Optional[str]:
    if not title: return None
    t = title.strip()
    if re.search(r"(（上）|\(上\)|上篇|上集)", t): return "上"
    if re.search(r"(（下）|\(下\)|下篇|下集)", t): return "下"
    return None

def get_base_key(section_title: str, title: str) -> str:
    s = (section_title or "").strip()
    t = (title or "").strip()
    t2 = EP_TAG_RE.sub("", t)
    t2 = re.sub(r"\s+", "", t2)
    s2 = re.sub(r"\s+", "", s)
    return f"{s2}||{t2}"

def reorder_episode_pairs(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[str, Dict[str, Any]] = {}
    for idx, r in enumerate(results):
        key = get_base_key(r.get("section_title"), r.get("title"))
        score = float(r.get("_score", 0.0))
        g = groups.get(key)
        if g is None:
            groups[key] = { "items": [], "best_score": score, "first_idx": idx }
            g = groups[key]
        g["items"].append(r)
        if score > g["best_score"]: g["best_score"] = score
    def item_rank(r: Dict[str, Any]) -> int:
        tag = get_episode_tag(r.get("title") or "")
        if tag == "上": return 0
        if tag == "下": return 1
        return 2
    for g in groups.values():
        g["items"].sort(key=lambda r: (item_rank(r), -float(r.get("_score", 0.0))))
    ordered_groups = sorted(groups.values(), key=lambda g: (-g["best_score"], g["first_idx"]))
    out: List[Dict[str, Any]] = []
    for g in ordered_groups: out.extend(g["items"])
    return out

def format_time(seconds: float) -> str:
    s = int(seconds)
    h = s // 3600
    m = (s % 3600) // 60
    sec = s % 60
    if h > 0: return f"{h:02d}:{m:02d}:{sec:02d}"
    return f"{m:02d}:{sec:02d}"

def search_units(units: List[Dict[str, Any]], query: str, top_k: int = TOP_K):
    user_core, expanded_core, other_terms = normalize_query(query)
    if not user_core and len(query) >= 2: user_core = [query]
    if not user_core and not other_terms: return []
    results = []
    for u in units:
        score, best_seg = score_unit(u, user_core, expanded_core, other_terms)
        if score > 0:
            r = dict(u)
            r["_score"] = score
            r["_best_segment"] = best_seg
            results.append(r)
    results.sort(key=lambda x: x["_score"], reverse=True)
    return results

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
    return 6371 * c 

def geocode_address(address: str):
    if not address: return None
    def try_geocode(addr: str):
        url = "https://nominatim.openstreetmap.org/search"
        params = {"q": addr, "format": "json", "limit": 1}
        headers = {"User-Agent": "xin-bot/1.0"}
        try:
            r = requests.get(url, params=params, headers=headers, timeout=5)
            r.raise_for_status()
            data = r.json()
            if data:
                return float(data[0]["lat"]), float(data[0]["lon"])
        except Exception: pass
        return None

    res = try_geocode(address)
    if res: return res
    if "臺" in address:
        res = try_geocode(address.replace("臺", "台"))
        if res: return res
    
    # 模糊搜尋
    addr3 = re.sub(r"\d+號.*", "", address)
    if addr3 != address:
        res = try_geocode(addr3)
        if res: return res
    
    m = re.match(
        r"(台北市|臺北市|新北市|桃園市|臺中市|台中市|臺南市|台南市|高雄市|"
        r"基隆市|新竹市|嘉義市|新竹縣|苗栗縣|彰化縣|南投縣|雲林縣|嘉義縣|"
        r"屏東縣|宜蘭縣|花蓮縣|臺東縣|台東縣|澎湖縣|金門縣|連江縣)"
        r"(.+?(區|市|鎮|鄉))",
        address
    )
    if m:
        addr6 = m.group(1) + m.group(2)
        res = try_geocode(addr6)
        if res: return res

    return None

def find_nearby_points(lat, lon, max_km=5, top_k=5):
    points = load_xin_points()
    results = []
    for p in points:
        if p.get("lat") and p.get("lon"):
            d = haversine_km(lon, lat, p["lon"], p["lat"])
            if d <= max_km: results.append((p, d))
    results.sort(key=lambda x: x[1])
    return results[:top_k]

def build_nearby_points_response(address: str, results):
    if not results:
        return {
            "type": "xin_points",
            "address": address,
            "points": [],
            "message": f"在「{address}」5 公里內沒有找到心據點"
        }

    points = []
    origin_encoded = urllib.parse.quote(address)

    for p, d in results:
        dest_address = p.get("address", "")
        dest_encoded = urllib.parse.quote(dest_address)
        map_url = f"https://www.google.com/maps/dir/?api=1&origin={origin_encoded}&destination={dest_encoded}&hl=zh-TW"
        points.append({
            "title": p.get("title"),
            "address": dest_address,
            "tel": p.get("tel"),
            "distance_km": round(d, 2),
            "map_url": map_url
        })

    return {
        "type": "xin_points",
        "address": address,
        "points": points
    }

def load_all_units() -> List[Dict[str, Any]]:
    data = json.loads(UNITS_FILE.read_text("utf-8"))
    raw_units = data.get("units", [])
    units = []
    for u in raw_units:
        u = dict(u)
        section_title = u.get("section_title") or ""
        subtitle_texts = " ".join(seg.get("text", "") for seg in u.get("subtitles", []) or [])
        content_text = u.get("content_text", "") or ""
        search_text = " ".join(s for s in [section_title, u.get("title") or "", content_text, subtitle_texts] if s)
        u["_search_text"] = search_text
        units.append(u)
    print(f"[load] ✅ 共載入 {len(units)} 個單元")
    return units

# --- 介面回應建構 ---
def build_recommendations_response(query: str, results: List[Dict[str, Any]], 
                                   offset: int = 0, limit: int = TOP_K, 
                                   target_lang: str = "zh-TW"):
    
    # 1. UI 模板
    ui = {}
    if target_lang == 'ja':
        ui = {
            "not_found": "条件に合うコンテンツが見つかりませんでした。「ストレス」、「不眠」、「不安」などのキーワードで試してみてください。",
            "found_msg": "📚 合計 {total} 件見つかりました（🎥 動画 {v_count}、📄 記事 {a_count}）\n表示中: {start}～{end} 件\n\nご相談内容に基づき、以下のコース/記事を検索しました：",
            "hint_prefix": "ヒント: ",
            "hint_default": "字幕に特定のキーワードは見つかりませんでした。最初からご覧ください。",
            "video_link": "🎥 リンク: ",
            "more_btn": "👉 「次の5件」をクリックしてもっと見る"
        }
    elif target_lang == 'en':
        ui = {
            "not_found": "Currently no relevant courses found. Try keywords like: stress, insomnia...",
            "found_msg": "📚 Found {total} results (🎥 Video {v_count}, 📄 Article {a_count})\nShowing items {start}-{end}\n\nBased on your description, I found these courses/articles:",
            "hint_prefix": "Tips: ",
            "hint_default": "No specific keywords found in subtitles, you can watch from the beginning.",
            "video_link": "🎥 Link: ",
            "more_btn": "👉 Click 'Next 5' for more"
        }
    elif target_lang == 'vi':
        ui = {
            "not_found": "Hiện không tìm thấy nội dung phù hợp. Bạn có thể thử các từ khóa như: căng thẳng, mất ngủ, trầm cảm...",
            "found_msg": "📚 Tìm thấy {total} kết quả (🎥 Video {v_count}, 📄 Bài viết {a_count})\nĐang hiển thị mục {start}～{end}\n\nDựa trên mô tả của bạn, tôi đã tìm thấy các khóa học / bài viết này:",
            "hint_prefix": "💡 Gợi ý:",
            "hint_default": "Không tìm thấy từ khóa cụ thể trong phụ đề, bạn có thể xem từ đầu.",
            "video_link": "🎥 Link video:",
            "more_btn": "👉 Nhấn \"5 mục tiếp theo\" để xem thêm"
        }
    elif target_lang == 'ms':
        ui = {
            "not_found": "Tiada kursus berkaitan ditemui buat masa ini. Cuba kata kunci seperti: stres, insomnia, kemurungan...",
            "found_msg": "📚 Menjumpai {total} keputusan (🎥 Video {v_count}, 📄 Artikel {a_count})\nMenunjukkan item {start}～{end}\n\nBerdasarkan huraian anda, saya menemui kursus / artikel ini:",
            "hint_prefix": "💡 Tips:",
            "hint_default": "Tiada kata kunci khusus ditemui dalam sari kata, anda boleh tonton dari awal.",
            "video_link": "🎥 Pautan video:",
            "more_btn": "👉 Klik \"5 Seterusnya\" untuk lihat lagi"
        }
    elif target_lang == 'zh-CN':
        ui = {
            "not_found": "目前找不到很符合的课程，可以试着用：婆媳、压力、忧郁、失眠… 等词再试试看。",
            "found_msg": "📚 共找到 {total} 笔内容（🎥 视频 {v_count}、📄 文章 {a_count}）\n目前显示第 {start}～{end} 笔\n\n根据你的描述，我帮你找了这些课程 / 文章：",
            "hint_prefix": "💡 小提醒：",
            "hint_default": "字幕里没有特别命中关键句，可以从头开始看。",
            "video_link": "🎥 视频链接：",
            "more_btn": "👉 点击 “给我后五个” 查看更多"
        }
    else:
        # 預設中文
        ui = {
            "not_found": "目前找不到很符合的課程，可以試著用：婆媳、壓力、憂鬱、失眠… 等詞再試試看。",
            "found_msg": "📚 共找到 {total} 筆內容（🎥 影片 {v_count}、📄 文章 {a_count}）\n目前顯示第 {start}～{end} 筆\n\n根據你的敘述，我幫你找了這些課程 / 文章：",
            "hint_prefix": "💡 小提醒：",
            "hint_default": "字幕裡沒有特別命中關鍵句，可以從頭開始看。",
            "video_link": "🎥 影片連結：",
            "more_btn": "👉 點擊 「給我後五個」 可以看更多"
        }

    # 其他語言動態翻譯
    if target_lang not in ['ja', 'en', 'zh-TW']:
        for k, v in ui.items():
            if "{total}" not in v:
                ui[k] = translate_text(v, target_lang)

    # 2. 處理無結果
    if not results:
        return {
            "type": "course_recommendation", "query": query, "total": 0, "video_count": 0, "article_count": 0,
            "offset": offset, "limit": limit, "has_more": False, "results": [],
            "message": ui["not_found"]
        }

    # 3. 數據計算與 Header
    results = reorder_episode_pairs(results)
    total = len(results)
    video_count = sum(1 for r in results if not r.get("is_article"))
    article_count = sum(1 for r in results if r.get("is_article"))
    page_results = results[offset: offset + limit]
    
    start_idx = offset + 1
    end_idx = min(offset + limit, total)
    
    header_msg = ui["found_msg"].format(
        total=total, v_count=video_count, a_count=article_count,
        start=start_idx, end=end_idx
    )

    items = []
    
    # 4. 逐筆處理
    for r in page_results:
        raw_title = r.get("title") or "(無標題)"
        raw_section = r.get("section_title") or ""
        
        # 標題翻譯與格式
        if target_lang != "zh-TW":
            pre_trans_title = raw_title
            
            # [補丁] 針對日文的預處理字典
            if target_lang == 'ja':
                replacements = {
                    "銀髮族": "高齢者", "好眠": "快眠", "睡眠障礙": "睡眠障害",
                    "困擾": "悩み", "處方": "処方", "筆記": "ノート",
                    "如何": "いかにして", "職人": "プロ", "臨床心理師": "臨床心理士",
                    "醫師": "医師", "教授": "先生", "影片": "動画", "文章": "記事",
                    "（上）": "（前編）", "（下）": "（後編）", "與": "と", "的": "の",
                    # 擴充
                    "生理期": "生理", "樂齡": "シニア", "也能": "も", "好好": "ちゃんと",
                    "診治": "診断・治療", "疾患": "病気", "力量": "力", "保健": "健康",
                    "習慣": "習慣", "總是": "いつも", "睡不好": "よく眠れない",
                    "擁有": "持つ", "秘訣": "秘訣", "疲累": "疲れ", "青少年": "青少年",
                    "影響": "影響", "知多少": "知っていますか",
                    "別害怕": "怖がらないで", "老年": "老年", "特色": "特徴",
                    "適度": "適度な", "減輕": "軽減", "關節炎": "関節炎", "情緒": "気分"
                }
                for zh_term, ja_term in replacements.items():
                    pre_trans_title = pre_trans_title.replace(zh_term, ja_term)
                
                pre_trans_title = pre_trans_title.replace("【", "[").replace("】", "] ")

            trans_title = translate_text(pre_trans_title, target_lang)
            
            if trans_title and len(trans_title) > 2 and trans_title != raw_title:
                display_title = f"{raw_title}\n{trans_title}"
            else:
                display_title = raw_title
            
            if raw_section:
                trans_section = translate_text(raw_section, target_lang)
                if trans_section and trans_section != raw_section:
                    display_section = f"{raw_section} / {trans_section}"
                else:
                    display_section = raw_section
            else:
                display_section = ""
        else:
            display_title = raw_title
            display_section = raw_section

        score = r.get("_score", 0.0)
        is_article = bool(r.get("is_article"))
        youtube_url = r.get("youtube_url")

        entry = {
            "section_title": display_section, 
            "title": display_title, 
            "score": score,
            "is_article": is_article, 
            "type": "article" if is_article else "video",
        }

        if is_article:
            content_text = (r.get("content_text") or "").replace("\n", " ")
            snippet_raw = content_text[:100] + "..."
            entry["article_url"] = r.get("article_url") or r.get("url")
            
            if target_lang != "zh-TW":
                trans_snippet = translate_text(snippet_raw, target_lang)
                entry["snippet"] = trans_snippet
            else:
                entry["snippet"] = snippet_raw     
        else:
            seg = r.get("_best_segment")
            if seg:
                start_str = format_time(seg.get("start_sec", 0.0))
                seg_text = seg.get('text', '')[:30]
                
                if target_lang != "zh-TW":
                    trans_seg = translate_text(seg_text, target_lang)
                    if target_lang == "ja":
                        hint_body = f"{start_str} にて言及: 「{trans_seg}...」"
                    else:
                        hint_body = f"Mentioned at {start_str}: \"{trans_seg}...\""
                else:
                    hint_body = f"該單元在 {start_str} 有提到：「{seg_text}...」"
            else:
                hint_body = ui["hint_default"]
            
            entry["hint"] = f"{ui['hint_prefix']} {hint_body}"
            entry["youtube_url"] = youtube_url
            entry["link_label"] = ui["video_link"] 

        items.append(entry)
    
    # Debug tag
    debug_lang = f" (Debug: UI={target_lang})" if target_lang != 'zh-TW' else ""
    
    return {
        "type": "course_recommendation", 
        "query": query, 
        "total": total,
        "video_count": video_count, 
        "article_count": article_count,
        "offset": offset, 
        "limit": limit, 
        "has_more": offset + limit < total,
        "results": items,
        "header_text": header_msg, 
        "message": (ui["more_btn"] if (offset + limit < total) else "") + debug_lang
    }

def execute_hybrid_search(search_query: str, model_key: str = "v3") -> List[Dict[str, Any]]:
    # 防呆：如果傳進來的 key 不在快取裡 (例如前端亂傳)，就預設回 v3
    if model_key not in VECTOR_CACHE:
        print(f"[hybrid] ⚠️ 請求的模型 {model_key} 不存在，切換回 v3")
        model_key = "v3"
        # 如果連 v3 都沒有，就隨便抓一個，避免報錯
        if "v3" not in VECTOR_CACHE and VECTOR_CACHE:
             model_key = list(VECTOR_CACHE.keys())[0]

    print(f"[hybrid] 開始搜尋: {search_query} | 使用模型: {model_key}")
    
    # 1. 關鍵字搜尋 (這部分不受模型版本影響)
    kw_results = search_units(UNITS_CACHE, search_query, top_k=9999)
    
    # 2. 語意搜尋 (★關鍵修改：傳入 model_key)
    vec_results = search_units_semantic(search_query, model_key, top_k=50)
    
    # 3. 混合搜尋加權邏輯 (RRF 或 加權相加)
    combined_map = {}
    
    # 先放入關鍵字結果
    for r in kw_results:
        key = get_base_key(r.get("section_title"), r.get("title"))
        combined_map[key] = r

    # 再疊加向量結果
    for r in vec_results:
        key = get_base_key(r.get("section_title"), r.get("title"))
        
        # 權重設定
        VECTOR_WEIGHT_BOOST = 20.0 
        VECTOR_WEIGHT_BASE = 10.0
        
        if key in combined_map:
            # 如果兩邊都找到，大幅加分
            combined_map[key]["_score"] += (r["_score"] * VECTOR_WEIGHT_BOOST)
        else:
            # 如果只有向量找到，給予基礎分
            if r["_score"] > 0.25: 
                r["_score"] = r["_score"] * VECTOR_WEIGHT_BASE
                combined_map[key] = r
    
    final_results = list(combined_map.values())
    final_results.sort(key=lambda x: x["_score"], reverse=True)
    return final_results

app = FastAPI(title="心快活課程推薦 API")

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", include_in_schema=False)
def serve_index():
    return FileResponse("static/index.html")

UNITS_CACHE = load_all_units()
init_vector_model()

HISTORY: Dict[str, List[Dict[str, Any]] ] = {}

class ChatRequest(BaseModel):
    query: str
    session_id: Optional[str] = None
    model: Optional[str] = "v3"  # 新增這個欄位，預設 v3

class NearbyRequest(BaseModel):
    address: str

class RecommendRequest(BaseModel):
    query: str

@app.get("/ping")
def ping(): return {"status": "ok"}

@app.post("/chat")
def chat(req: ChatRequest):
    start_time = time.time()

    # 1. 基礎參數初始化
    q_origin = req.query.strip()
    session_id = req.session_id or "anonymous"
    target_model = req.model or "v3"  # 取得前端傳來的模型選擇
    
    history_list = HISTORY.get(session_id, [])
    is_pagination = detect_pagination_intent(q_origin)
    
    # 2. 語言偵測與歷史偏好
    # A. 偵測當前輸入
    current_detected = detect_language(q_origin)
    
    # B. 檢查歷史偏好
    historical_lang = "zh-TW"
    if history_list:
        for h in reversed(history_list):
            lang = h.get("detected_lang", "zh-TW")
            if lang != "zh-TW":
                historical_lang = lang
                break
    
    # C. 決策邏輯
    final_lang = "zh-TW"
    if current_detected != "zh-TW":
        final_lang = current_detected
    elif is_pagination and historical_lang != "zh-TW":
        final_lang = historical_lang
    else:
        final_lang = "zh-TW"

    print(f">>> [/chat] Origin: {q_origin} | Detected: {current_detected} | History: {historical_lang} -> Final: {final_lang}")

    # 3. 翻譯與前處理
    if final_lang != "zh-TW":
        q_search = translate_text(q_origin, "zh-TW")
    else:
        q_search = q_origin

    def detect_media_preference(text: str) -> Optional[str]:
        if any(w in text for w in ["想看文章", "給我文章", "只有文章", "文章推薦", "找文章", "只想看文章"]): return "article"
        if any(w in text for w in ["想看影片", "給我影片", "播放影片", "影音", "看影片", "youtube", "只想看影片"]): return "video"
        return None

    media_pref_check = detect_media_preference(q_search)
    q_cleaned = q_search

    if media_pref_check == "article":
        for w in ["想看文章", "給我文章", "只有文章", "文章推薦", "找文章", "只想看文章", "文章"]: q_cleaned = q_cleaned.replace(w, "")
    elif media_pref_check == "video":
        for w in ["想看影片", "給我影片", "播放影片", "影音", "看影片", "youtube", "只想看影片", "影片"]: q_cleaned = q_cleaned.replace(w, "")
    q_cleaned = q_cleaned.strip()
    
    user_core, _, _ = normalize_query(q_cleaned)
    if not user_core and len(q_cleaned) >= 2: user_core = [q_cleaned]

    resp = {}

    # 4. 意圖路由 (Routing)
    
    # Case A: 地址查詢
    if ("附近" in q_search) and ("心據點" in q_search or "看診" in q_search or "門診" in q_search):
        addr = extract_address_from_query(q_search)
        if not addr: 
            msg = "我有點抓不到地址，請嘗試輸入完整地址"
            if final_lang != "zh-TW": msg = translate_text(msg, final_lang)
            resp = {"type": "xin_points", "address": None, "points": [], "message": msg}
        else:
            geo = geocode_address(addr)
            if not geo: 
                msg = f"查不到「{addr}」這個地址"
                if final_lang != "zh-TW": msg = translate_text(msg, final_lang)
                resp = {"type": "xin_points", "address": addr, "points": [], "message": msg}
            else:
                lat, lon = geo
                results = find_nearby_points(lat, lon, max_km=5, top_k=TOP_K)
                resp = build_nearby_points_response(addr, results)

    elif ADDR_HEAD_RE.match(q_search):
        geo = geocode_address(q_search)
        if not geo: 
            msg = f"查不到「{q_search}」這個地址"
            if final_lang != "zh-TW": msg = translate_text(msg, final_lang)
            resp = {"type": "xin_points", "address": q_search, "points": [], "message": msg}
        else:
            lat, lon = geo
            results = find_nearby_points(lat, lon, max_km=5, top_k=TOP_K)
            resp = build_nearby_points_response(q_search, results)

    # Case B: 分頁指令 (下一頁)
    elif detect_pagination_intent(q_search):
        if not history_list:
            msg = "目前沒有上一筆推薦結果，可以先問一個問題 😊"
            if final_lang != "zh-TW": msg = translate_text(msg, final_lang)
            resp = {"type": "text", "message": msg}
        else:
            last_recommendation = next((h for h in reversed(history_list) if isinstance(h.get("response"), dict) and h["response"].get("type") == "course_recommendation"), None)
            
            if not last_recommendation:
                 msg = "目前沒有上一筆推薦結果，可以先問一個問題 😊"
                 if final_lang != "zh-TW": msg = translate_text(msg, final_lang)
                 resp = {"type": "text", "message": msg}
            else:
                prev_resp = last_recommendation["response"]
                prev_query = prev_resp.get("query_raw") or prev_resp.get("query")
                prev_filter = prev_resp.get("filter_type", None)
                new_offset = prev_resp["offset"] + prev_resp["limit"]
                
                # 這裡也要加上 model_key
                full_results = execute_hybrid_search(prev_query, model_key=target_model)
                
                if prev_filter == "article": full_results = [r for r in full_results if r.get("is_article")]
                elif prev_filter == "video": full_results = [r for r in full_results if not r.get("is_article")]
                
                resp = build_recommendations_response(
                    prev_query, full_results, offset=new_offset, limit=TOP_K, 
                    target_lang=final_lang
                )
                resp["filter_type"] = prev_filter
                resp["query_raw"] = prev_query

    # Case C: 只有媒體偏好修正 (例如用戶只說 "只想看影片")
    elif media_pref_check and not q_cleaned:
        if not history_list:
            msg = "請先輸入一個主題，例如「焦慮」或「失眠」。"
            if final_lang != "zh-TW": msg = translate_text(msg, final_lang)
            resp = {"type": "course_recommendation", "query": q_search, "total": 0, "video_count": 0, "article_count": 0, "offset": 0, "limit": TOP_K, "has_more": False, "results": [], "message": msg}
        else:
            last = next((h for h in reversed(history_list) if isinstance(h.get("response"), dict) and h["response"].get("type") == "course_recommendation"), None)
            if not last:
                 msg = "請先輸入一個主題，例如「焦慮」或「失眠」。"
                 if final_lang != "zh-TW": msg = translate_text(msg, final_lang)
                 resp = {"type": "course_recommendation", "query": q_search, "total": 0, "video_count": 0, "article_count": 0, "offset": 0, "limit": TOP_K, "has_more": False, "results": [], "message": msg}
            else:
                prev_resp = last["response"]
                original_topic = prev_resp.get("query_raw") or prev_resp.get("query")
                
                # 這裡也要加上 model_key
                full_results = execute_hybrid_search(original_topic, model_key=target_model)
                
                if media_pref_check == "article": full_results = [r for r in full_results if r.get("is_article")]
                elif media_pref_check == "video": full_results = [r for r in full_results if not r.get("is_article")]
                
                resp = build_recommendations_response(
                    original_topic, full_results, offset=0, limit=TOP_K, 
                    target_lang=final_lang
                )
                resp["filter_type"] = media_pref_check
                resp["query_raw"] = original_topic
                
                if not resp["results"]: 
                    msg = f"關於「{original_topic}」目前沒有相關的內容。" 
                    if final_lang != "zh-TW": msg = translate_text(msg, final_lang)
                    resp["message"] = msg

    # Case D: 一般搜尋 (這是你原本報錯的地方，現在修好了)
    else:
        # 1. 確保 search_q 有值
        search_q = q_cleaned if q_cleaned else q_search
        
        # 2. 執行搜尋 (傳入 model_key)
        full_results = execute_hybrid_search(search_q, model_key=target_model)
        
        final_filter = None
        if media_pref_check == "article":
            full_results = [r for r in full_results if r.get("is_article")]
            final_filter = "article"
        elif media_pref_check == "video":
            full_results = [r for r in full_results if not r.get("is_article")]
            final_filter = "video"
        
        resp = build_recommendations_response(
            q_origin, 
            full_results, 
            offset=0, 
            limit=TOP_K, 
            target_lang=final_lang
        )
        resp["filter_type"] = final_filter
        
        resp["query_raw"] = search_q 
        
        resp["detected_lang"] = final_lang
        resp["query_search_zh"] = search_q

        if media_pref_check and not resp["results"]: 
            msg = f"關於「{search_q}」目前沒有相關的內容。"
            if final_lang != "zh-TW": msg = translate_text(msg, final_lang)
            resp["message"] = msg

    # 5. 後處理 (儲存歷史、計算時間)
    
    # 回傳使用的模型資訊 (方便前端顯示)
    resp["used_model"] = target_model

    end_time = time.time()
    execution_time = end_time - start_time
    resp["process_time"] = f"{execution_time:.4f}s"

    print(f"DEBUG: 計算耗時: {resp['process_time']} | Model: {target_model} | Keys: {list(resp.keys())}")

    # 儲存到歷史紀錄
    history_list = HISTORY.setdefault(session_id, [])
    history_list.append({
        "query": q_origin, 
        "response": resp, 
        "detected_lang": final_lang
    })
    if len(history_list) > 50: history_list.pop(0)

    return resp

@app.get("/history")
def get_history(session_id: str):
    return { "items": HISTORY.get(session_id, []) }

@app.post("/nearby")
def nearby(req: NearbyRequest):
    start_time = time.time()
    addr = req.address.strip()
    resp = {}
    if not addr: 
        return {"type": "xin_points", "address": None, "points": [], "message": "請提供完整地址"}
    else:
        geo = geocode_address(addr)
        if not geo: 
            resp = {"type": "xin_points", "address": addr, "points": [], "message": f"查不到「{addr}」這個地址"}
        else:
            results = find_nearby_points(geo[0], geo[1], max_km=5, top_k=TOP_K)
            resp = build_nearby_points_response(addr, results)
    
    end_time = time.time()
    resp["process_time"] = f"{end_time - start_time:.3f}s"

    return resp

@app.post("/recommend")
def recommend(req: RecommendRequest):
    start_time = time.time()

    q = req.query.strip()
    sid = "anonymous" 
    pref = None
    if any(w in q for w in ["文章"]): pref = "article"
    elif any(w in q for w in ["影片"]): pref = "video"
    
    full_results = execute_hybrid_search(q)
    
    if pref == "article": full_results = [r for r in full_results if r.get("is_article")]
    elif pref == "video": full_results = [r for r in full_results if not r.get("is_article")]

    resp = build_recommendations_response(q, full_results, offset=0, limit=TOP_K)
    
    end_time = time.time()
    resp["process_time"] = f"{end_time - start_time:.3f}s"

    history_list = HISTORY.setdefault(sid, [])
    history_list.append({"query": q, "response": resp})
    return resp

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("xin_api:app", host="0.0.0.0", port=8000, reload=True)