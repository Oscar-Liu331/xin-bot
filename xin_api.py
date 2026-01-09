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

VECTORS_FILE = Path("vectors.json")
CORPUS_VECTORS = None 

JINA_API_URL = "https://api.jina.ai/v1/embeddings"
JINA_API_KEY = None

KEYWORDS_FILE = Path("keywords.json")
KEYWORDS_DATA = {} 
MENTAL_KEYWORDS = [] 
STOP_WORDS = []

# 翻譯用快取
TRANSLATION_CACHE = {}

# --- 核心工具函式 ---

def detect_language(text: str) -> str:
    """
    語言偵測最終版：
    1. 優先檢查日文假名 -> ja
    2. 優先檢查韓文諺文 -> ko
    3. 檢查漢字 -> zh-TW
    4. 純英數 -> en
    """
    if not text: return "zh-TW"
    
    # 1. [優先] 檢查日文 (平假名 \u3040-\u309f / 片假名 \u30a0-\u30ff)
    if re.search(r'[\u3040-\u309f\u30a0-\u30ff]', text):
        return "ja"

    # 2. [優先] 檢查韓文 (諺文 \uac00-\ud7af)
    if re.search(r'[\uac00-\ud7af]', text):
        return "ko"

    # 3. 檢查中文 (漢字 \u4e00-\u9fa5)
    if re.search(r'[\u4e00-\u9fa5]', text):
        return "zh-TW"

    # 4. 檢查純英文
    clean_text = re.sub(r'[0-9\s,.?!:;\'"()\[\]]', '', text)
    if clean_text and all(ord(c) < 128 for c in clean_text):
        return "en"

    # 5. 其他情況交給模型
    try:
        lang = detect(text)
        if lang.startswith("zh"): return "zh-TW"
        return lang
    except LangDetectException:
        return "zh-TW"

def translate_text(text: str, target: str) -> str:
    """
    翻譯函式 (整合智慧重試機制)：
    1. 若是日文目標，先嘗試移除符號再翻 (提高準確度)。
    2. 整合快取。
    """
    if not text: return ""
    # 如果目標是中文，且原文就是中文 (簡單判斷)，直接回傳
    if target == "zh-TW" and detect_language(text) == "zh-TW":
        return text
    
    # 建立快取鍵值
    cache_key = f"{text}_{target}"
    if cache_key in TRANSLATION_CACHE:
        return TRANSLATION_CACHE[cache_key]
    
    try:
        translator = GoogleTranslator(source='auto', target=target)
        
        # [優化] 針對日文翻譯，為了避免 API 把 "【影片】" 當作不需翻譯的符號，
        # 我們優先送出乾淨的文字，這樣 "女性與睡眠障礙" 的 "與" 比較容易被翻成 "と"
        text_to_translate = text
        if target == 'ja':
             # 移除常見干擾符號
            clean_text = re.sub(r"[【】《》「」()（）]", " ", text)
            clean_text = re.sub(r"\s+", " ", clean_text).strip()
            if clean_text:
                text_to_translate = clean_text

        # 執行翻譯
        result = translator.translate(text_to_translate)
        
        # 防呆：如果翻譯失敗 (結果跟原文完全一樣，且長度足夠)
        # 且我們還沒試過移除符號 (即非日文模式)，則嘗試移除符號重試
        if result == text_to_translate and len(text) > 5 and target != 'ja':
             clean_text = re.sub(r"[【】《》「」()（）]", " ", text)
             clean_text = re.sub(r"\s+", " ", clean_text).strip()
             if clean_text != text:
                retry_result = translator.translate(clean_text)
                if retry_result != clean_text:
                    result = retry_result
        
        # 寫入快取
        TRANSLATION_CACHE[cache_key] = result
        return result

    except Exception as e:
        print(f"!!! [Translate Error] Text: {text[:10]}... | Error: {e}")
        return text # 失敗時回傳原文

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

def init_vector_model():
    global CORPUS_VECTORS, JINA_API_KEY
    
    JINA_API_KEY = os.environ.get("JINA_API_KEY")
    if not JINA_API_KEY:
        print("[init] ⚠️ 警告：找不到 JINA_API_KEY，語意搜尋將無法運作！")
    else:
        print("[init] ✅ Jina API Key 已設定")

    if VECTORS_FILE.exists():
        print(f"[init] 正在讀取向量快取: {VECTORS_FILE} ...")
        try:
            with open(VECTORS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                CORPUS_VECTORS = np.array(data, dtype="float32") 
            print(f"[init] ✅ 成功載入 {len(CORPUS_VECTORS)} 筆向量資料")
        except Exception as e:
            print(f"[init] ❌ 讀取向量檔失敗: {e}")
    else:
        print("[init] ⚠️ 找不到 vectors.json")

def get_jina_embedding(text):
    """透過 requests 呼叫 Jina API"""
    if not JINA_API_KEY:
        raise Exception("JINA_API_KEY not set")
        
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {JINA_API_KEY}"
    }
    data = {
        "model": "jina-embeddings-v3",
        "input": [text] 
    }
    
    try:
        resp = requests.post(JINA_API_URL, headers=headers, json=data, timeout=10)
        resp.raise_for_status()
        result = resp.json()
        return result["data"][0]["embedding"]
    except Exception as e:
        print(f"[Jina API Error] {e}")
        return None

def search_units_semantic(query: str, top_k: int = 5):
    global CORPUS_VECTORS
    
    if not JINA_API_KEY or CORPUS_VECTORS is None:
        return []

    try:
        query_vec_list = get_jina_embedding(query)
        if not query_vec_list:
            return []
            
        query_vec = np.array(query_vec_list, dtype="float32")

        scores = np.dot(CORPUS_VECTORS, query_vec)

        top_indices = np.argsort(scores)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            score = float(scores[idx])
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
    return any(w in q for w in ["給我後五個","給我下五個","後五個","下五個","下一頁","更多推薦"])

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
            if d <= max_km: results.append((p, d))
    results.sort(key=lambda x: x[1])
    return results[:top_k]

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

def build_recommendations_response(query: str, results: List[Dict[str, Any]], 
                                   offset: int = 0, limit: int = TOP_K, 
                                   target_lang: str = "zh-TW"):
    
    # --- 1. 定義介面文字 (使用 Hardcode 保證翻譯品質) ---
    if target_lang == 'en':
        ui = {
            "not_found": "Currently no relevant courses found. You can try keywords like: stress, insomnia, depression...",
            "found_pattern": "Found {total} results (🎥 Video {v_count}, 📄 Article {a_count})",
            "showing": "Showing items",
            "intro": "Based on your description, I found these courses/articles:",
            "hint_prefix": "Tips: ",
            "hint_default": "No specific keywords found in subtitles, you can watch from the beginning.",
            "video_link": "🎥 Link: ",
            "more_btn": "👉 Click 'Next 5' for more"
        }
    elif target_lang == 'ja':
        # [新增] 日文介面 hardcode
        ui = {
            "not_found": "条件に合うコンテンツが見つかりませんでした。「ストレス」、「不眠」、「不安」などのキーワードで試してみてください。",
            "found_pattern": "合計 {total} 件見つかりました（🎥 動画 {v_count}、📄 記事 {a_count}）",
            "showing": "表示中: ",
            "intro": "ご相談内容に基づき、以下のコース/記事を検索しました：",
            "hint_prefix": "ヒント: ",
            "hint_default": "字幕に特定のキーワードは見つかりませんでした。最初からご覧ください。",
            "video_link": "🎥 リンク: ",
            "more_btn": "👉 「次の5件」をクリックしてもっと見る"
        }
    else:
        # 預設中文
        ui = {
            "not_found": "目前找不到很符合的課程，可以試著用：婆媳、壓力、憂鬱、失眠… 等詞再試試看。",
            "found_pattern": "共找到 {total} 筆內容（🎥 影片 {v_count}、📄 文章 {a_count}）",
            "showing": "目前顯示第",
            "intro": "根據你的敘述，我幫你找了這些課程 / 文章：",
            "hint_prefix": "💡 小提醒：",
            "hint_default": "字幕裡沒有特別命中關鍵句，可以從頭開始看。",
            "video_link": "🎥 影片連結：",
            "more_btn": "👉 點擊 「給我後五個」 可以看更多"
        }
        
        # 如果是其他語言 (韓文/歐語等)，才使用即時翻譯
        if target_lang != "zh-TW":
            for k, v in ui.items():
                if "{total}" not in v: # 簡單字串直接翻
                    ui[k] = translate_text(v, target_lang)

    # --- 2. 處理無結果 ---
    if not results:
        return {
            "type": "course_recommendation", "query": query, "total": 0, "video_count": 0, "article_count": 0,
            "offset": offset, "limit": limit, "has_more": False, "results": [],
            "message": ui["not_found"]
        }

    # --- 3. 數據計算與 Header ---
    results = reorder_episode_pairs(results)
    total = len(results)
    video_count = sum(1 for r in results if not r.get("is_article"))
    article_count = sum(1 for r in results if r.get("is_article"))
    page_results = results[offset: offset + limit]
    
    # 根據語言組裝 Header
    if target_lang == 'en':
        header_msg = (
            f"📚 {ui['found_pattern'].format(total=total, v_count=video_count, a_count=article_count)}\n"
            f"{ui['showing']} {offset + 1}-{min(offset + limit, total)}\n\n"
            f"{ui['intro']}"
        )
    elif target_lang == 'ja':
        header_msg = (
            f"📚 {ui['found_pattern'].format(total=total, v_count=video_count, a_count=article_count)}\n"
            f"{ui['showing']} {offset + 1}～{min(offset + limit, total)} 件\n\n"
            f"{ui['intro']}"
        )
    else:
        # 中文
        header_msg = (
            f"📚 {ui['found_pattern'].format(total=total, v_count=video_count, a_count=article_count)}\n"
            f"{ui['showing']} {offset + 1}～{min(offset + limit, total)} 筆\n\n"
            f"{ui['intro']}"
        )

    items = []
    
    # --- 4. 逐筆處理 ---
    for r in page_results:
        raw_title = r.get("title") or "(Untitled)"
        raw_section = r.get("section_title") or ""
        
        # 標題處理
        if target_lang != "zh-TW":
            trans_title = translate_text(raw_title, target_lang)
            
            # 檢查翻譯是否有效 (不為空 且 不等於原文)
            if trans_title and trans_title != raw_title:
                display_title = f"{raw_title}\n   [{trans_title}]"
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
            # 處理影片提示語
            seg = r.get("_best_segment")
            if seg:
                start_str = format_time(seg.get("start_sec", 0.0))
                seg_text = seg.get('text', '')[:30]
                
                # 若是英日文，使用翻譯並 Hardcode 結構
                if target_lang == "en":
                    trans_seg = translate_text(seg_text, target_lang)
                    hint_body = f"Mentioned at {start_str}: \"{trans_seg}...\""
                elif target_lang == "ja":
                    trans_seg = translate_text(seg_text, target_lang)
                    hint_body = f"{start_str} にて言及: 「{trans_seg}...」"
                elif target_lang != "zh-TW":
                    trans_seg = translate_text(seg_text, target_lang)
                    hint_body = f"{start_str}: \"{trans_seg}...\""
                else:
                    hint_body = f"該單元在 {start_str} 有提到：「{seg_text}...」"
            else:
                # 這裡會用到上方定義好的 ui["hint_default"] (已經是日文了)
                hint_body = ui["hint_default"]
            
            entry["hint"] = f"{ui['hint_prefix']} {hint_body}"
            entry["youtube_url"] = youtube_url
            entry["link_label"] = ui["video_link"] 

        items.append(entry)
    
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
        "message": ui["more_btn"] if (offset + limit < total) else "" 
    }

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

def execute_hybrid_search(search_query: str) -> List[Dict[str, Any]]:
    print(f"[hybrid] 開始搜尋: {search_query}")
    kw_results = search_units(UNITS_CACHE, search_query, top_k=9999)
    vec_results = search_units_semantic(search_query, top_k=50)
    
    combined_map = {}
    for r in kw_results:
        key = get_base_key(r.get("section_title"), r.get("title"))
        combined_map[key] = r

    for r in vec_results:
        key = get_base_key(r.get("section_title"), r.get("title"))
        VECTOR_WEIGHT_BOOST = 20.0 
        VECTOR_WEIGHT_BASE = 10.0
        if key in combined_map:
            combined_map[key]["_score"] += (r["_score"] * VECTOR_WEIGHT_BOOST)
        else:
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

class NearbyRequest(BaseModel):
    address: str

class RecommendRequest(BaseModel):
    query: str

@app.get("/ping")
def ping(): return {"status": "ok"}

@app.post("/chat")
def chat(req: ChatRequest):
    q_origin = req.query.strip()
    session_id = req.session_id or "anonymous"
    
    user_lang = detect_language(q_origin)
    
    print(f">>> [/chat] session_id: {session_id} | User Lang: {user_lang} | Origin: {q_origin}")

    if user_lang != "zh-TW":
        q_search = translate_text(q_origin, "zh-TW")
        print(f"    Translated for search: {q_search}")
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

    if ("附近" in q_search) and ("心據點" in q_search or "看診" in q_search or "門診" in q_search):
        addr = extract_address_from_query(q_search)
        if not addr: 
            msg = "我有點抓不到地址，請嘗試輸入完整地址"
            if user_lang != "zh-TW": msg = translate_text(msg, user_lang)
            resp = {"type": "xin_points", "address": None, "points": [], "message": msg}
        else:
            geo = geocode_address(addr)
            if not geo: 
                msg = f"查不到「{addr}」這個地址"
                if user_lang != "zh-TW": msg = translate_text(msg, user_lang)
                resp = {"type": "xin_points", "address": addr, "points": [], "message": msg}
            else:
                lat, lon = geo
                results = find_nearby_points(lat, lon, max_km=5, top_k=TOP_K)
                resp = build_nearby_points_response(addr, results)

    elif ADDR_HEAD_RE.match(q_search):
        geo = geocode_address(q_search)
        if not geo: 
            msg = f"查不到「{q_search}」這個地址"
            if user_lang != "zh-TW": msg = translate_text(msg, user_lang)
            resp = {"type": "xin_points", "address": q_search, "points": [], "message": msg}
        else:
            lat, lon = geo
            results = find_nearby_points(lat, lon, max_km=5, top_k=TOP_K)
            resp = build_nearby_points_response(q_search, results)

    elif detect_pagination_intent(q_search):
        history = HISTORY.get(session_id, [])
        last = next((h for h in reversed(history) if h["response"].get("type") == "course_recommendation"), None)
        
        if not last: 
            msg = "目前沒有上一筆推薦結果，可以先問一個問題 😊"
            if user_lang != "zh-TW": msg = translate_text(msg, user_lang)
            resp = {"type": "text", "message": msg}
        else:
            prev_resp = last["response"]
            prev_query = prev_resp.get("query_raw") or prev_resp.get("query")
            prev_filter = prev_resp.get("filter_type", None)
            new_offset = prev_resp["offset"] + prev_resp["limit"]
            
            full_results = execute_hybrid_search(prev_query)
            if prev_filter == "article": full_results = [r for r in full_results if r.get("is_article")]
            elif prev_filter == "video": full_results = [r for r in full_results if not r.get("is_article")]
            
            resp = build_recommendations_response(
                prev_query, full_results, offset=new_offset, limit=TOP_K, 
                target_lang=user_lang
            )
            resp["filter_type"] = prev_filter
            resp["query_raw"] = prev_query

    elif media_pref_check and not q_cleaned:
        history = HISTORY.get(session_id, [])
        last = next((h for h in reversed(history) if isinstance(h.get("response"), dict) and h["response"].get("type") == "course_recommendation"), None)
        
        if not last:
            msg = "請先輸入一個主題，例如「焦慮」或「失眠」。"
            if user_lang != "zh-TW": msg = translate_text(msg, user_lang)
            resp = {"type": "course_recommendation", "query": q_search, "total": 0, "video_count": 0, "article_count": 0, "offset": 0, "limit": TOP_K, "has_more": False, "results": [], "message": msg}
        else:
            prev_resp = last["response"]
            original_topic = prev_resp.get("query_raw") or prev_resp.get("query")
            
            full_results = execute_hybrid_search(original_topic)
            if media_pref_check == "article": full_results = [r for r in full_results if r.get("is_article")]
            elif media_pref_check == "video": full_results = [r for r in full_results if not r.get("is_article")]
            
            resp = build_recommendations_response(
                original_topic, full_results, offset=0, limit=TOP_K, 
                target_lang=user_lang
            )
            resp["filter_type"] = media_pref_check
            resp["query_raw"] = original_topic
            
            if not resp["results"]: 
                msg = f"關於「{original_topic}」目前沒有相關的內容。" 
                if user_lang != "zh-TW": msg = translate_text(msg, user_lang)
                resp["message"] = msg

    else:
        search_q = q_cleaned if q_cleaned else q_search
        full_results = execute_hybrid_search(search_q)
        
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
            target_lang=user_lang
        )
        resp["filter_type"] = final_filter
        
        resp["query_raw"] = search_q 
        
        resp["detected_lang"] = user_lang
        resp["query_search_zh"] = search_q

        if media_pref_check and not resp["results"]: 
            msg = f"關於「{search_q}」目前沒有相關的內容。"
            if user_lang != "zh-TW": msg = translate_text(msg, user_lang)
            resp["message"] = msg

    history_list = HISTORY.setdefault(session_id, [])
    history_list.append({"query": q_origin, "response": resp})
    if len(history_list) > 50: history_list.pop(0)
    
    return resp

@app.get("/history")
def get_history(session_id: str):
    return { "items": HISTORY.get(session_id, []) }

@app.post("/nearby")
def nearby(req: NearbyRequest):
    addr = req.address.strip()
    if not addr: return {"type": "xin_points", "address": None, "points": [], "message": "請提供完整地址"}
    geo = geocode_address(addr)
    if not geo: return {"type": "xin_points", "address": addr, "points": [], "message": f"查不到「{addr}」這個地址"}
    results = find_nearby_points(geo[0], geo[1], max_km=5, top_k=TOP_K)
    return build_nearby_points_response(addr, results)

@app.post("/recommend")
def recommend(req: RecommendRequest):
    q = req.query.strip()
    sid = "anonymous" 
    pref = None
    if any(w in q for w in ["文章"]): pref = "article"
    elif any(w in q for w in ["影片"]): pref = "video"
    
    full_results = execute_hybrid_search(q)
    
    if pref == "article": full_results = [r for r in full_results if r.get("is_article")]
    elif pref == "video": full_results = [r for r in full_results if not r.get("is_article")]

    resp = build_recommendations_response(q, full_results, offset=0, limit=TOP_K)
    history_list = HISTORY.setdefault(sid, [])
    history_list.append({"query": q, "response": resp})
    return resp

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("xin_api:app", host="0.0.0.0", port=8000, reload=True)