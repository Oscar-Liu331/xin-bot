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

from typing import Optional


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
    
    "繪本","故事",
    "長輩過世",
]

STOP_WORDS = [
    "我", "你", "他", "她", "它", "我們", "你們", "他們",
    "最近", "一直", "覺得", "有點", "有一點",
    "如果", "好像", "是不是",
    "該怎麼辦", "怎麼辦", "怎麼做", "該怎麼做",
    "可以", "覺得", "自己",
    "的", "了", "呢", "嗎", "吧"
]

def detect_pagination_intent(q: str) -> bool:
    return any(w in q for w in ["給我後五個","給我下五個","後五個","下五個","下一頁","更多推薦"])

def detect_special_intent(q: str) -> Optional[str]:
    """
    偵測是不是屬於「要直接給建議」的幾種常見問題：
      - 憂鬱要不要看醫師
      - 擔心家人失智，看哪一科
      - 小學生手機玩太多
      - 婆婆帶小孩教養衝突
    回傳 intent 名稱，沒命中就回傳 None。
    """
    # 去掉空白符號
    text = re.sub(r"\s+", "", q)
    print(f"[intent-debug] raw='{q}' text='{text}'")

    # ---------- 1) 我覺得有點憂鬱，要不要看醫師？ ----------
    has_depression_word = any(w in text for w in ["憂鬱", "心情低落", "心情不好", "情緒低落"])
    # 只要有「看」+（醫師/醫生/心理/身心科/精神科）就視為問就醫
    has_see_doctor = (
        "看" in text and (
            "醫師" in text or "醫生" in text or "心理師" in text or
            "心理醫師" in text or "身心科" in text or "精神科" in text
        )
    )
    # 額外加一些常見語氣
    has_doubt_phrase = any(w in text for w in ["要不要", "該不該", "需不需要", "要不要去"])

    if has_depression_word and (has_see_doctor or has_doubt_phrase):
        return "depression_go_doctor"

    # ---------- 2) 擔心爸爸 / 長輩失智，看哪一科？ ----------
    if "失智" in text and any(w in text for w in ["爸爸", "媽媽", "父親", "母親", "長輩", "家人", "阿公", "阿嬤"]):
        return "dementia_parent"

    # ---------- 3) 國小小孩手機玩太兇，怎麼辦？ ----------
    has_device = any(w in text for w in ["手機", "平板", "遊戲", "電動", "線上遊戲"])
    has_child = any(w in text for w in ["國小", "小學", "小孩", "孩子", "兒子", "女兒", "國三", "國二", "國一"])
    if has_device and has_child:
        return "child_phone"

    # ---------- 4) 婆婆照顧小孩方式和我很不一樣 ----------
    has_inlaw = any(w in text for w in ["婆婆", "公婆", "婆家"])
    has_childcare = any(w in text for w in ["小孩", "孩子", "顧小孩", "帶小孩", "照顧", "教養"])
    if has_inlaw and has_childcare:
        return "mother_in_law_childcare"

    return None


def build_special_intent_response(intent: str, q: str) -> Dict[str, Any]:
    """
    把四種情境的建議，包成結構化 JSON，給前端渲染。
    """
    if intent == "depression_go_doctor":
        title = "覺得自己有點憂鬱，該不該去看心理醫師？"
        sections = [
            {
                "title": "1️⃣ 什麼情況比較建議找專業醫師或心理師？",
                "items": [
                    "情緒低落、沒動力、容易想哭，持續超過 2 週以上。",
                    "影響到工作、課業、睡眠、食慾或人際關係。",
                    "出現「不如消失算了」、「活著好累」這類負面或自傷的想法。"
                ]
            },
            {
                "title": "2️⃣ 可以看的科別 / 專業",
                "items": [
                    "醫院的「身心科 / 精神科」門診，可以評估是否需要用藥或進一步檢查。",
                    "醫院或社區的「臨床心理師、諮商心理師」做心理諮商。",
                    "如果還不確定，也可以先從「家醫科」或社區心理衛生中心諮詢開始。"
                ]
            },
            {
                "title": "3️⃣ 如果現在還勉強撐得住，可以先嘗試的調整",
                "items": [
                    "固定睡覺、起床時間，盡量少熬夜。",
                    "找一個信任的人聊聊，把壓力說出來。",
                    "先從短時間散步或簡單運動開始，讓身體動起來。"
                ]
            },
            {
                "title": "⚠️ 什麼情況要立刻尋求協助？",
                "items": [
                    "有明顯的自殺念頭、衝動，或已經想好方法。",
                    "此時請儘快到醫院急診，或請家人朋友陪同就醫，並可聯絡自殺防治 / 心理支持專線。"
                ]
            }
        ]

    elif intent == "dementia_parent":
        title = "擔心爸爸 / 家人可能有失智，怎麼確定？看哪一科？"
        sections = [
            {
                "title": "1️⃣ 常見的失智警訊（舉例）",
                "items": [
                    "記憶力明顯變差：同一件事問很多次，忘記剛發生的事情。",
                    "容易迷路：在熟悉的環境反而會走錯、找不到路。",
                    "判斷力變差：例如容易被詐騙、做一些以前不會做的怪決定。",
                    "性格或行為改變：變得明顯暴躁、退縮，或跟以前個性差很多。"
                ]
            },
            {
                "title": "2️⃣ 要怎麼比較確定是不是失智？",
                "items": [
                    "需要由醫師做完整評估，可能包括問診、神經心理量表、血液檢查、影像檢查等。",
                    "家屬可以先整理最近觀察到的改變（從什麼時候開始、發生在什麼情境）。"
                ]
            },
            {
                "title": "3️⃣ 建議先看哪一科？",
                "items": [
                    "大型醫院常見科別：神經內科、家醫科、老年醫學科，部分醫院有「失智共同照護門診」。",
                    "若長輩有明顯情緒或行為改變（如妄想、幻覺、嚴重焦慮），身心科 / 精神科也能協助評估。"
                ]
            },
            {
                "title": "4️⃣ 帶長輩看診的小技巧",
                "items": [
                    "可以用「做健康檢查」的方式邀請，而不是直接說「懷疑你失智」。",
                    "看診時，家屬可把在家觀察到的狀況整理在紙上給醫師看，比較不會漏講。"
                ]
            }
        ]

    elif intent == "child_phone":
        title = "國小孩子手機越玩越兇，我該怎麼辦？"
        sections = [
            {
                "title": "1️⃣ 先了解「怎麼玩」而不只是「玩多久」",
                "items": [
                    "主要是在玩遊戲、看影片，還是跟同學聊天？",
                    "通常在什麼時間點玩：寫功課前？睡前？假日整天？"
                ]
            },
            {
                "title": "2️⃣ 跟孩子一起「談規則」，不是只下命令",
                "items": [
                    "例如：平日每天可以玩 30–60 分鐘，先寫完作業、洗澡再開機。",
                    "避免睡前 1 小時用手機，讓大腦有時間「降速」比較好睡。",
                    "把規則寫在紙上貼出來，減少臨時吵架。"
                ]
            },
            {
                "title": "3️⃣ 提供替代活動，不是只有「不准玩」",
                "items": [
                    "安排可以一起做的事：桌遊、運動、散步、畫畫、做料理。",
                    "假日約定「無手機時段」，全家一起做別的活動。"
                ]
            },
            {
                "title": "4️⃣ 大人也要當榜樣",
                "items": [
                    "如果父母一直滑手機，小孩很難接受「你不可以滑」。",
                    "可以一起約定：吃飯、睡前半小時，全家都不看手機。"
                ]
            },
            {
                "title": "5️⃣ 什麼時候需要專業協助？",
                "items": [
                    "已經影響到學業、睡眠，或為了手機會大吵大鬧、摔東西。",
                    "可以考慮諮詢：學校輔導老師、兒童青少年身心科、兒童臨床 / 諮商心理師。"
                ]
            }
        ]

    elif intent == "mother_in_law_childcare":
        title = "婆婆照顧小孩方式跟我很不一樣，我該怎麼辦？"
        sections = [
            {
                "title": "1️⃣ 先分辨：是『做法不同』還是『安全有疑慮』",
                "items": [
                    "做法不同但安全：例如零食多一點、看電視久一點，可以先當成「風格差異」。",
                    "涉及安全：例如讓小孩單獨在陽台、吃容易噎到的食物，就需要比較堅定地溝通。"
                ]
            },
            {
                "title": "2️⃣ 溝通時，先感謝再表達擔心（用「我」訊息）",
                "items": [
                    "如：「我真的很感謝妳幫忙顧小孩，我會比較放心。」",
                    "再接：「只是我有點擔心，他吃太多糖對牙齒不好，我想我們可不可以一起幫他少一點？」",
                    "避免用「妳這樣不對」「妳把小孩帶壞了」，比較不會立刻變成吵架。"
                ]
            },
            {
                "title": "3️⃣ 儘量讓另一半當橋樑",
                "items": [
                    "自己的爸媽通常比較聽自己小孩的話。",
                    "可以先跟另一半私下溝通好立場與底線，再由他 / 她跟婆婆說。"
                ]
            },
            {
                "title": "4️⃣ 建立幾條「全家共同的原則」",
                "items": [
                    "例如：用藥一定要問爸媽、不能打小孩、大約幾點睡覺。",
                    "原則越清楚，越不會每件小事都吵成一團。"
                ]
            },
            {
                "title": "5️⃣ 如果衝突影響到你自己的情緒",
                "items": [
                    "可以考慮和諮商心理師談談，整理自己的委屈與角色壓力（媳婦 / 媽媽雙重身分）。",
                    "有時候問題不只是教養技巧，也是你和先生、你和婆婆之間的界線。"
                ]
            }
        ]

    else:
        title = "一般建議"
        sections = [
            {
                "title": "",
                "items": ["這個問題目前還沒有專門寫好的建議回答，先用課程與文章推薦模式幫你找資料。"]
            }
        ]

    return {
        "type": "advice",
        "intent": intent,
        "query": q,
        "title": title,
        "sections": sections
    }


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


def build_recommendations_response(
    query: str,
    results: List[Dict[str, Any]],
    offset: int = 0,
    limit: int = TOP_K
):
    # 沒有結果
    if not results:
        return {
            "type": "course_recommendation",
            "query": query,
            "total": 0,
            "video_count": 0,
            "article_count": 0,
            "offset": offset,
            "limit": limit,
            "has_more": False,
            "results": [],
            "message": "目前找不到很符合的課程，可以試著用：婆媳、壓力、憂鬱、失眠… 等詞再試試看。"
        }

    # ✅ 先重排（分頁前）
    results = reorder_episode_pairs(results)

    total = len(results)
    # ✅ 計算總數（用重排後的 results 統計即可）
    video_count = sum(1 for r in results if not r.get("is_article"))
    article_count = sum(1 for r in results if r.get("is_article"))

    # ✅ 正確切分頁
    page_results = results[offset: offset + limit]

    items = []
    for r in page_results:  # ✅ 一定要用 page_results
        title = r.get("title") or "(無標題)"
        section_title = r.get("section_title") or "(未分類小節)"
        score = r.get("_score", 0.0)

        is_article = bool(r.get("is_article"))
        youtube_url = r.get("youtube_url")

        entry: Dict[str, Any] = {
            "section_title": section_title,
            "title": title,
            "score": score,
            "is_article": is_article,  # ✅ 給前端直接用
            "type": "article" if is_article else "video",
        }

        if is_article:
            article_url = r.get("article_url") or r.get("url")
            content_text = (r.get("content_text") or "").replace("\n", " ")
            snippet = content_text[:100] + ("..." if len(content_text) > 100 else "")
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
            entry["youtube_url"] = youtube_url
            entry["hint"] = hint

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

# 把「(上)/(下)/(（上）)/(（下）)/上篇/下篇/上集/下集」視為集數標記（可出現在任何位置）
EP_TAG_RE = re.compile(r"(（上）|（下）|\(上\)|\(下\)|上篇|下篇|上集|下集)")

def get_episode_tag(title: str) -> Optional[str]:
    """回傳 '上' / '下' / None（不限出現在結尾）"""
    if not title:
        return None
    t = title.strip()
    if re.search(r"(（上）|\(上\)|上篇|上集)", t):
        return "上"
    if re.search(r"(（下）|\(下\)|下篇|下集)", t):
        return "下"
    return None

def get_base_key(section_title: str, title: str) -> str:
    """
    用來把「上/下」視為同一組的 key
    - 移除標題中的上/下標記（不限位置）
    - 再用 section_title + 清理後 title 當 key
    """
    s = (section_title or "").strip()
    t = (title or "").strip()
    t2 = EP_TAG_RE.sub("", t)  # ✅ 不限結尾，直接把(上)/(下)移除
    t2 = re.sub(r"\s+", "", t2)  # 可選：去空白，讓 key 更穩
    s2 = re.sub(r"\s+", "", s)
    return f"{s2}||{t2}"


def reorder_episode_pairs(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    目標：
    - 同一系列（同 base_key）的上/下要黏在一起：上 → 下
    - 系列與系列之間，用該系列的代表分數排序（預設取系列內最高 _score）
    - 沒有上/下標記的單篇，視為一個獨立系列，保留原本項目本身
    """

    # 1) 建立群組：key -> {"items": [...], "best_score": float, "first_idx": int}
    groups: Dict[str, Dict[str, Any]] = {}

    for idx, r in enumerate(results):
        key = get_base_key(r.get("section_title"), r.get("title"))
        score = float(r.get("_score", 0.0))

        g = groups.get(key)
        if g is None:
            groups[key] = {
                "items": [],
                "best_score": score,
                "first_idx": idx,  # 若分數一樣，用最早出現順序當 tie-break
            }
            g = groups[key]

        g["items"].append(r)
        if score > g["best_score"]:
            g["best_score"] = score

    # 2) 每組內：上→下→其他（如果有奇怪的沒標記）
    def item_rank(r: Dict[str, Any]) -> int:
        tag = get_episode_tag(r.get("title") or "")
        if tag == "上":
            return 0
        if tag == "下":
            return 1
        return 2

    for g in groups.values():
        # 同一組可能有多個上或多個下：同 rank 內再用分數高的排前
        g["items"].sort(key=lambda r: (item_rank(r), -float(r.get("_score", 0.0))))

    # 3) 組與組之間排序：分數高的組排前；分數同則用 first_idx 保持穩定
    ordered_groups = sorted(
        groups.values(),
        key=lambda g: (-g["best_score"], g["first_idx"])
    )

    # 4) 攤平成一條 list
    out: List[Dict[str, Any]] = []
    for g in ordered_groups:
        out.extend(g["items"])

    return out

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
@app.get("/", include_in_schema=False)
def serve_index():
    return FileResponse("static/index.html")

# 啟動時就先載入所有單元
UNITS_CACHE: List[Dict[str, Any]] = load_all_units()

# 簡單放在記憶體的聊天紀錄（server 重啟會清空）
HISTORY: Dict[str, List[Dict[str, Any]] ] = {}


class ChatRequest(BaseModel):
    query: str
    session_id: Optional[str] = None


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
    session_id = req.session_id or "anonymous"  # 沒傳就先歸到 anonymous（理論上前端都會傳）
    resp: Dict[str, Any]
    print(">>> /chat session_id =", session_id)
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
    elif detect_pagination_intent(q):
        history = HISTORY.get(session_id, [])
        # 找最近一筆「課程推薦」
        last = next(
            (h for h in reversed(history)
            if h["response"].get("type") == "course_recommendation"),
            None
        )

        if not last:
            resp = {
                "type": "text",
                "message": "目前沒有上一筆推薦結果，可以先問一個問題 😊"
            }
        else:
            prev = last["response"]
            new_offset = prev["offset"] + prev["limit"]

            full_results = search_units(UNITS_CACHE, prev["query"], top_k=9999)
            resp = build_recommendations_response(
                prev["query"],
                full_results,
                offset=new_offset,
                limit=TOP_K
            )
    # 3) 特定情境：直接給建議，不走課程推薦
    else:
        special_intent = detect_special_intent(q)
        print(f"[chat-debug] special_intent={special_intent}")
        if special_intent:
            resp = build_special_intent_response(special_intent, q)
        else:
            # 4) 其他情況：當作課程 / 文章推薦查詢
            full_results = search_units(UNITS_CACHE, q, top_k=9999)
            resp = build_recommendations_response(q, full_results, offset=0, limit=TOP_K)

    # --- 記錄歷史：依 session_id 分開 ---
    history_list = HISTORY.setdefault(session_id, [])
    history_list.append({
        "query": q,
        "response": resp,
    })
    if len(history_list) > 50:
        history_list.pop(0)

    return resp


@app.get("/history")
def get_history(session_id: str):
    """
    依 session_id 回傳專屬聊天紀錄。
    /history?session_id=xxxx
    """
    print(">>> /history called session_id =", session_id)
    items = HISTORY.get(session_id, [])
    print(">>> HISTORY keys =", list(HISTORY.keys()))
    return {
        "items": items
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
    full_results = search_units(UNITS_CACHE, q, top_k=9999)
    resp = build_recommendations_response(q, full_results, offset=0, limit=TOP_K)
    return resp

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("xin_api:app", host="0.0.0.0", port=8000, reload=True)