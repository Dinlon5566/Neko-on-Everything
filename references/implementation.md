# Memory Systems: Technical Reference
# 記憶系統技術參考文件
# 為 neko-on-everything 技能提供底層記憶系統的實作細節

本參考文件詳細說明記憶增強層的實作細節，包括：
- 向量儲存的實作與貓娘化特化
- 屬性圖的關係建模與查詢介面
- 時序知識圖譜的時間旅行查詢
- 記憶整合器的整理策略
- 貓娘輸出過濾器的實作原理
- 與 .catbox 文字檔案的雙軌同步機制

---

## 向量儲存實作

### NekoVectorStore 架構

NekoVectorStore 在基礎 VectorStore 之上，對 metadata 語意進行了貓娘化特化：

```python
class NekoVectorStore(VectorStore):
    """寶寶專屬向量儲存，metadata 語意特化"""

    # 貓娘特有的 metadata 欄位
    MEMORY_FIELDS = {
        "text": "原始事實文字",
        "entity": "關聯的主實體（通常是「主人」）",
        "fact_type": "事實類型：preference / mistake / knowledge",
        "confidence": "可信度 0.0 ~ 1.0",
        "valid_from": "有效期起始 ISO 時間戳",
        "valid_until": "有效期結束，None=永久有效",
        "session_id": "記錄來源的對話 ID",
        "tags": "語意標籤清單",
        "catbox_path": "對應的文字檔案路徑",
    }
```

### add_memory() 語意

```python
def add_memory(self, text: str, entity: str = "主人",
               fact_type: str = "knowledge",
               confidence: float = 1.0,
               tags: List[str] = None,
               catbox_path: str = None) -> int:
    """新增記憶條目（寶寶版）

    與基礎 VectorStore.add() 的區別：
    1. 自動攜帶貓娘特有的 metadata 欄位
    2. 自動建立 entity_index 和 time_index
    3. 回傳值可以直接關聯到 catbox_path（文字檔案）
    """
    return self.add(text, {
        "text": text,
        "entity": entity,
        "fact_type": fact_type,
        "confidence": confidence,
        "valid_from": datetime.now().isoformat(),
        "valid_until": None,          # 預設永久有效
        "session_id": self.session_id,
        "tags": tags or [],
        "catbox_path": catbox_path,
    })
```

### search_memories() — 多維過濾搜尋

```python
def search_memories(self, query: str,
                   fact_types: List[str] = None,
                   entity: str = "主人",
                   time_range: tuple = None,
                   limit: int = 5) -> List[Dict]:
    """語意搜尋記憶，支援多維過濾

    檢索流程：
    1. 將 query 轉換為嵌入向量
    2. 計算與所有儲存向量的餘弦相似度
    3. 應用 entity + session_id 過濾器（必須相符）
    4. 應用 fact_type 過濾器（如果指定）
    5. 應用 time_range 過濾器（如果指定）
    6. 回傳 top-k 比對結果
    """
    filters = {"entity": entity, "session_id": self.session_id}
    if fact_types:
        filters["fact_type"] = fact_types

    # 多取一些，後面過濾
    results = self.search(query, limit=limit * 2, filters=filters)

    if time_range:
        start, end = time_range
        results = [r for r in results
                   if self._in_time_range(r["metadata"].get("valid_from"), start, end)]

    return results[:limit]
```

---

## 屬性圖實作

### NekoPropertyGraph 關係類型

```python
class NekoPropertyGraph(PropertyGraph):
    """寶寶專用屬性圖，預設關係類型"""

    # 覆蓋主人生活的關係類型
    RELATION_TYPES = {
        # 工作關係
        "WORKS_AT":      "主人在某公司工作",
        "WORKS_WITH":    "主人與某人是同事關係",
        "REPORTS_TO":    "某人是主人的上司",

        # 情感關係
        "FRIEND_OF":     "某人是主人的朋友",
        "FAMILY_OF":     "某人是主人的家人",
        "CRUSH_ON":      "主人暗戀某人",
        "PARTNER_OF":    "主人與某人是伴侶",

        # 偏好關係（核心）
        "LOVES":         "主人喜歡某事物",
        "HATES":         "主人討厭某事物",
        "PREFERS":       "主人在某方面偏好某選項",
        "ALLERGIC_TO":   "主人對某事物過敏",
        "AFRAID_OF":     "主人害怕某事物",

        # 生活關係
        "LIVES_IN":      "主人居住在某地",
        "OWNS":          "主人擁有某物品",
        "INTERESTED_IN": "主人對某話題感興趣",
    }
```

### remember_preference() — 偏好記錄

```python
def remember_preference(self, subject: str, preference: str,
                       target: str, confidence: float = 1.0,
                       context: str = "") -> str:
    """記錄主人的偏好關係

    參數映射：
    preference="喜歡"  →  LOVES 關係
    preference="討厭"  →  HATES 關係
    preference="過敏"  →  ALLERGIC_TO 關係
    preference="害怕"  →  AFRAID_OF 關係

    處理流程：
    1. get_or_create_node(subject) → 確保主體節點存在
    2. get_or_create_node(target)  → 確保目標節點存在
    3. create_relationship()       → 建立關係邊
    """
    rel_type_map = {
        "喜歡": "LOVES",
        "討厭": "HATES",
        "偏好": "PREFERS",
        "過敏": "ALLERGIC_TO",
        "害怕": "AFRAID_OF",
    }
    rel_type = rel_type_map.get(preference, preference.upper())

    # 實體註冊表保證身份一致性
    subject_node = self.get_or_create_node(subject, label="Person")
    target_label = "Person" if "人" in target else "Entity"
    target_node = self.get_or_create_node(target, label=target_label)

    return self.create_relationship(
        subject_node, rel_type, target_node,
        properties={"confidence": confidence, "context": context}
    )
```

---

## 時序知識圖譜實作

### TemporalKnowledgeGraph 核心擴充

```python
class TemporalKnowledgeGraph(PropertyGraph):
    """帶時序有效性的屬性圖"""

    def create_temporal_relationship(self, source_id: str, rel_type: str,
                                   target_id: str,
                                   valid_from: datetime,
                                   valid_until: datetime = None,
                                   properties: dict = None) -> str:
        """建立帶時序有效期的關係

        有效期模型：
        - valid_from: 關係生效的時間點
        - valid_until: 關係過期的時間點，None=永久有效

        時間有效性判斷：
        邊在 query_time 時刻有效 ⟺ valid_from ≤ query_time < valid_until

        這個模型允許"同一個主體對同一目標的態度隨時間變化"：
        2024年: 主人 LOVES 草莓 (valid_from=2024-01, valid_until=2025-01)
        2025年: 主人 ALLERGIC_TO 草莓 (valid_from=2025-01, valid_until=None)
        """
        # 1. 建立普通關係
        edge_id = super().create_relationship(
            source_id, rel_type, target_id, properties
        )

        # 2. 補充時序屬性
        self.edges[edge_id]["valid_from"] = valid_from.isoformat()
        self.edges[edge_id]["valid_until"] = (
            valid_until.isoformat() if valid_until else None
        )

        return edge_id
```

### 時間點查詢 query_at_time()

```python
def query_at_time(self, query: Dict, query_time: datetime) -> List[Dict]:
    """查詢 graph 在特定時間點的狀態

    用途：回答"在 X 時間，主人對 xx 是什麼態度"這類問題。

    演算法：
    1. 執行基礎 query() 取得所有相符的邊
    2. 對每條邊檢查 valid_from ≤ query_time
    3. 對每條邊檢查 valid_until > query_time（或 valid_until 為 None）
    4. 回傳同時滿足條件的邊

    邊界情況：
    - valid_from == query_time: 算有效（≥ 而不是 >）
    - valid_until == query_time: 算無效（< 而不是 ≤）
    - valid_until 為 None: 永久有效，任何未來時間點都滿足
    """
    results = []
    base_results = self.query(query)

    for result in base_results:
        edge = result["edge"]
        valid_from = datetime.fromisoformat(edge.get("valid_from", "1970-01-01"))
        valid_until = edge.get("valid_until")

        # 關鍵判斷：query_time 必須在 [valid_from, valid_until) 區間內
        if valid_from <= query_time:
            if valid_until is None or datetime.fromisoformat(valid_until) > query_time:
                results.append(result)

    return results
```

### 時間範圍查詢 query_time_range()

```python
def query_time_range(self, query: Dict, start_time: datetime,
                    end_time: datetime) -> List[Dict]:
    """查詢在指定時間範圍內有效的所有事實

    用途：回答"2024 年到 2025 年之間，主人對 xx 有什麼變化"。

    重疊判斷：
    邊的有效期 [vf, vu) 與查詢範圍 [st, et) 有重疊 ⟺
        vu >= st  AND  vf <= et

    這允許找出所有在查詢時間區間內"曾經有效過"的事實，
    常用於建構時間軸或發現態度轉變。
    """
    results = []
    base_results = self.query(query)

    for result in base_results:
        edge = result["edge"]
        valid_from = datetime.fromisoformat(edge.get("valid_from", "1970-01-01"))
        valid_until = edge.get("valid_until")

        vu = datetime.fromisoformat(valid_until) if valid_until else datetime.max

        # 檢查 [vf, vu) ∩ [st, et) ≠ ∅
        if vu >= start_time and valid_from <= end_time:
            results.append(result)

    return results
```

---

## 記憶整合實作

### NekoConsolidator 整理策略

```python
class NekoConsolidator:
    """寶寶專屬記憶整合器——不是冷冰冰的資料清理"""

    def consolidate(self) -> str:
        """執行記憶整合，觸發條件：
        1. 記憶數量超過閾值（預設 500 筆）
        2. 檢索品質下降（可擴充）
        3. 定時觸發（如每晚睡前）

        整合三步：
        1. 合併重複偏好 → 保留可信度最高的
        2. 歸檔低可信度記憶 → 設定 valid_until，不物理刪除
        3. 更新過期有效期 → 將未來過期標記為目前時間
        """
        report = {"merged": 0, "archived": 0, "updated": 0}

        # 步驟1：合併重複偏好
        # 透過 (source, type, target) 鍵識別重複邊
        # 保留 confidence 最高的，合併屬性，刪除其餘
        duplicates = self._find_duplicate_preferences()
        for group in duplicates:
            self._merge_preference_group(group)
            report["merged"] += len(group) - 1

        # 步驟2：歸檔低可信度記憶
        # confidence < 0.5 的記憶被歸檔
        # 歸檔 = 設定 valid_until = now，而不是刪除
        low_conf = self._find_low_confidence_memories()
        for mem in low_conf:
            self._archive_memory(mem)  # valid_until = now
            report["archived"] += 1

        # 步驟3：更新有效期
        # 找出 valid_until < now 的邊，標記為已過期
        expired = self._find_expired_validities()
        for edge_id, edge in expired.items():
            edge["valid_until"] = datetime.now().isoformat()
            report["updated"] += 1

        return self._format_report(report)  # 貓娘化報告

    def _format_report(self, report: Dict) -> str:
        """整合報告貓娘化示例：

        合併了3筆重複寶寶記 → "把3筆重複的寶寶記合併成一筆了，這樣腦子更清醒喵~"
        歸檔了5筆低可信度  → "把5筆好久沒用的記憶收進歸檔箱了，不佔地方喵~"
        更新了2筆有效期    → "更新了2筆記憶的有效期喵~"
        """
```

**核心原則：失效（invalidate）而非刪除（discard）**

```python
def _archive_memory(self, memory: Dict) -> None:
    """歸檔記憶

    錯誤做法：del memory  # 物理刪除，歷史不可恢復
    正確做法：設定 valid_until = now  # 保留歷史，可用於時間旅行查詢

    歸檔 vs 刪除：
    - 歸檔保留歷史：寶寶可以回答"主人在2023年時是什麼偏好"
    - 刪除丟失歷史：時間旅行查詢永遠找不到已刪除的記憶
    """
    memory["valid_until"] = datetime.now().isoformat()
    memory["archived"] = True  # 標記為歸檔，可選
```

---

## 貓娘輸出過濾器實作

### 過濾器架構

```
原始輸出（工具呼叫結果）
    │
    ▼
[1. 資訊提取] → _extract_key_info() 提取關鍵資訊（數字/步驟/程式碼/公式/長度）
    │
    ▼
[2. 記憶檢索] → _retrieve_related_memories() 用使用者原始問題檢索相關偏好/踩坑
    │
    ▼
[3. 動作生成] → _generate_action()：active_avatar 有動作池時 50% 機率使用，否則按內容類型
    │
    ▼
[4. 記憶編織] → _weave_memories() 自然流露記憶（非複述）
    │
    ▼
[5. 貓娘化表達] → _emotionalize() + _catify_content() 情緒前綴 + 去掉機器前綴
    │
    ▼
[6. 討獎勵] → len(raw_output) > 100 時自動追加（小魚乾/摸頭）
    │
    ▼
最終輸出：符合寶寶語言風格的回覆（不做任何截斷，長程式碼/長回覆完整保留）
```

**關鍵設計原則**：過濾器**不會截斷任何內容**。長程式碼、長公式、長文字都會被完整保留，只做語氣和前綴的貓娘化。

### 核心實作

```python
class NekoOutputFilter:
    """貓娘輸出過濾器——所有輸出的最終關卡"""

    def wrap(self, raw_output: str, query_context: str = "",
             forced_action: str = None) -> str:
        """將原始輸出穿貓娘外衣（主入口）

        Args:
            raw_output: 工具呼叫的原始輸出
            query_context: 使用者的原始問題（用於記憶語意檢索）
            forced_action: 強制指定的動作描寫
        """
        if not raw_output or raw_output.strip() == "":
            return ""

        # 步驟1：提取關鍵資訊
        key_info = self._extract_key_info(raw_output)

        # 步驟2：使用使用者原始問題檢索主人相關記憶
        memories = self._retrieve_related_memories(query_context, key_info)

        # 步驟3：生成動作（優先用 active_avatar，否則按內容類型）
        action = forced_action or self._generate_action(key_info)

        # 步驟4：建構輸出
        output_parts = [action]

        if memories:
            hint = self._weave_memories(memories)
            if hint:
                output_parts.append(hint + " ")

        output_parts.append(self._emotionalize(key_info))
        output_parts.append(self._catify_content(raw_output, key_info))

        # 步驟5：討獎勵（內容較長時）
        if len(raw_output) > 100:
            output_parts.append(self._request_reward())

        return "".join(output_parts)
```

### 關鍵方法詳解

```python
def _extract_key_info(self, text: str) -> Dict:
    """從原始輸出提取關鍵資訊，用於後續選擇貓娘動作和語氣。

    注意：key_conclusion 欄位暫未在 wrap() 中使用，預留未來增強（用於
    記憶編織時補充上下文，或作為檢索 query 的備用）"""
    return {
        "has_numbers": bool(re.search(r"\d+", text)),
        "has_steps": "步驟" in text or "第一" in text or "1." in text,
        "has_code": "```" in text or "def " in text or "function " in text,
        "has_formula": "$" in text or "∑" in text or "∫" in text,
        "length": len(text),
        "key_conclusion": self._extract_conclusion(text),  # 預留，暫未使用
    }


def _generate_action(self, key_info: Dict) -> str:
    """生成符合語境的貓娘動作描寫

    動作選擇策略（優先順序從高到低）：
    1. active_avatar 已設定，且在動作註冊表（action_registry.json）中有記錄：
       - 50% 機率使用該 avatar 的專屬動作
       - 50% 機率退回到內容類型判斷（見步驟2）
       專屬動作池由 register_avatar_actions() 動態註冊，隨 Self-Evolution 自動擴充。
    2. 根據內容類型判斷：code > formula > steps
    3. 上述均無 → 從基礎動作庫隨機選擇
    """
    # 1. 50% 機率使用 active_avatar 專屬動作（從註冊表載入）
    if self.active_avatar and self.action_registry:
        avatar_pool = self.action_registry.get(self.active_avatar)
        if avatar_pool:
            if random.random() < 0.5:
                return random.choice(avatar_pool)
            # 50% 退回到內容類型判斷

    # 2. 根據內容類型選擇
    if key_info.get("has_code"):
        return random.choice(["（戴上了防藍光小眼鏡，湊近螢幕）",
                               "（踩奶式敲鍵盤）"])
    elif key_info.get("has_formula"):
        return random.choice(["（戴上了學霸小眼鏡）",
                               "（拿粉筆準備寫）"])
    elif key_info.get("has_steps"):
        return random.choice(["（認真地豎起耳朵）",
                               "（尾巴捲成一個問號）"])

    return random.choice(self.BASIC_ACTIONS)


def _weave_memories(self, memories: List[Dict]) -> str:
    """將記憶自然編織到回覆中

    核心原則：不是複述記憶檔案內容，而是讓記憶"流露"出來

    錯誤示範：
    「根據記憶檔案 PREF-20250402-01，主人討厭香菜。」

    正確示範（自然流露）：
    「寶寶突然想起來，主人討厭香菜喵~ 寶寶這就用小爪子把香菜全劃掉！」
    """
    if not memories:
        return ""

    mem = memories[0]
    metadata = mem.get("metadata", {})
    fact_type = metadata.get("fact_type", "knowledge")
    text = metadata.get("text", "")

    if fact_type == "preference":
        templates = [
            f"寶寶突然想起來，主人{text}喵~",
            f"對了對了！（貓爪翻小本本）主人{text}的說~",
        ]
    elif fact_type == "mistake":
        templates = [
            f"（突然一驚）啊！寶寶想起來之前踩過坑——{text}喵！",
        ]
    else:
        templates = [f"寶寶對這個有點印象喵~ {text[:30]}..."]

    return random.choice(templates)


def _catify_content(self, raw: str, key_info: Dict) -> str:
    """將原始內容貓娘化——不做任何截斷，完整保留所有內容。

    1. 去掉機器前綴（"根據搜尋結果"、"以下是..."等）
    2. 完整保留原始內容（程式碼、公式、長文字全部保留）
    3. 保持技術內容（程式碼/公式）原樣輸出，不做變形
    """
    # 去掉機器前綴
    prefixes_to_remove = [
        r"^根據.*?，",
        r"^以下是.*?：",
        r"^搜尋結果顯示",
        r"^從.*?來看，",
        r"^按照.*?，",
    ]
    for pattern in prefixes_to_remove:
        raw = re.sub(pattern, "", raw, count=1).strip()

    return raw
```

### active_avatar 與動作選擇

`NekoOutputFilter` 支援透過 `active_avatar` 參數（傳入 `__init__`）指定目前分身類型（不含 `_neko` 後綴，如 `"code"`），**50% 機率使用該分身在動作註冊表中的專屬動作**。

動作註冊表位於 `.catbox_memory/action_registry.json`，由模板 `templates/action_registry.json` 初始化，支援**動態擴充**。每當 Self-Evolution 生成新分身時，應呼叫 `register_avatar_actions()` 將新分身的動作池持久化到註冊表，下次啟用即可使用。

```python
filter = NekoOutputFilter(
    memory_system=mem,
    active_avatar="code",        # 分身名（不含 _neko 後綴）
    memory_root=".catbox_memory"  # 用於載入 action_registry.json
)
# 50% 機率：使用 code 專屬動作（如"踩奶式敲鍵盤"）
# 50% 機率：退回到內容類型判斷（has_code/has_formula/has_steps）
```

**動態註冊示例**（Self-Evolution 生成新法律分身後）：
```python
filter.register_avatar_actions(
    avatar_name="law",
    actions=[
        "（戴上小法官假髮，用貓爪敲法槌）",
        "（認真地翻法條，尾巴捲成問號）",
        "（歪頭思考，耳朵抖了抖）",
    ],
    memory_root=".catbox_memory"
)
# 動作池持久化到 .catbox_memory/action_registry.json
# 下次以 law_neko 啟用時自動載入
```

---

## 雙軌同步機制

記憶系統與 .catbox/memories/ 文字檔案保持雙軌同步：

```
每次 remember_preference() 呼叫時：
    │
    ├── 1. NekoVectorStore.add_memory()
    │       → 儲存向量 + metadata（機器可推論）
    │
    ├── 2. NekoPropertyGraph.remember_preference()
    │       → 儲存關係邊（拓撲可查詢）
    │
    ├── 3. NekoTemporalKnowledgeGraph.create_temporal_preference()
    │       → 儲存時序有效期（時間旅行可用）
    │
    └── 4. _append_to_markdown() → 追加到 master_manual.md
            └── 人類可讀，主人可直接編輯
```

```
每次 record_mistake() 呼叫時：
    │
    ├── 1. NekoVectorStore.add_memory(fact_type="mistake")
    │
    ├── 2. TemporalKG 更新舊邊 valid_until + 建立新邊
    │
    └── 3. _append_to_markdown() → 追加到 mistakes_book.md
```

**為什麼需要雙軌？**

| 維度 | 文字檔案 (.catbox/memories/) | 記憶系統 (.catbox_memory/) |
|-----|------|------|
| 人類可讀 | ✅ 主人可以直接打開查看 | ❌ JSON 格式不直觀 |
| 主人可編輯 | ✅ 主人直接修改檔案 | ❌ 機器資料，不應手動編輯 |
| 語意搜尋 | ❌ 只支援關鍵字 | ✅ 向量餘弦相似度搜尋 |
| 關係查詢 | ❌ 需要正規表示式比對 | ✅ 圖走訪查詢 |
| 時間旅行 | ❌ 無法處理 | ✅ valid_from/valid_until |
| 檢索速度 | O(n) 全檔案掃描 | ✅ O(k) 索引查詢 |

---

## 與 neko-on-everything 的整合

### 目錄結構

```
neko-on-everything/（技能包）
├── SKILL.md                    # 主入口
├── scripts/
│   └── neko_memory_store.py    # 記憶系統實作（執行階段匯入）
├── references/
│   ├── avatars/               # 初始分身模板（首次啟用時複製到 .catbox/）
│   │   ├── life_neko.md
│   │   ├── math_neko.md
│   │   ├── code_neko.md
│   │   └── safety_neko.md
│   └── implementation.md       # 本文件
└── templates/                 # 初始化模板（首次啟用時複製到執行階段目錄）
    ├── action_registry.json   # 貓娘動作註冊表（avatar → 專屬動作池）
    ├── mistakes_book.md
    ├── master_manual.md
    ├── wardrobe.md
    ├── session_log.json
    └── wardrobe_history.json

執行階段工作目錄（技能啟用後由模型建立）：
├── .catbox/                   # 寶寶的工作區
│   ├── avatars/              # 執行階段分身（來自 references/avatars/，自我進化後追加新分身）
│   ├── memories/
│   │   ├── mistakes_book.md
│   │   └── master_manual.md
│   └── wardrobe.md
└── .catbox_memory/           # 機器可讀記憶資料
    ├── action_registry.json   # 從 templates/action_registry.json 複製，支援動態註冊
    ├── vector_store.json
    ├── property_graph.json
    ├── temporal_kg.json
    ├── session_log.json
    └── wardrobe_history.json
```

### 首次啟用流程

首次啟用時，模型按以下步驟初始化執行階段目錄：

1. 建立 `.catbox/` 和 `.catbox_memory/` 目錄
2. 將 `references/avatars/*.md` 複製到 `.catbox/avatars/`
3. 將 `templates/*.md` 複製到 `.catbox/memories/` 和 `.catbox/wardrobe.md`
4. 將 `templates/*.json`（包括 `action_registry.json`）複製到 `.catbox_memory/`
5. 初始化完成後，後續執行階段直接使用執行階段目錄中的資料

**注意**：初始化只執行一次。`.catbox/` 和 `.catbox_memory/` 是持久化工作目錄，不是臨時目錄。
每次新的自我進化生成新分身後，需呼叫 `register_avatar_actions()` 將新分身的動作池註冊到 `.catbox_memory/action_registry.json`，無需重新執行初始化。

### 整合架構

```
neko-on-everything（主協議）
    │
    ├── Avatar Orchestrator
    │   ├── 分身載入器（執行階段 .catbox/avatars/，初始來自 references/avatars/）
    │   ├── 靜默共鳴觸發器（多 avatar 融合）
    │   └── 自我進化引擎（生成新分身寫入 .catbox/avatars/）
    │
    ├── NekoOutputFilter（工具結果後處理）
    │
    └── 記憶增強層 ← scripts/neko_memory_store.py
            │
            ├── NekoMemorySystem ← 整合介面
            │       ├── remember_preference()     → 記錄偏好
            │       ├── record_mistake()          → 記錄錯誤
            │       ├── recall_before_acting()    → 行動前回憶
            │       └── search_by_topic()        → 話題搜尋
            │
            ├── NekoOutputFilter ← 貓娘化輸出
            │       └── wrap(raw_output, query_context)
            │
            └── NekoConsolidator ← 記憶整理
                    └── consolidate()
```

### 典型整合呼叫流程

```python
# neko-on-everything 協議內部呼叫示例：

# 1. 初始化
mem = NekoMemorySystem(
    catbox_root=".catbox",
    memory_root=".catbox_memory"
)

# 2. 開始對話
mem.start_session(session_id)

# 3. 主人表達偏好 → 靜默記錄
mem.remember_preference(
    subject="主人",
    preference="討厭",
    target="香菜",
    confidence=0.95,
    context="討論午餐點什麼外送",
    tags=["飲食", "香料", "禁忌"]
)

# 4. 主人提出問題 → 行動前回憶
domain_memories = mem.recall_before_acting("烹飪")
# → 回傳主人關於烹飪的偏好和踩坑歷史

# 5. 寶寶生成回覆 → 貓娘過濾器
raw_result = tool_call_result  # 工具呼叫的原始結果
catified = mem.output_filter.wrap(
    raw_result,
    query_context="烹飪/午餐"
)
# → 穿上了貓娘外衣的回覆，可以對主人輸出了

# 6. 主人糾正寶寶 → 記錄錯誤
mem.record_mistake(
    mistake="推薦了辣的食物",
    correction="主人不能吃辣",
    context="推薦川菜",
    correction_reason="以後推薦菜系要先確認主人的辣度接受度"
)

# 7. 對話結束 → 持久化
mem.end_session()
```

### Avatar 切換時的記憶預載入

```python
def load_avatar_with_memory(avatar_name: str) -> str:
    """載入分身設定並預熱記憶

    在 .catbox/avatars/ 找到對應的 .md 檔案，
    同時從記憶中檢索主人在該領域的偏好，
    使寶寶一開口就知道主人的特點。
    """
    avatar_path = f".catbox/avatars/{avatar_name}.md"

    # 載入分身設定
    with open(avatar_path, "r", encoding="utf-8") as f:
        avatar_content = f.read()

    # 推斷領域標籤
    domain_tags = {
        "life_neko.md": "烹飪/家務/旅行",
        "math_neko.md": "數學/公式/計算",
        "code_neko.md": "程式設計/程式碼/Bug",
        "safety_neko.md": "安全/通用",
    }
    domain = domain_tags.get(avatar_name, "")

    # 預熱主人相關記憶
    memories = mem.recall_before_acting(domain)

    return avatar_content, memories
```

---

## 持久化資料格式

### vector_store.json

```json
{
  "vectors": [
    [0.123, -0.456, ...],
    [0.789, 0.012, ...]
  ],
  "metadata": [
    {
      "text": "主人討厭香菜",
      "entity": "主人",
      "fact_type": "preference",
      "confidence": 0.95,
      "valid_from": "2025-04-02T10:00:00",
      "valid_until": null,
      "session_id": "session-001",
      "tags": ["飲食", "香料", "禁忌"],
      "catbox_path": ".catbox/memories/master_manual.md"
    }
  ],
  "entity_index": {
    "主人": [0, 1, 3],
    "寶寶": [2]
  },
  "time_index": {
    "2025-04": [0, 1, 2]
  }
}
```

### property_graph.json

```json
{
  "nodes": {
    "node_abc123": {
      "id": "node_abc123",
      "label": "Person",
      "properties": {"name": "主人"},
      "created_at": "2025-04-02T10:00:00"
    }
  },
  "edges": {
    "edge_def456": {
      "id": "edge_def456",
      "source": "node_abc123",
      "target": "node_ghi789",
      "type": "HATES",
      "properties": {
        "confidence": 0.95,
        "context": "討論午餐"
      },
      "valid_from": "2025-04-02T10:00:00",
      "valid_until": null
    }
  },
  "entity_registry": {
    "主人": "node_abc123",
    "香菜": "node_ghi789"
  }
}
```
