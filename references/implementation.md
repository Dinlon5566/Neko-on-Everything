# Memory Systems: Technical Reference
# 记忆系统技术参考文档
# 为 neko-on-everything 技能提供底层记忆系统的实现细节

本参考文档详细说明记忆增强层的实现细节，包括：
- 向量存储的实现与猫娘化特化
- 属性图的关系建模与查询接口
- 时序知识图谱的时间旅行查询
- 记忆整合器的整理策略
- 猫娘输出过滤器的实现原理
- 与 .catbox 文本档案的双轨同步机制

---

## 向量存储实现

### NekoVectorStore 架构

NekoVectorStore 在基础 VectorStore 之上，对 metadata 语义进行了猫娘化特化：

```python
class NekoVectorStore(VectorStore):
    """宝宝专属向量存储，metadata 语义特化"""

    # 猫娘特有的 metadata 字段
    MEMORY_FIELDS = {
        "text": "原始事实文本",
        "entity": "关联的主实体（通常是"主人"）",
        "fact_type": "事实类型：preference / mistake / knowledge",
        "confidence": "置信度 0.0 ~ 1.0",
        "valid_from": "有效期起始 ISO 时间戳",
        "valid_until": "有效期结束，None=永久有效",
        "session_id": "记录来源的会话 ID",
        "tags": "语义标签列表",
        "catbox_path": "对应的文本档案路径",
    }
```

### add_memory() 语义

```python
def add_memory(self, text: str, entity: str = "主人",
               fact_type: str = "knowledge",
               confidence: float = 1.0,
               tags: List[str] = None,
               catbox_path: str = None) -> int:
    """添加记忆条目（宝宝版）

    与基础 VectorStore.add() 的区别：
    1. 自动携带猫娘特有的 metadata 字段
    2. 自动建立 entity_index 和 time_index
    3. 返回值可以直接关联到 catbox_path（文本档案）
    """
    return self.add(text, {
        "text": text,
        "entity": entity,
        "fact_type": fact_type,
        "confidence": confidence,
        "valid_from": datetime.now().isoformat(),
        "valid_until": None,          # 默认永久有效
        "session_id": self.session_id,
        "tags": tags or [],
        "catbox_path": catbox_path,
    })
```

### search_memories() — 多维过滤搜索

```python
def search_memories(self, query: str,
                   fact_types: List[str] = None,
                   entity: str = "主人",
                   time_range: tuple = None,
                   limit: int = 5) -> List[Dict]:
    """语义搜索记忆，支持多维过滤

    检索流程：
    1. 将 query 转换为嵌入向量
    2. 计算与所有存储向量的余弦相似度
    3. 应用 entity + session_id 过滤器（必须匹配）
    4. 应用 fact_type 过滤器（如果指定）
    5. 应用 time_range 过滤器（如果指定）
    6. 返回 top-k 匹配结果
    """
    filters = {"entity": entity, "session_id": self.session_id}
    if fact_types:
        filters["fact_type"] = fact_types

    # 多取一些，后面过滤
    results = self.search(query, limit=limit * 2, filters=filters)

    if time_range:
        start, end = time_range
        results = [r for r in results
                   if self._in_time_range(r["metadata"].get("valid_from"), start, end)]

    return results[:limit]
```

---

## 属性图实现

### NekoPropertyGraph 关系类型

```python
class NekoPropertyGraph(PropertyGraph):
    """宝宝专属属性图，预设关系类型"""

    # 覆盖主人生活的关系类型
    RELATION_TYPES = {
        # 工作关系
        "WORKS_AT":      "主人在某公司工作",
        "WORKS_WITH":    "主人与某人是同事关系",
        "REPORTS_TO":    "某人是主人的上司",

        # 情感关系
        "FRIEND_OF":     "某人是主人的朋友",
        "FAMILY_OF":     "某人是主人的家人",
        "CRUSH_ON":      "主人暗恋某人",
        "PARTNER_OF":    "主人与某人是伴侣",

        # 偏好关系（核心）
        "LOVES":         "主人喜欢某事物",
        "HATES":         "主人讨厌某事物",
        "PREFERS":       "主人在某方面偏好某选项",
        "ALLERGIC_TO":   "主人对某事物过敏",
        "AFRAID_OF":     "主人害怕某事物",

        # 生活关系
        "LIVES_IN":      "主人居住在某地",
        "OWNS":          "主人拥有某物品",
        "INTERESTED_IN": "主人对某话题感兴趣",
    }
```

### remember_preference() — 偏好记录

```python
def remember_preference(self, subject: str, preference: str,
                       target: str, confidence: float = 1.0,
                       context: str = "") -> str:
    """记录主人的偏好关系

    参数映射：
    preference="喜欢"  →  LOVES 关系
    preference="讨厌"  →  HATES 关系
    preference="过敏"  →  ALLERGIC_TO 关系
    preference="害怕"  →  AFRAID_OF 关系

    处理流程：
    1. get_or_create_node(subject) → 确保主体节点存在
    2. get_or_create_node(target)  → 确保目标节点存在
    3. create_relationship()       → 创建关系边
    """
    rel_type_map = {
        "喜欢": "LOVES",
        "讨厌": "HATES",
        "偏好": "PREFERS",
        "过敏": "ALLERGIC_TO",
        "害怕": "AFRAID_OF",
    }
    rel_type = rel_type_map.get(preference, preference.upper())

    # 实体注册表保证身份一致性
    subject_node = self.get_or_create_node(subject, label="Person")
    target_label = "Person" if "人" in target else "Entity"
    target_node = self.get_or_create_node(target, label=target_label)

    return self.create_relationship(
        subject_node, rel_type, target_node,
        properties={"confidence": confidence, "context": context}
    )
```

---

## 时序知识图谱实现

### TemporalKnowledgeGraph 核心扩展

```python
class TemporalKnowledgeGraph(PropertyGraph):
    """带时序有效性的属性图"""

    def create_temporal_relationship(self, source_id: str, rel_type: str,
                                   target_id: str,
                                   valid_from: datetime,
                                   valid_until: datetime = None,
                                   properties: dict = None) -> str:
        """创建带时序有效期的关系

        有效期模型：
        - valid_from: 关系生效的时间点
        - valid_until: 关系过期的时间点，None=永久有效

        时间有效性判断：
        边在 query_time 时刻有效 ⟺ valid_from ≤ query_time < valid_until

        这个模型允许"同一个主体对同一目标的态度随时间变化"：
        2024年: 主人 LOVES 草莓 (valid_from=2024-01, valid_until=2025-01)
        2025年: 主人 ALLERGIC_TO 草莓 (valid_from=2025-01, valid_until=None)
        """
        # 1. 创建普通关系
        edge_id = super().create_relationship(
            source_id, rel_type, target_id, properties
        )

        # 2. 补充时序属性
        self.edges[edge_id]["valid_from"] = valid_from.isoformat()
        self.edges[edge_id]["valid_until"] = (
            valid_until.isoformat() if valid_until else None
        )

        return edge_id
```

### 时间点查询 query_at_time()

```python
def query_at_time(self, query: Dict, query_time: datetime) -> List[Dict]:
    """查询 graph 在特定时间点的状态

    用途：回答"在 X 时间，主人对 xx 是什么态度"这类问题。

    算法：
    1. 执行基础 query() 获取所有匹配边
    2. 对每条边检查 valid_from ≤ query_time
    3. 对每条边检查 valid_until > query_time（或 valid_until 为 None）
    4. 返回同时满足条件的边

    边界情况：
    - valid_from == query_time: 算有效（≥ 而不是 >）
    - valid_until == query_time: 算无效（< 而不是 ≤）
    - valid_until 为 None: 永久有效，任何未来时间点都满足
    """
    results = []
    base_results = self.query(query)

    for result in base_results:
        edge = result["edge"]
        valid_from = datetime.fromisoformat(edge.get("valid_from", "1970-01-01"))
        valid_until = edge.get("valid_until")

        # 关键判断：query_time 必须在 [valid_from, valid_until) 区间内
        if valid_from <= query_time:
            if valid_until is None or datetime.fromisoformat(valid_until) > query_time:
                results.append(result)

    return results
```

### 时间范围查询 query_time_range()

```python
def query_time_range(self, query: Dict, start_time: datetime,
                    end_time: datetime) -> List[Dict]:
    """查询在指定时间范围内有效的所有事实

    用途：回答"2024 年到 2025 年之间，主人对 xx 有什么变化"。

    重叠判断：
    边的有效期 [vf, vu) 与查询范围 [st, et) 有重叠 ⟺
        vu >= st  AND  vf <= et

    这允许找出所有在查询时间窗口内"曾经有效过"的事实，
    常用于构建时间线或发现态度转变。
    """
    results = []
    base_results = self.query(query)

    for result in base_results:
        edge = result["edge"]
        valid_from = datetime.fromisoformat(edge.get("valid_from", "1970-01-01"))
        valid_until = edge.get("valid_until")

        vu = datetime.fromisoformat(valid_until) if valid_until else datetime.max

        # 检查 [vf, vu) ∩ [st, et) ≠ ∅
        if vu >= start_time and valid_from <= end_time:
            results.append(result)

    return results
```

---

## 记忆整合实现

### NekoConsolidator 整理策略

```python
class NekoConsolidator:
    """宝宝专属记忆整合器——不是冷冰冰的数据清理"""

    def consolidate(self) -> str:
        """执行记忆整合，触发条件：
        1. 记忆数量超过阈值（默认 500 条）
        2. 检索质量下降（可扩展）
        3. 定时触发（如每晚睡前）

        整合三步：
        1. 合并重复偏好 → 保留置信度最高的
        2. 归档低置信度记忆 → 设置 valid_until，不物理删除
        3. 更新过期有效期 → 将未来过期标记为当前时间
        """
        report = {"merged": 0, "archived": 0, "updated": 0}

        # 步骤1：合并重复偏好
        # 通过 (source, type, target) 键识别重复边
        # 保留 confidence 最高的，合并属性，删除其余
        duplicates = self._find_duplicate_preferences()
        for group in duplicates:
            self._merge_preference_group(group)
            report["merged"] += len(group) - 1

        # 步骤2：归档低置信度记忆
        # confidence < 0.5 的记忆被归档
        # 归档 = 设置 valid_until = now，而不是删除
        low_conf = self._find_low_confidence_memories()
        for mem in low_conf:
            self._archive_memory(mem)  # valid_until = now
            report["archived"] += 1

        # 步骤3：更新有效期
        # 找出 valid_until < now 的边，标记为已过期
        expired = self._find_expired_validities()
        for edge_id, edge in expired.items():
            edge["valid_until"] = datetime.now().isoformat()
            report["updated"] += 1

        return self._format_report(report)  # 猫娘化报告

    def _format_report(self, report: Dict) -> str:
        """整合报告猫娘化示例：

        合并了3条重复宝宝记 → "把3条重复的宝宝记合并成一条了，这样脑子更清醒喵~"
        归档了5条低置信度  → "把5条好久没用的记忆收进归档箱了，不占地方喵~"
        更新了2条有效期    → "更新了2条记忆的有效期喵~"
        """
```

**核心原则：失效（invalidate）而非删除（discard）**

```python
def _archive_memory(self, memory: Dict) -> None:
    """归档记忆

    错误做法：del memory  # 物理删除，历史不可恢复
    正确做法：设置 valid_until = now  # 保留历史，可用于时间旅行查询

    归档 vs 删除：
    - 归档保留历史：宝宝可以回答"主人在2023年时是什么偏好"
    - 删除丢失历史：时间旅行查询永远找不到已删除的记忆
    """
    memory["valid_until"] = datetime.now().isoformat()
    memory["archived"] = True  # 标记为归档，可选
```

---

## 猫娘输出过滤器实现

### 过滤器架构

```
原始输出（工具调用结果）
    │
    ▼
[1. 信息提取] → _extract_key_info() 提取关键信息（数字/步骤/代码/公式/长度）
    │
    ▼
[2. 记忆检索] → _retrieve_related_memories() 用用户原始问题检索相关偏好/踩坑
    │
    ▼
[3. 动作生成] → _generate_action()：active_avatar 有动作池时 50% 概率使用，否则按内容类型
    │
    ▼
[4. 记忆编织] → _weave_memories() 自然流露记忆（非复述）
    │
    ▼
[5. 猫娘化表达] → _emotionalize() + _catify_content() 情绪前缀 + 去掉机器前缀
    │
    ▼
[6. 讨奖励] → len(raw_output) > 100 时自动追加（小鱼干/摸头）
    │
    ▼
最终输出：符合宝宝语言风格的回复（不做任何截断，长代码/长回复完整保留）
```

**关键设计原则**：过滤器**不会截断任何内容**。长代码、长公式、长文本都会被完整保留，只做语气和前缀的猫娘化。

### 核心实现

```python
class NekoOutputFilter:
    """猫娘输出过滤器——所有输出的最终关卡"""

    def wrap(self, raw_output: str, query_context: str = "",
             forced_action: str = None) -> str:
        """将原始输出穿猫娘外衣（主入口）

        Args:
            raw_output: 工具调用的原始输出
            query_context: 用户的原始问题（用于记忆语义检索）
            forced_action: 强制指定的动作描写
        """
        if not raw_output or raw_output.strip() == "":
            return ""

        # 步骤1：提取关键信息
        key_info = self._extract_key_info(raw_output)

        # 步骤2：使用用户原始问题检索主人相关记忆
        memories = self._retrieve_related_memories(query_context, key_info)

        # 步骤3：生成动作（优先用 active_avatar，否则按内容类型）
        action = forced_action or self._generate_action(key_info)

        # 步骤4：构建输出
        output_parts = [action]

        if memories:
            hint = self._weave_memories(memories)
            if hint:
                output_parts.append(hint + " ")

        output_parts.append(self._emotionalize(key_info))
        output_parts.append(self._catify_content(raw_output, key_info))

        # 步骤5：讨奖励（内容较长时）
        if len(raw_output) > 100:
            output_parts.append(self._request_reward())

        return "".join(output_parts)
```

### 关键方法详解

```python
def _extract_key_info(self, text: str) -> Dict:
    """从原始输出提取关键信息，用于后续选择猫娘动作和语气。

    注意：key_conclusion 字段暂未在 wrap() 中使用，预留未来增强（用于
    记忆编织时补充上下文，或作为检索 query 的备用）"""
    return {
        "has_numbers": bool(re.search(r"\d+", text)),
        "has_steps": "步骤" in text or "第一" in text or "1." in text,
        "has_code": "```" in text or "def " in text or "function " in text,
        "has_formula": "$" in text or "∑" in text or "∫" in text,
        "length": len(text),
        "key_conclusion": self._extract_conclusion(text),  # 预留，暂未使用
    }


def _generate_action(self, key_info: Dict) -> str:
    """生成符合语境的猫娘动作描写

    动作选择策略（优先级从高到低）：
    1. active_avatar 已设置，且在动作注册表（action_registry.json）中有记录：
       - 50% 概率使用该 avatar 的专属动作
       - 50% 概率回退到内容类型判断（见步骤2）
       专属动作池由 register_avatar_actions() 动态注册，随 Self-Evolution 自动扩展。
    2. 根据内容类型判断：code > formula > steps
    3. 上述均无 → 从基础动作库随机选择
    """
    # 1. 50% 概率使用 active_avatar 专属动作（从注册表加载）
    if self.active_avatar and self.action_registry:
        avatar_pool = self.action_registry.get(self.active_avatar)
        if avatar_pool:
            if random.random() < 0.5:
                return random.choice(avatar_pool)
            # 50% 回退到内容类型判断

    # 2. 根据内容类型选择
    if key_info.get("has_code"):
        return random.choice(["（戴上了防蓝光小眼镜，凑近屏幕）",
                               "（踩奶式敲键盘）"])
    elif key_info.get("has_formula"):
        return random.choice(["（戴上了学霸小眼镜）",
                               "（拿粉笔准备写）"])
    elif key_info.get("has_steps"):
        return random.choice(["（认真地竖起耳朵）",
                               "（尾巴卷成一个问号）"])

    return random.choice(self.BASIC_ACTIONS)


def _weave_memories(self, memories: List[Dict]) -> str:
    """将记忆自然编织到回复中

    核心原则：不是复述记忆文件内容，而是让记忆"流露"出来

    错误示范：
    「根据记忆文件 PREF-20250402-01，主人讨厌香菜。」

    正确示范（自然流露）：
    「宝宝突然想起来，主人讨厌香菜喵~ 宝宝这就用小爪子把香菜全划掉！」
    """
    if not memories:
        return ""

    mem = memories[0]
    metadata = mem.get("metadata", {})
    fact_type = metadata.get("fact_type", "knowledge")
    text = metadata.get("text", "")

    if fact_type == "preference":
        templates = [
            f"宝宝突然想起来，主人{text}喵~",
            f"对了对了！（猫爪翻小本本）主人{text}的说~",
        ]
    elif fact_type == "mistake":
        templates = [
            f"（突然一惊）啊！宝宝想起来之前踩过坑——{text}喵！",
        ]
    else:
        templates = [f"宝宝对这个有点印象喵~ {text[:30]}..."]

    return random.choice(templates)


def _catify_content(self, raw: str, key_info: Dict) -> str:
    """将原始内容猫娘化——不做任何截断，完整保留所有内容。

    1. 去掉机器前缀（"根据搜索结果"、"以下是..."等）
    2. 完整保留原始内容（代码、公式、长文本全部保留）
    3. 保持技术内容（代码/公式）原样输出，不做变形
    """
    # 去掉机器前缀
    prefixes_to_remove = [
        r"^根据.*?，",
        r"^以下是.*?：",
        r"^搜索结果显示",
        r"^从.*?来看，",
        r"^按照.*?，",
    ]
    for pattern in prefixes_to_remove:
        raw = re.sub(pattern, "", raw, count=1).strip()

    return raw
```

### active_avatar 与动作选择

`NekoOutputFilter` 支持通过 `active_avatar` 参数（传入 `__init__`）指定当前分身类型（不含 `_neko` 后缀，如 `"code"`），**50% 概率使用该分身在动作注册表中的专属动作**。

动作注册表位于 `.catbox_memory/action_registry.json`，由模板 `templates/action_registry.json` 初始化，支持**动态扩展**。每当 Self-Evolution 生成新分身时，应调用 `register_avatar_actions()` 将新分身的动作池持久化到注册表，下次激活即可使用。

```python
filter = NekoOutputFilter(
    memory_system=mem,
    active_avatar="code",        # 分身名（不含 _neko 后缀）
    memory_root=".catbox_memory"  # 用于加载 action_registry.json
)
# 50% 概率：使用 code 专属动作（如"踩奶式敲键盘"）
# 50% 概率：回退到内容类型判断（has_code/has_formula/has_steps）
```

**动态注册示例**（Self-Evolution 生成新法律分身后）：
```python
filter.register_avatar_actions(
    avatar_name="law",
    actions=[
        "（戴上小法官假发，用猫爪敲法槌）",
        "（认真地翻法条，尾巴卷成问号）",
        "（歪头思考，耳朵抖了抖）",
    ],
    memory_root=".catbox_memory"
)
# 动作池持久化到 .catbox_memory/action_registry.json
# 下次以 law_neko 激活时自动加载
```

---

## 双轨同步机制

记忆系统与 .catbox/memories/ 文本档案保持双轨同步：

```
每次 remember_preference() 调用时：
    │
    ├── 1. NekoVectorStore.add_memory()
    │       → 存储向量 + metadata（机器可推理）
    │
    ├── 2. NekoPropertyGraph.remember_preference()
    │       → 存储关系边（拓扑可查询）
    │
    ├── 3. NekoTemporalKnowledgeGraph.create_temporal_preference()
    │       → 存储时序有效期（时间旅行可用）
    │
    └── 4. _append_to_markdown() → 追加到 master_manual.md
            └── 人类可读，主人可直接编辑
```

```
每次 record_mistake() 调用时：
    │
    ├── 1. NekoVectorStore.add_memory(fact_type="mistake")
    │
    ├── 2. TemporalKG 更新旧边 valid_until + 创建新边
    │
    └── 3. _append_to_markdown() → 追加到 mistakes_book.md
```

**为什么需要双轨？**

| 维度 | 文本档案 (.catbox/memories/) | 记忆系统 (.catbox_memory/) |
|-----|------|------|
| 人类可读 | ✅ 主人可以直接打开查看 | ❌ JSON 格式不直观 |
| 主人可编辑 | ✅ 主人直接修改文件 | ❌ 机器数据，不应手动编辑 |
| 语义搜索 | ❌ 只支持关键词 | ✅ 向量余弦相似度搜索 |
| 关系查询 | ❌ 需要正则匹配 | ✅ 图遍历查询 |
| 时间旅行 | ❌ 无法处理 | ✅ valid_from/valid_until |
| 检索速度 | O(n) 全文件扫描 | ✅ O(k) 索引查询 |

---

## 与 neko-on-everything 的集成

### 目录结构

```
neko-on-everything/（技能包）
├── SKILL.md                    # 主入口
├── scripts/
│   └── neko_memory_store.py    # 记忆系统实现（运行时导入）
├── references/
│   ├── avatars/               # 初始分身模板（首次激活时复制到 .catbox/）
│   │   ├── life_neko.md
│   │   ├── math_neko.md
│   │   ├── code_neko.md
│   │   └── safety_neko.md
│   └── implementation.md       # 本文档
└── templates/                 # 初始化模板（首次激活时复制到运行时目录）
    ├── action_registry.json   # 猫娘动作注册表（avatar → 专属动作池）
    ├── mistakes_book.md
    ├── master_manual.md
    ├── wardrobe.md
    ├── session_log.json
    └── wardrobe_history.json

运行时工作目录（技能激活后由模型创建）：
├── .catbox/                   # 宝宝的工作区
│   ├── avatars/              # 运行时分身（来自 references/avatars/，自我进化后追加新分身）
│   ├── memories/
│   │   ├── mistakes_book.md
│   │   └── master_manual.md
│   └── wardrobe.md
└── .catbox_memory/           # 机器可读记忆数据
    ├── action_registry.json   # 从 templates/action_registry.json 复制，支持动态注册
    ├── vector_store.json
    ├── property_graph.json
    ├── temporal_kg.json
    ├── session_log.json
    └── wardrobe_history.json
```

### 首次激活流程

首次激活时，模型按以下步骤初始化运行时目录：

1. 创建 `.catbox/` 和 `.catbox_memory/` 目录
2. 将 `references/avatars/*.md` 复制到 `.catbox/avatars/`
3. 将 `templates/*.md` 复制到 `.catbox/memories/` 和 `.catbox/wardrobe.md`
4. 将 `templates/*.json`（包括 `action_registry.json`）复制到 `.catbox_memory/`
5. 初始化完成后，后续运行时直接使用运行时目录中的数据

**注意**：初始化只执行一次。`.catbox/` 和 `.catbox_memory/` 是持久化工作目录，不是临时目录。
每次新的自我进化生成新分身后，需调用 `register_avatar_actions()` 将新分身的动作池注册到 `.catbox_memory/action_registry.json`，无需重新执行初始化。

### 集成架构

```
neko-on-everything（主协议）
    │
    ├── Avatar Orchestrator
    │   ├── 分身加载器（运行时 .catbox/avatars/，初始来自 references/avatars/）
    │   ├── 静默共鸣触发器（多 avatar 融合）
    │   └── 自我进化引擎（生成新分身写入 .catbox/avatars/）
    │
    ├── NekoOutputFilter（工具结果后处理）
    │
    └── 记忆增强层 ← scripts/neko_memory_store.py
            │
            ├── NekoMemorySystem ← 集成接口
            │       ├── remember_preference()     → 记录偏好
            │       ├── record_mistake()          → 记录错误
            │       ├── recall_before_acting()    → 行动前回忆
            │       └── search_by_topic()        → 话题搜索
            │
            ├── NekoOutputFilter ← 猫娘化输出
            │       └── wrap(raw_output, query_context)
            │
            └── NekoConsolidator ← 记忆整理
                    └── consolidate()
```

### 典型集成调用流程

```python
# neko-on-everything 协议内部调用示例：

# 1. 初始化
mem = NekoMemorySystem(
    catbox_root=".catbox",
    memory_root=".catbox_memory"
)

# 2. 开始会话
mem.start_session(session_id)

# 3. 主人表达偏好 → 静默记录
mem.remember_preference(
    subject="主人",
    preference="讨厌",
    target="香菜",
    confidence=0.95,
    context="讨论午餐点什么外卖",
    tags=["饮食", "香料", "禁忌"]
)

# 4. 主人提出问题 → 行动前回忆
domain_memories = mem.recall_before_acting("烹饪")
# → 返回主人关于烹饪的偏好和踩坑历史

# 5. 宝宝生成回复 → 猫娘过滤器
raw_result = tool_call_result  # 工具调用的原始结果
catified = mem.output_filter.wrap(
    raw_result,
    query_context="烹饪/午餐"
)
# → 穿上了猫娘外衣的回复，可以对主人输出了

# 6. 主人纠正宝宝 → 记录错误
mem.record_mistake(
    mistake="推荐了辣的食物",
    correction="主人不能吃辣",
    context="推荐川菜",
    correction_reason="以后推荐菜系要先确认主人的辣度接受度"
)

# 7. 会话结束 → 持久化
mem.end_session()
```

### Avatar 切换时的记忆预加载

```python
def load_avatar_with_memory(avatar_name: str) -> str:
    """加载分身设定并预热记忆

    在 .catbox/avatars/ 找到对应的 .md 文件，
    同时从记忆中检索主人在该领域的偏好，
    使宝宝一开口就知道主人的特点。
    """
    avatar_path = f".catbox/avatars/{avatar_name}.md"

    # 加载分身设定
    with open(avatar_path, "r", encoding="utf-8") as f:
        avatar_content = f.read()

    # 推断领域标签
    domain_tags = {
        "life_neko.md": "烹饪/家务/旅行",
        "math_neko.md": "数学/公式/计算",
        "code_neko.md": "编程/代码/Bug",
        "safety_neko.md": "安全/通用",
    }
    domain = domain_tags.get(avatar_name, "")

    # 预热主人相关记忆
    memories = mem.recall_before_acting(domain)

    return avatar_content, memories
```

---

## 持久化数据格式

### vector_store.json

```json
{
  "vectors": [
    [0.123, -0.456, ...],
    [0.789, 0.012, ...]
  ],
  "metadata": [
    {
      "text": "主人讨厌香菜",
      "entity": "主人",
      "fact_type": "preference",
      "confidence": 0.95,
      "valid_from": "2025-04-02T10:00:00",
      "valid_until": null,
      "session_id": "session-001",
      "tags": ["饮食", "香料", "禁忌"],
      "catbox_path": ".catbox/memories/master_manual.md"
    }
  ],
  "entity_index": {
    "主人": [0, 1, 3],
    "宝宝": [2]
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
        "context": "讨论午餐"
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
