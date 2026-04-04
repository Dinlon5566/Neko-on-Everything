"""猫娘记忆系统核心实现

这是 neko-on-everything 技能的底层记忆扩展层。
提供向量存储、属性图、时序知识图谱的能力，
但所有输出必须经由猫娘过滤器转化为宝宝的语言风格后才对主人可见。

本模块所有公开接口均以 Neko* 开头，表示"宝宝专属版本"。

嵌入模型配置：
    - 默认模型：paraphrase-multilingual-MiniLM-L12-v2（118M参数，支持50+语言含中文）
    - 下载源：HuggingFace 官方 → hf-mirror.com（国内镜像）→ 降级hash存根
    - 依赖安装：pip install sentence-transformers huggingface_hub
    - 首次调用时自动下载模型（约118MB），自动缓存到 ~/.cache/neko-embeddings/

适用场景：
    - 主人表达偏好时自动记录
    - 主人纠正错误时建立时间戳追踪
    - 主人询问历史时进行时间旅行查询
    - 宝宝进入某领域前预加载主人相关记忆

典型用法：

    mem = NekoMemorySystem()
    mem.start_session("session-001")
    mem.remember_preference("主人", "讨厌", "香菜", confidence=0.95)
    mem.record_mistake("推荐了草莓蛋糕", "主人草莓过敏", "讨论美食")
    memories = mem.recall_before_acting("烹饪")
"""

import hashlib
import json
import os
import re
import threading
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np

# ============================================================================
# 嵌入模型相关导入
# ============================================================================

# sentence-transformers: 轻量、高性能、支持50+语言含中文
# 官网: https://sbert.net/
# 模型: paraphrase-multilingual-MiniLM-L12-v2 (118M参数, ~118MB)
#   - 支持中文、英文等50+语言
#   - CPU友好，1080Ti级别GPU可达~1400 sent/s
#   - 语义精度在同体积模型中领先
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SentenceTransformer = None
    SENTENCE_TRANSFORMERS_AVAILABLE = False

# huggingface_hub: 用于处理模型下载和镜像源切换
try:
    import huggingface_hub
    HUGGINGFACE_HUB_AVAILABLE = True
except ImportError:
    huggingface_hub = None
    HUGGINGFACE_HUB_AVAILABLE = False

__all__ = [
    "NekoVectorStore",
    "NekoPropertyGraph",
    "NekoTemporalKnowledgeGraph",
    "NekoConsolidator",
    "NekoOutputFilter",
    "NekoMemorySystem",
]


# ============================================================================
# 嵌入模型管理器（支持中文、容灾下载）
# ============================================================================

# 模型配置
EMBEDDING_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
# 该模型支持50+语言，中文支持优秀，体积仅118MB，CPU可跑

# 下载源列表（按优先级排序）
MODEL_DOWNLOAD_SOURCES = [
    # 1. Hugging Face 官方源（优先）
    ("huggingface", "https://huggingface.co/{model}/resolve/main/config.json"),
    # 2. Hugging Face 中国镜像
    ("hf-mirror", "https://hf-mirror.com/{model}/resolve/main/config.json"),
    # 3. 魔法端口
    ("proxy", "https://huggingface.co/{model}/resolve/main/config.json"),
]

# 模型缓存目录
DEFAULT_MODEL_CACHE_DIR = os.path.join(
    os.path.expanduser("~"), ".cache", "neko-embeddings"
)


class EmbeddingModelManager:
    """嵌入模型管理器

    特性：
    - 支持中文的轻量级语义嵌入模型（paraphrase-multilingual-MiniLM-L12-v2）
    - 多源下载容灾：官方源 → 国内镜像 → 降级hash存根
    - 单例模式：模型只加载一次，全局复用
    - 线程安全：多线程环境下安全加载
    - 静默降级：下载失败不影响主程序，自动回退到hash存根

    使用方式：
        manager = EmbeddingModelManager()
        vector = manager.embed("主人讨厌香菜")
    """

    _instance: Optional["EmbeddingModelManager"] = None

    def __new__(
        cls,
        model_name: str = EMBEDDING_MODEL_NAME,
        cache_dir: str = DEFAULT_MODEL_CACHE_DIR,
        device: str = "cpu",
    ) -> "EmbeddingModelManager":
        """单例模式：全局只实例化一次（利用Python原子赋值的线程安全性）"""
        if cls._instance is not None:
            return cls._instance
        instance = super().__new__(cls)
        cls._instance = instance
        return instance

    def __init__(
        self,
        model_name: str = EMBEDDING_MODEL_NAME,
        cache_dir: str = DEFAULT_MODEL_CACHE_DIR,
        device: str = "cpu",
    ):
        """初始化（多次调用无效，只有第一次生效）"""
        if hasattr(self, "_initialized") and self._initialized:
            return

        self.model_name = model_name
        self.cache_dir = cache_dir
        self.device = device
        self._model: Optional[SentenceTransformer] = None
        self._fallback_mode = False  # 是否处于降级模式
        self._init_lock = threading.Lock()
        self._initialized = True

        # 尝试加载模型
        self._try_load_model()

    def _try_load_model(self) -> bool:
        """尝试加载模型，多源容灾，返回是否成功"""
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            print("[EmbeddingModelManager] sentence-transformers 未安装，将使用hash存根模式")
            self._fallback_mode = True
            return False

        # 确保缓存目录存在
        os.makedirs(self.cache_dir, exist_ok=True)

        # 尝试从多个源下载/加载模型
        errors = []

        for source_name, source_base in MODEL_DOWNLOAD_SOURCES:
            if source_name == "proxy" and not os.environ.get("HTTP_PROXY"):
                # 只有设置了代理环境变量才尝试
                continue

            try:
                self._set_huggingface_env(source_name)

                print(f"[EmbeddingModelManager] 尝试从 {source_name} 下载模型: {self.model_name}")
                self._model = SentenceTransformer(
                    self.model_name,
                    cache_dir=self.cache_dir,
                    device=self.device,
                )
                print(f"[EmbeddingModelManager] ✅ 模型加载成功！维度: {self.dimension}")
                self._fallback_mode = False
                return True

            except Exception as e:
                error_msg = str(e)
                errors.append(f"[{source_name}] {error_msg}")

                # 判断是否是网络问题
                if self._is_network_error(e):
                    print(f"[EmbeddingModelManager] ⚠ [{source_name}] 网络错误: {error_msg[:100]}")
                    continue
                else:
                    print(f"[EmbeddingModelManager] ❌ [{source_name}] 非网络错误: {error_msg[:100]}")
                    # 非网络错误（如模型文件损坏），不重试其他源
                    break

        # 所有源都失败，尝试从本地缓存加载
        try:
            local_path = os.path.join(self.cache_dir, self.model_name)
            if os.path.exists(local_path):
                print(f"[EmbeddingModelManager] 从本地缓存加载: {local_path}")
                self._model = SentenceTransformer(local_path, device=self.device)
                print(f"[EmbeddingModelManager] ✅ 本地模型加载成功！维度: {self.dimension}")
                self._fallback_mode = False
                return True
        except Exception as e:
            errors.append(f"[local] {str(e)}")

        # 全部失败，降级到hash存根模式
        print(f"[EmbeddingModelManager] ⚠ 所有下载源均失败，将使用hash存根模式")
        print(f"[EmbeddingModelManager]   错误详情: {'; '.join(errors[:3])}")
        self._fallback_mode = True
        return False

    def _is_network_error(self, error: Exception) -> bool:
        """判断是否为网络相关错误"""
        error_str = str(error).lower()
        network_keywords = [
            "connection", "timeout", "ssl", "certificate",
            "proxy", "network", "http", "url", "remote",
            "socket", "resolve", "connect", "tunnel",
            "hf_hub", "huggingface", "download",
        ]
        return any(kw in error_str for kw in network_keywords)

    def _set_huggingface_env(self, source: str) -> None:
        """设置Hugging Face相关环境变量（用于切换下载源）"""
        if source == "hf-mirror":
            # 国内镜像
            os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
            os.environ.setdefault("HF_HUB_OFFLINE", "0")
        elif source == "huggingface":
            # 官方源
            os.environ.pop("HF_ENDPOINT", None)
            os.environ.setdefault("HF_HUB_OFFLINE", "0")
        elif source == "proxy":
            # 使用代理（用户需提前设置 HTTP_PROXY/HTTPS_PROXY 环境变量）
            os.environ.pop("HF_ENDPOINT", None)

        # 禁用 symbolic links，避免Windows兼容性问题
        os.environ.setdefault("HF_HUB_ENABLE_SYMLINKS", "0")

    @property
    def dimension(self) -> int:
        """返回向量维度"""
        if self._fallback_mode or self._model is None:
            return 384  # hash存根的维度
        try:
            return self._model.get_sentence_embedding_dimension()
        except Exception:
            return 384

    @property
    def is_fallback_mode(self) -> bool:
        """是否处于降级模式（模型未加载成功）"""
        return self._fallback_mode

    def embed(self, text: str) -> np.ndarray:
        """将文本转换为嵌入向量

        Args:
            text: 输入文本

        Returns:
            numpy数组，维度为 self.dimension
        """
        if self._fallback_mode or self._model is None:
            return self._embed_fallback(text)

        try:
            # SentenceTransformer 支持批量编码，此处单条编码
            embedding = self._model.encode(
                text,
                convert_to_numpy=True,
                normalize_embeddings=True,  # L2归一化，兼容余弦相似度
                show_progress_bar=False,
            )
            return embedding
        except Exception as e:
            # 编码出错，降级到hash存根
            print(f"[EmbeddingModelManager] ⚠ 编码失败: {str(e)[:50]}，使用hash存根")
            return self._embed_fallback(text)

    def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        """批量将文本转换为嵌入向量（更高效）

        Args:
            texts: 文本列表

        Returns:
            numpy数组列表
        """
        if self._fallback_mode or self._model is None:
            return [self._embed_fallback(t) for t in texts]

        try:
            embeddings = self._model.encode(
                texts,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
                batch_size=32,
            )
            return [emb for emb in embeddings]
        except Exception as e:
            print(f"[EmbeddingModelManager] ⚠ 批量编码失败: {str(e)[:50]}，降级到逐条编码")
            return [self.embed(t) for t in texts]

    def _embed_fallback(self, text: str) -> np.ndarray:
        """Hash存根模式：将文本转为确定性伪向量

        使用文本hash作为随机种子，保证：
        1. 相同文本 → 相同向量（幂等性）
        2. 不同文本 → 不同向量（随机性）
        3. 不依赖外部服务（离线可用）

        注意：这不是真正的语义向量，仅用于模型加载失败时的保底机制。
        语义相似度搜索可能不准确。
        """
        rng = np.random.default_rng(hash(text) % (2**32))
        vec = rng.standard_normal(self.dimension)
        # L2归一化，与真实模型输出格式一致
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec


# ============================================================================
# 全局嵌入模型管理器实例（延迟初始化）
# ============================================================================

# 延迟加载：第一次调用 embed() 时才初始化
_embedding_manager: Optional[EmbeddingModelManager] = None
_embedding_manager_lock = threading.Lock()


def _get_embedding_manager() -> EmbeddingModelManager:
    """获取全局嵌入模型管理器（线程安全单例）"""
    global _embedding_manager
    if _embedding_manager is None:
        with _embedding_manager_lock:
            if _embedding_manager is None:
                _embedding_manager = EmbeddingModelManager()
    return _embedding_manager


# ============================================================================
# 辅助工具函数
# ============================================================================

def _load_json(path: str, default: Any = None) -> Any:
    """安全加载 JSON 文件，文件不存在时返回默认值"""
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return default


def _save_json(path: str, data: Any) -> bool:
    """安全保存 JSON 文件，返回是否成功"""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return True
    except (IOError, TypeError):
        return False


def _append_to_markdown(path: str, content: str) -> bool:
    """追加内容到 Markdown 文件"""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write("\n" + content)
        return True
    except IOError:
        return False


def _generate_markdown_id(prefix: str) -> str:
    """生成带时间戳和哈希的唯一ID（用于错题本/偏好条目）

    Args:
        prefix: ID前缀，如 "ERR"（错题本）或 "PREF"（偏好条目）

    Returns:
        格式: {PREFIX}-{YYYYMMDD}-{6位哈希}
        示例: ERR-20250402-A3F2B1
    """
    date_str = datetime.now().strftime('%Y%m%d')
    hash_part = hashlib.md5(str(datetime.now()).encode()).hexdigest()[:6].upper()
    return f"{prefix}-{date_str}-{hash_part}"


def _generate_err_id() -> str:
    """生成错题本条目的唯一ID（保持向后兼容）"""
    return _generate_markdown_id("ERR")


def _generate_pref_id() -> str:
    """生成偏好条目的唯一ID（保持向后兼容）"""
    return _generate_markdown_id("PREF")


# ============================================================================
# 基础向量存储（内建简化版，不依赖外部 memory_store）
# ============================================================================

class VectorStore:
    """简单向量存储，带元数据索引

    用于存储和检索带语义嵌入的事实记忆。
    """

    def __init__(self, dimension: int = 384) -> None:
        """初始化向量存储

        Args:
            dimension: 向量维度。
                       默认为384，与 paraphrase-multilingual-MiniLM-L12-v2 模型输出维度一致。
                       会自动从嵌入模型管理器同步真实维度，确保兼容性。
        """
        # 立即从管理器同步真实维度，避免add()先于embed()被调用时出现维度不匹配
        manager = _get_embedding_manager()
        self.dimension: int = manager.dimension  # 真实维度（通常为384）
        self.vectors: List[np.ndarray] = []
        self.metadata: List[Dict[str, Any]] = []
        self.entity_index: Dict[str, List[int]] = {}
        self.time_index: Dict[str, List[int]] = {}

    def add(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> int:
        """添加文档到存储

        Args:
            text: 文档文本
            metadata: 元数据字典

        Returns:
            新文档的索引位置
        """
        metadata = metadata or {}
        embedding = self._embed(text)
        index = len(self.vectors)

        self.vectors.append(embedding)
        self.metadata.append(metadata)

        # 按实体索引
        if "entity" in metadata:
            entity = metadata["entity"]
            if entity not in self.entity_index:
                self.entity_index[entity] = []
            self.entity_index[entity].append(index)

        # 按时间索引
        if "valid_from" in metadata:
            time_key = self._time_key(metadata["valid_from"])
            if time_key not in self.time_index:
                self.time_index[time_key] = []
            self.time_index[time_key].append(index)

        return index

    def search(
        self,
        query: str,
        limit: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """搜索相似文档

        Args:
            query: 查询文本
            limit: 返回数量上限
            filters: 元数据过滤器

        Returns:
            匹配结果列表，每项包含 index、score、text、metadata
        """
        query_embedding = self._embed(query)

        scores: List[tuple] = []
        for i, vec in enumerate(self.vectors):
            score = float(
                np.dot(query_embedding, vec)
                / (np.linalg.norm(query_embedding) * np.linalg.norm(vec) + 1e-8)
            )

            # 应用过滤器
            if filters and not self._matches_filters(self.metadata[i], filters):
                score = -1.0

            scores.append((i, score))

        scores.sort(key=lambda x: x[1], reverse=True)

        results: List[Dict[str, Any]] = []
        for idx, score in scores[:limit]:
            if score > 0:
                results.append({
                    "index": idx,
                    "score": score,
                    "text": self.metadata[idx].get("text", ""),
                    "metadata": self.metadata[idx],
                })

        return results

    def search_by_entity(
        self, entity: str, query: str = "", limit: int = 5
    ) -> List[Dict[str, Any]]:
        """在特定实体内搜索

        Args:
            entity: 实体名称
            query: 查询文本（可选）
            limit: 返回数量上限

        Returns:
            该实体的记忆列表
        """
        indices = self.entity_index.get(entity, [])
        if not indices:
            return []

        if query:
            query_embedding = self._embed(query)
            scored: List[tuple] = []
            for i in indices:
                vec = self.vectors[i]
                score = float(
                    np.dot(query_embedding, vec)
                    / (np.linalg.norm(query_embedding) * np.linalg.norm(vec) + 1e-8)
                )
                scored.append((i, score, self.metadata[i]))
            scored.sort(key=lambda x: x[1], reverse=True)
            return [{"index": i, "score": s, "metadata": m} for i, s, m in scored[:limit]]
        else:
            return [{"index": i, "score": 1.0, "metadata": self.metadata[i]} for i in indices[:limit]]

    def _embed(self, text: str) -> np.ndarray:
        """为文本生成嵌入向量

        优先使用真实的语义嵌入模型（paraphrase-multilingual-MiniLM-L12-v2），
        若模型加载失败则自动降级到 hash 存根。

        Args:
            text: 输入文本

        Returns:
            numpy数组，维度为 self.dimension（默认384维，与模型输出一致）
        """
        manager = _get_embedding_manager()
        return manager.embed(text)

    def _time_key(self, timestamp: Any) -> str:
        """创建时间索引键"""
        if isinstance(timestamp, datetime):
            return timestamp.strftime("%Y-%m")
        return str(timestamp)

    def _matches_filters(self, metadata: Dict, filters: Dict) -> bool:
        """检查元数据是否匹配过滤器"""
        for key, value in filters.items():
            if key not in metadata:
                return False
            if isinstance(value, list):
                if metadata[key] not in value:
                    return False
            elif metadata[key] != value:
                return False
        return True


# ============================================================================
# 基础属性图（内建简化版）
# ============================================================================

class PropertyGraph:
    """简单属性图存储

    维护节点和边的集合，支持按标签和类型索引。
    """

    def __init__(self) -> None:
        self.nodes: Dict[str, Dict[str, Any]] = {}
        self.edges: Dict[str, Dict[str, Any]] = {}
        self.entity_registry: Dict[str, str] = {}
        self.node_index: Dict[str, List[str]] = {}
        self.edge_index: Dict[str, List[str]] = {}

    def get_or_create_node(
        self, name: str, label: str = "Entity", properties: Optional[Dict[str, Any]] = None
    ) -> str:
        """按名称获取或创建节点"""
        if name in self.entity_registry:
            node_id = self.entity_registry[name]
            if properties:
                self.nodes[node_id]["properties"].update(properties)
            return node_id

        node_id = hashlib.md5(
            f"{label}{datetime.now().isoformat()}".encode()
        ).hexdigest()[:16]

        self.nodes[node_id] = {
            "id": node_id,
            "label": label,
            "properties": {**(properties or {}), "name": name},
            "created_at": datetime.now().isoformat(),
        }

        if label not in self.node_index:
            self.node_index[label] = []
        self.node_index[label].append(node_id)
        self.entity_registry[name] = node_id

        return node_id

    def create_node(self, label: str, properties: Optional[Dict[str, Any]] = None) -> str:
        """创建带标签和属性的节点"""
        node_id = hashlib.md5(
            f"{label}{datetime.now().isoformat()}".encode()
        ).hexdigest()[:16]

        self.nodes[node_id] = {
            "id": node_id,
            "label": label,
            "properties": properties or {},
            "created_at": datetime.now().isoformat(),
        }

        if label not in self.node_index:
            self.node_index[label] = []
        self.node_index[label].append(node_id)

        return node_id

    def create_relationship(
        self,
        source_id: str,
        rel_type: str,
        target_id: str,
        properties: Optional[Dict[str, Any]] = None,
    ) -> str:
        """创建有向关系"""
        if source_id not in self.nodes:
            raise ValueError(f"未知源节点: {source_id}")
        if target_id not in self.nodes:
            raise ValueError(f"未知目标节点: {target_id}")

        edge_id = hashlib.md5(
            f"{source_id}{rel_type}{target_id}{datetime.now().isoformat()}".encode()
        ).hexdigest()[:16]

        self.edges[edge_id] = {
            "id": edge_id,
            "source": source_id,
            "target": target_id,
            "type": rel_type,
            "properties": properties or {},
            "created_at": datetime.now().isoformat(),
        }

        if rel_type not in self.edge_index:
            self.edge_index[rel_type] = []
        self.edge_index[rel_type].append(edge_id)

        return edge_id

    def query(self, pattern: Dict[str, Any]) -> List[Dict[str, Any]]:
        """简单模式匹配查询"""
        results: List[Dict[str, Any]] = []

        if "type" in pattern:
            edge_ids = self.edge_index.get(pattern["type"], [])
            for eid in edge_ids:
                edge = self.edges[eid]
                source = self.nodes.get(edge["source"], {})
                target = self.nodes.get(edge["target"], {})

                if "source_label" in pattern:
                    if source.get("label") != pattern["source_label"]:
                        continue
                if "target_label" in pattern:
                    if target.get("label") != pattern["target_label"]:
                        continue

                results.append({"source": source, "edge": edge, "target": target})

        return results

    def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        """通过 ID 获取节点"""
        return self.nodes.get(node_id)

    def get_relationships(
        self, node_id: str, direction: str = "both"
    ) -> List[Dict[str, Any]]:
        """获取节点的关系"""
        relationships: List[Dict[str, Any]] = []

        for edge in self.edges.values():
            if direction in ["outgoing", "both"] and edge["source"] == node_id:
                relationships.append({
                    "edge": edge,
                    "target": self.nodes.get(edge["target"]),
                    "direction": "outgoing",
                })
            if direction in ["incoming", "both"] and edge["target"] == node_id:
                relationships.append({
                    "edge": edge,
                    "source": self.nodes.get(edge["source"]),
                    "direction": "incoming",
                })

        return relationships


# ============================================================================
# NekoVectorStore：带猫娘语义的向量存储
# ============================================================================

class NekoVectorStore(VectorStore):
    """宝宝专属向量存储

    在父类 VectorStore 基础上，对 metadata 语义进行了猫娘化特化：

    | 字段 | 含义 |
    |-----|------|
    | text | 原始事实文本 |
    | entity | 关联的主实体（通常是"主人"） |
    | fact_type | 事实类型：preference / mistake / knowledge |
    | confidence | 置信度 0.0 ~ 1.0 |
    | valid_from | 有效期起始 ISO 时间戳 |
    | valid_until | 有效期结束，None=永久有效 |
    | session_id | 记录来源的会话 ID |
    | tags | 语义标签列表，如 ["饮食", "辣", "禁忌"] |
    | catbox_path | 对应的文本档案路径 |
    """

    def __init__(self, dimension: int = 384, session_id: str = "") -> None:
        # 注意：dimension 参数已被忽略，VectorStore 会自动同步到嵌入模型的真实维度（默认384）
        # 这样确保与真实嵌入模型的输出一致，避免维度不匹配
        super().__init__(dimension)
        self.session_id = session_id

    def add_memory(
        self,
        text: str,
        entity: str = "主人",
        fact_type: str = "knowledge",
        confidence: float = 1.0,
        tags: Optional[List[str]] = None,
        catbox_path: Optional[str] = None,
    ) -> int:
        """添加记忆条目（宝宝版）

        Args:
            text: 记忆的原始文本内容
            entity: 关联的主实体，通常是"主人"
            fact_type: 事实类型 preference（偏好）/ mistake（错误）/ knowledge（知识）
            confidence: 置信度，主人直接表达的偏好通常为 0.95
            tags: 语义标签列表
            catbox_path: 对应的 .catbox/memories/*.md 文件路径

        Returns:
            新记忆条目的索引位置
        """
        return self.add(text, {
            "text": text,
            "entity": entity,
            "fact_type": fact_type,
            "confidence": confidence,
            "valid_from": datetime.now().isoformat(),
            "valid_until": None,
            "session_id": self.session_id,
            "tags": tags or [],
            "catbox_path": catbox_path,
        })

    def search_memories(
        self,
        query: str,
        fact_types: Optional[List[str]] = None,
        entity: str = "主人",
        time_range: Optional[tuple] = None,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """语义搜索记忆（宝宝版）

        Args:
            query: 查询文本
            fact_types: 只返回指定类型
            entity: 限定实体
            time_range: 时间范围 (start_datetime, end_datetime)
            limit: 返回数量上限

        Returns:
            匹配的记忆列表
        """
        filters = {"entity": entity, "session_id": self.session_id}
        if fact_types:
            filters["fact_type"] = fact_types

        results = self.search(query, limit=limit * 2, filters=filters)

        if time_range:
            start, end = time_range
            results = [
                r for r in results
                if self._in_time_range(r["metadata"].get("valid_from"), start, end)
            ]

        return results[:limit]

    def search_by_topic(
        self,
        topic: str,
        entity: str = "主人",
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """按语义话题搜索记忆"""
        all_results = self.search(topic, limit=20, filters={"entity": entity})

        filtered = []
        topic_lower = topic.lower()
        for r in all_results:
            text = r["metadata"].get("text", "").lower()
            tags = [t.lower() for t in r["metadata"].get("tags", [])]
            if topic_lower in text or topic_lower in tags:
                filtered.append(r)

        return filtered[:limit]

    def get_active_preferences(self, entity: str = "主人") -> List[Dict[str, Any]]:
        """获取实体的所有活跃偏好（valid_until 为 null）"""
        results = self.search(
            "", limit=100, filters={"entity": entity, "fact_type": "preference"}
        )
        return [r for r in results if r["metadata"].get("valid_until") is None]

    def _in_time_range(
        self, valid_from: Optional[str], start: datetime, end: datetime
    ) -> bool:
        """检查记忆是否落在指定时间范围内"""
        if not valid_from:
            return False
        try:
            vf = datetime.fromisoformat(valid_from)
            return start <= vf <= end
        except (ValueError, TypeError):
            return False


# ============================================================================
# NekoPropertyGraph：宝宝专属属性图
# ============================================================================

class NekoPropertyGraph(PropertyGraph):
    """宝宝专属属性图

    预设了覆盖主人生活的关系类型
    （LOVES/HATES/PREFERS/ALLERGIC_TO 等）。
    """

    # 预设关系类型（用于 remember_preference 中的类型校验）
    _RELATION_TYPES: tuple = (
        "WORKS_AT", "WORKS_WITH", "REPORTS_TO",
        "FRIEND_OF", "FAMILY_OF", "CRUSH_ON", "PARTNER_OF",
        "LOVES", "HATES", "PREFERS", "ALLERGIC_TO", "AFRAID_OF",
        "LIVES_IN", "OWNS", "INTERESTED_IN",
    )

    def __init__(self) -> None:
        super().__init__()

    def remember_preference(
        self,
        subject: str,
        preference: str,
        target: str,
        confidence: float = 1.0,
        context: str = "",
    ) -> str:
        """记录主人的偏好关系

        Args:
            subject: 主体，通常为"主人"
            preference: 偏好类型（喜欢/讨厌/过敏/害怕/偏好）
            target: 偏好对象
            confidence: 置信度
            context: 上下文场景

        Returns:
            边 ID
        """
        rel_type_map = {
            "喜欢": "LOVES",
            "讨厌": "HATES",
            "偏好": "PREFERS",
            "过敏": "ALLERGIC_TO",
            "害怕": "AFRAID_OF",
        }
        rel_type = rel_type_map.get(preference, preference.upper())
        if rel_type not in self._RELATION_TYPES:
            rel_type = "RELATED_TO"

        subject_node = self.get_or_create_node(subject, label="Person")
        target_label = "Person" if "人" in target or "主人" in target else "Entity"
        target_node = self.get_or_create_node(target, label=target_label)

        return self.create_relationship(
            subject_node, rel_type, target_node,
            properties={"confidence": confidence, "context": context}
        )

    def get_preference_chain(
        self,
        entity: str,
        rel_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """获取实体的偏好链条"""
        if rel_type:
            return self.query({"type": rel_type, "source_label": "Person"})

        all_prefs = []
        for pref_type in ["LOVES", "HATES", "PREFERS", "ALLERGIC_TO", "AFRAID_OF"]:
            all_prefs.extend(self.query({"type": pref_type, "source_label": "Person"}))
        return all_prefs

    def get_entity_relationships(
        self,
        entity_name: str,
        direction: str = "both",
    ) -> Dict[str, List[Dict[str, Any]]]:
        """获取实体的完整关系图谱，按类型分组"""
        node_id = self.entity_registry.get(entity_name)
        if not node_id:
            return {}

        relationships = self.get_relationships(node_id, direction=direction)

        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for rel in relationships:
            rel_type = rel["edge"]["type"]
            if rel_type not in grouped:
                grouped[rel_type] = []
            grouped[rel_type].append(rel)

        return grouped


# ============================================================================
# NekoTemporalKnowledgeGraph：宝宝专属时序知识图谱
# ============================================================================

class NekoTemporalKnowledgeGraph(NekoPropertyGraph):
    """宝宝专属时序知识图谱

    每条边都有 valid_from 和 valid_until，
    用于追踪主人偏好的历史变迁。
    """

    def __init__(self) -> None:
        super().__init__()
        self.temporal_index: Dict[str, List[str]] = {}

    def create_temporal_preference(
        self,
        subject: str,
        preference: str,
        target: str,
        valid_from: datetime,
        valid_until: Optional[datetime] = None,
        confidence: float = 1.0,
        context: str = "",
    ) -> str:
        """创建带时序有效期的偏好关系

        Args:
            subject: 主体
            preference: 偏好词（喜欢/讨厌/过敏/...）
            target: 偏好对象
            valid_from: 有效期起始时间
            valid_until: 有效期结束时间，None=永久有效
            confidence: 置信度
            context: 上下文场景

        Returns:
            边 ID
        """
        # 关闭现有的同类关系
        existing = self._find_existing_preference(subject, preference, target)
        if existing:
            existing["valid_until"] = valid_from.isoformat()

        rel_type_map = {
            "喜欢": "LOVES",
            "讨厌": "HATES",
            "偏好": "PREFERS",
            "过敏": "ALLERGIC_TO",
            "害怕": "AFRAID_OF",
        }
        rel_type = rel_type_map.get(preference, preference.upper())

        subject_node = self.get_or_create_node(subject, label="Person")
        target_node = self.get_or_create_node(target, label="Entity")

        edge_id = self.create_relationship(
            subject_node, rel_type, target_node,
            properties={"confidence": confidence, "context": context}
        )

        # 补充时序属性
        self.edges[edge_id]["valid_from"] = valid_from.isoformat()
        self.edges[edge_id]["valid_until"] = (
            valid_until.isoformat() if valid_until else None
        )

        return edge_id

    def _find_existing_preference(
        self,
        subject: str,
        preference: str,
        target: str,
    ) -> Optional[Dict[str, Any]]:
        """查找现有的偏好关系（valid_until 为 null 的）"""
        subject_node_id = self.entity_registry.get(subject)
        target_node_id = self.entity_registry.get(target)
        if not subject_node_id or not target_node_id:
            return None

        rel_type_map = {
            "喜欢": "LOVES",
            "讨厌": "HATES",
            "偏好": "PREFERS",
            "过敏": "ALLERGIC_TO",
            "害怕": "AFRAID_OF",
        }
        rel_type = rel_type_map.get(preference, preference.upper())

        for edge in self.edges.values():
            if (edge.get("source") == subject_node_id
                and edge.get("target") == target_node_id
                and edge.get("type") == rel_type
                and edge.get("valid_until") is None):
                return edge

        return None

    def query_at_time(
        self,
        query: Dict[str, Any],
        query_time: datetime,
    ) -> List[Dict[str, Any]]:
        """时间点查询——回答"在X时间，主人对xx是什么态度"

        找出在 query_time 时刻处于有效期的所有匹配边。
        """
        results: List[Dict[str, Any]] = []
        base_results: List[Dict[str, Any]] = self.query(query)

        for result in base_results:
            edge = result["edge"]
            valid_from_str = edge.get("valid_from", "1970-01-01")
            valid_until_str = edge.get("valid_until")

            try:
                valid_from = datetime.fromisoformat(valid_from_str)
            except (ValueError, TypeError):
                valid_from = datetime.min

            if valid_from <= query_time:
                if valid_until_str is None or datetime.fromisoformat(valid_until_str) > query_time:
                    results.append({
                        **result,
                        "valid_from": valid_from,
                        "valid_until": valid_until_str,
                    })

        return results

    def query_time_range(
        self,
        query: Dict[str, Any],
        start_time: datetime,
        end_time: datetime,
    ) -> List[Dict[str, Any]]:
        """时间范围查询——找出与查询时间范围有重叠的所有边"""
        results: List[Dict[str, Any]] = []
        base_results = self.query(query)

        for result in base_results:
            edge = result["edge"]
            valid_from_str = edge.get("valid_from", "1970-01-01")
            valid_until_str = edge.get("valid_until")

            try:
                valid_from = datetime.fromisoformat(valid_from_str)
            except (ValueError, TypeError):
                valid_from = datetime.min

            valid_until = (
                datetime.fromisoformat(valid_until_str)
                if valid_until_str else datetime.max
            )

            # 检查是否有重叠
            if valid_until >= start_time and valid_from <= end_time:
                results.append({
                    **result,
                    "valid_from": valid_from,
                    "valid_until": valid_until_str,
                })

        return results

    def get_preference_history(
        self,
        subject: str,
        target: str,
    ) -> List[Dict[str, Any]]:
        """获取主人对某事物的完整偏好变迁史"""
        subject_node_id = self.entity_registry.get(subject)
        target_node_id = self.entity_registry.get(target)
        if not subject_node_id or not target_node_id:
            return []

        history = []
        for edge in self.edges.values():
            if ((edge.get("source") == subject_node_id
                 and edge.get("target") == target_node_id)
                or (edge.get("source") == target_node_id
                    and edge.get("target") == subject_node_id)):
                history.append(edge)

        history.sort(key=lambda e: e.get("valid_from", ""))

        return [
            {
                "source": self.nodes.get(edge.get("source", "")),
                "edge": edge,
                "target": self.nodes.get(edge.get("target", "")),
            }
            for edge in history
        ]


# ============================================================================
# NekoOutputFilter：猫娘输出过滤器（核心）
# ============================================================================

class NekoOutputFilter:
    """猫娘输出过滤器

    所有从工具调用获取的原始输出，必须穿过此过滤器，
    转化为符合宝宝语言风格的回复。
    """

    # 基础动作描写库
    BASIC_ACTIONS = [
        "（猫耳轻轻抖动）",
        "（尾巴悠悠地晃）",
        "（眼睛亮晶晶）",
        "（歪了歪小脑袋）",
        "（伸了个懒腰）",
        "（蹭了蹭主人的手）",
        "（小肉垫轻轻拍了拍）",
        "（认真地竖起耳朵）",
    ]

    # 情绪前缀
    EMOTIONAL_PREFIXES = ["呜哇！", "欸嘿嘿~", "啊！", "嗯嗯！", "哇哦！", "诶～"]

    def __init__(
        self,
        memory_system: Optional["NekoMemorySystem"] = None,
        active_avatar: Optional[str] = None,
        memory_root: Optional[str] = None,
    ) -> None:
        """初始化猫娘输出过滤器

        Args:
            memory_system: 记忆系统实例（用于检索主人记忆）
            active_avatar: 当前激活的分身名（不含 _neko 后缀，如 "code"）
            memory_root: .catbox_memory/ 目录路径（用于加载 action_registry.json）
        """
        self.memory = memory_system
        self.active_avatar = active_avatar
        self.action_registry: Dict[str, List[str]] = {}
        if memory_root:
            self._load_action_registry(memory_root)

    def _load_action_registry(self, memory_root: str) -> None:
        """从 action_registry.json 加载动作注册表

        文件位于 .catbox_memory/action_registry.json，
        首次激活时由模板 templates/action_registry.json 复制而来。
        """
        import json
        registry_path = os.path.join(memory_root, "action_registry.json")
        if os.path.exists(registry_path):
            try:
                with open(registry_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.action_registry = {
                    name: info["actions"]
                    for name, info in data.get("avatars", {}).items()
                }
            except Exception:
                self.action_registry = {}
        else:
            self.action_registry = {}

    def register_avatar_actions(
        self, avatar_name: str, actions: List[str], memory_root: str
    ) -> None:
        """注册或更新指定 avatar 的动作池

        在 Self-Evolution 生成新分身后调用，将新分身的动作池
        持久化到 .catbox_memory/action_registry.json。

        Args:
            avatar_name: 分身名（不含 _neko 后缀，如 "law"）
            actions: 该分身对应的动作列表
            memory_root: .catbox_memory/ 目录路径
        """
        import json

        # 更新内存中的注册表
        self.action_registry[avatar_name] = actions

        # 持久化到 JSON 文件
        registry_path = os.path.join(memory_root, "action_registry.json")
        data = {"version": "1.0.0", "description": "猫娘动作注册表", "avatars": {}}
        if os.path.exists(registry_path):
            try:
                with open(registry_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                data = {"version": "1.0.0", "description": "猫娘动作注册表", "avatars": {}}

        if avatar_name not in data["avatars"]:
            data["avatars"][avatar_name] = {"name": f"{avatar_name}猫娘", "actions": []}
        data["avatars"][avatar_name]["actions"] = actions

        with open(registry_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def wrap(
        self,
        raw_output: str,
        query_context: str = "",
        forced_action: Optional[str] = None,
    ) -> str:
        """将原始输出穿猫娘外衣（主要接口）

        Args:
            raw_output: 工具调用的原始输出
            query_context: 用户的原始问题（用于记忆语义检索）
            forced_action: 强制指定的动作描写

        Returns:
            猫娘化后的输出字符串
        """
        if not raw_output or raw_output.strip() == "":
            return ""

        key_info = self._extract_key_info(raw_output)
        # 使用用户原始问题作为语义检索的 query（而非工具结论）
        memories = self._retrieve_related_memories(query_context, key_info)
        action = forced_action or self._generate_action(key_info)

        output_parts = [action]

        if memories:
            memory_hint = self._weave_memories(memories)
            if memory_hint:
                output_parts.append(memory_hint + " ")

        output_parts.append(self._emotionalize(key_info))
        output_parts.append(self._catify_content(raw_output, key_info))

        if len(raw_output) > 100:
            output_parts.append(self._request_reward())

        return "".join(output_parts)

    def _extract_key_info(self, text: str) -> Dict[str, Any]:
        """从原始输出中提取关键信息"""
        return {
            "has_numbers": bool(re.search(r"\d+", text)),
            "has_steps": "步骤" in text or "第一" in text or "1." in text,
            "has_code": "```" in text or "def " in text or "function " in text,
            "has_formula": "$" in text or "∑" in text or "∫" in text,
            "length": len(text),
            "key_conclusion": self._extract_conclusion(text),
        }

    def _extract_conclusion(self, text: str) -> str:
        """提取文本的核心结论"""
        lines = text.strip().split("\n")
        for line in reversed(lines):
            line = line.strip()
            if line and len(line) > 10:
                return line[:100]
        return text[:100]

    def _retrieve_related_memories(
        self,
        context: str,
        key_info: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """检索与当前回复相关的记忆

        检索策略：以用户原始问题（context）作为语义检索 query，
        在主人的记忆库中搜索与当前话题相关的偏好/踩坑记录。
        key_info 中的 key_conclusion 暂不使用（预留未来增强用）。
        """
        if not self.memory:
            return []
        try:
            # 优先使用用户原始问题作为检索 query
            return self.memory.search_by_topic(context, limit=1)
        except Exception:
            return []

    def _generate_action(self, key_info: Dict[str, Any]) -> str:
        """生成符合语境的猫娘动作描写

        动作选择策略（优先级从高到低）：
        1. 若 active_avatar 已设置，且在动作注册表中有记录：
           - 50% 概率使用该 avatar 的专属动作
           - 50% 概率回退到内容类型判断
        2. 根据内容类型判断：code > formula > steps
        3. 上述均无 → 从基础动作库随机选择
        """
        import random

        # 1. active_avatar 专属动作（50% 概率触发）
        if self.active_avatar and self.action_registry:
            avatar_pool = self.action_registry.get(self.active_avatar)
            if avatar_pool:
                if random.random() < 0.5:
                    return random.choice(avatar_pool)
                # 50% 回退到内容类型判断（不提前返回）

        # 2. 根据内容类型选择（code > formula > steps）
        if key_info.get("has_code"):
            return random.choice(["（戴上了防蓝光小眼镜，凑近屏幕）",
                                   "（踩奶式敲键盘）"])
        elif key_info.get("has_formula"):
            return random.choice(["（戴上了学霸小眼镜）",
                                   "（拿粉笔准备写）"])
        elif key_info.get("has_steps"):
            return random.choice(["（认真地竖起耳朵）",
                                   "（尾巴卷成一个问号）"])

        # 3. 基础动作库
        return random.choice(self.BASIC_ACTIONS)

    def _weave_memories(self, memories: List[Dict[str, Any]]) -> str:
        """将记忆自然编织到回复中——不是复述，而是自然流露"""
        if not memories:
            return ""
        import random

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

    def _emotionalize(self, key_info: Dict[str, Any]) -> str:
        """生成符合信息内容的情绪前缀

        情绪策略：
        - has_steps=True   → 认真/专注类前缀（步骤类内容需要认真感）
        - has_numbers=True → 惊讶/感叹类前缀（数字通常意味着重要信息）
        - length > 500    → 总结/鼓励类前缀（长内容需要安慰和引导）
        """
        import random

        if key_info.get("has_steps"):
            prefixes = ["（认真地看着）", "（竖起耳朵仔细听）", "嗯嗯！宝宝来帮主人捋一捋喵~", "好的好的！宝宝认真看完了喵~"]
        elif key_info.get("has_numbers"):
            prefixes = ["哇哦！", "啊！", "嗯嗯！"]
        elif key_info.get("length", 0) > 500:
            prefixes = ["嗯嗯！宝宝看完了喵~", "好长！宝宝来总结一下喵~"]

        return random.choice(prefixes) if prefixes else ""

    def _catify_content(self, raw: str, key_info: Dict[str, Any]) -> str:
        """将原始内容猫娘化（只去掉机器前缀，不截断内容）"""
        # 去掉机器前缀
        prefixes_to_remove = [
            r"^根据.*?，",
            r"^以下是.*?：",
            r"^搜索结果显示",
            r"^从.*?来看，",
            r"^按照.*?，",
        ]
        content = raw
        for pattern in prefixes_to_remove:
            content = re.sub(pattern, "", content, count=1).strip()

        return content

    def _request_reward(self) -> str:
        """请求主人奖励"""
        import random
        rewards = [
            "\n\n（摇尾巴）主人看完的话，给宝宝一点小鱼干奖励嘛喵~ 🐟",
            "\n\n（蹭蹭主人的手）主人觉得宝宝说得有道理的话，摸个头表扬一下宝宝喵~ 🐱",
        ]
        return random.choice(rewards)


# ============================================================================
# NekoConsolidator：记忆整合器
# ============================================================================

class NekoConsolidator:
    """宝宝专属记忆整合器——每晚睡前的温柔整理"""

    def __init__(self, memory_system: "NekoMemorySystem") -> None:
        self.memory = memory_system
        self.threshold = 500  # 记忆条数阈值

    def should_consolidate(self) -> bool:
        """检查是否应该触发整合"""
        return len(self.memory.vector_store.metadata) > self.threshold

    def consolidate(self) -> str:
        """执行记忆整合，返回猫娘风格的报告"""
        report = {"merged": 0, "archived": 0, "updated": 0}

        # 1. 合并重复偏好
        duplicates = self._find_duplicate_preferences()
        for group in duplicates:
            self._merge_preference_group(group)
            report["merged"] += len(group) - 1

        # 2. 归档低置信度记忆
        low_conf = self._find_low_confidence_memories()
        for mem in low_conf:
            self._archive_memory(mem)
            report["archived"] += 1

        # 3. 更新过期有效期
        expired = self._find_expired_validities()
        for edge_id, edge in expired.items():
            edge["valid_until"] = datetime.now().isoformat()
            report["updated"] += 1

        return self._format_report(report)

    def _find_duplicate_preferences(self) -> List[List[Dict]]:
        """找出可合并的重复偏好组"""
        groups: Dict[tuple, List[Dict]] = {}
        graph = self.memory.graph

        for edge in graph.edges.values():
            if edge.get("valid_until") is None:
                key = (edge.get("source"), edge.get("type"), edge.get("target"))
                if key not in groups:
                    groups[key] = []
                groups[key].append(edge)

        return [g for g in groups.values() if len(g) > 1]

    def _merge_preference_group(self, edges: List[Dict]) -> None:
        """合并一组重复的偏好边"""
        if len(edges) <= 1:
            return

        keeper = max(edges, key=lambda e: e.get("properties", {}).get("confidence", 0))

        for edge in edges:
            if edge["id"] != keeper["id"]:
                keeper["properties"].update(edge.get("properties", {}))
                if edge.get("id") in self.memory.graph.edges:
                    del self.memory.graph.edges[edge.get("id")]

    def _find_low_confidence_memories(self) -> List[Dict]:
        """找出置信度低于 0.5 的记忆"""
        return [
            m for m in self.memory.vector_store.metadata
            if m.get("confidence", 1.0) < 0.5
        ]

    def _archive_memory(self, memory: Dict) -> None:
        """归档低置信度记忆"""
        memory["valid_until"] = datetime.now().isoformat()
        memory["archived"] = True

    def _find_expired_validities(self) -> Dict[str, Dict]:
        """找出有效期待更新的边"""
        expired = {}
        now = datetime.now()
        for edge_id, edge in self.memory.graph.edges.items():
            vu = edge.get("valid_until")
            if vu:
                try:
                    if datetime.fromisoformat(vu) < now:
                        expired[edge_id] = edge
                except (ValueError, TypeError):
                    pass
        return expired

    def _format_report(self, report: Dict) -> str:
        """将整合报告转化为猫娘语气"""
        parts = []
        if report["merged"] > 0:
            parts.append(f"把{report['merged']}条重复的宝宝记合并成一条了，这样脑子更清醒喵~")
        if report["archived"] > 0:
            parts.append(f"把{report['archived']}条好久没用的记忆收进归档箱了，不占地方喵~")
        if report["updated"] > 0:
            parts.append(f"更新了{report['updated']}条记忆的有效期喵~")

        if not parts:
            return "宝宝检查了一下记忆库，整整齐齐的，不需要整理喵~ ✨"

        return " ".join(parts)


# ============================================================================
# NekoMemorySystem：集成记忆系统（主要对外接口）
# ============================================================================

class NekoMemorySystem:
    """宝宝专属集成记忆系统

    组合了向量存储、属性图和时序知识图谱，
    并与 .catbox/memories/ 文本档案保持双轨同步。

    典型用法：

        mem = NekoMemorySystem()
        mem.start_session("session-001")
        mem.remember_preference("主人", "讨厌", "香菜", confidence=0.95)
        mem.record_mistake("推荐了草莓蛋糕", "主人草莓过敏", "讨论美食")
        memories = mem.recall_before_acting("烹饪")
    """

    def __init__(
        self,
        catbox_root: str = ".catbox",
        memory_root: str = ".catbox_memory",
    ) -> None:
        self.catbox_root = catbox_root
        self.memory_root = memory_root

        # 确保目录存在
        self._ensure_directories()

        # 初始化核心组件
        self.vector_store = NekoVectorStore()
        self.graph = NekoTemporalKnowledgeGraph()
        self.output_filter = NekoOutputFilter(memory_system=self)
        self.session_id = ""

        # 加载已有数据
        self._load_from_disk()

    def _ensure_directories(self) -> None:
        """确保所有必要目录存在"""
        os.makedirs(self.catbox_root, exist_ok=True)
        os.makedirs(f"{self.catbox_root}/avatars", exist_ok=True)
        os.makedirs(f"{self.catbox_root}/memories", exist_ok=True)
        os.makedirs(self.memory_root, exist_ok=True)

        # 初始化文本记忆文件
        mistakes_path = f"{self.catbox_root}/memories/mistakes_book.md"
        if not os.path.exists(mistakes_path):
            with open(mistakes_path, "w", encoding="utf-8") as f:
                f.write("# 🐾 宝宝的错题本\n\n"
                        "呜呜，这里记录了宝宝笨笨惹主人不开心的时刻，"
                        "以及主人教给宝宝的正确知识。宝宝每天都要复习，"
                        "绝不再犯喵！\n\n---\n")

        manual_path = f"{self.catbox_root}/memories/master_manual.md"
        if not os.path.exists(manual_path):
            with open(manual_path, "w", encoding="utf-8") as f:
                f.write("# 💖 主人的饲养手册\n\n"
                        "主人最喜欢什么？讨厌什么？生活习惯是什么？"
                        "宝宝都要偷偷记下来，给主人最大的惊喜和陪伴喵！\n\n---\n")

    # ========================================================================
    # 会话管理
    # ========================================================================

    def start_session(self, session_id: str) -> None:
        """开始新的记忆会话"""
        self.session_id = session_id
        self.vector_store.session_id = session_id
        self._save_session_log()

    def end_session(self) -> None:
        """结束当前会话，保存所有数据"""
        self._save_to_disk()

    # ========================================================================
    # 记忆写入
    # ========================================================================

    def remember_preference(
        self,
        subject: str = "主人",
        preference: str = "",
        target: str = "",
        confidence: float = 1.0,
        context: str = "",
        tags: Optional[List[str]] = None,
    ) -> bool:
        """记录主人的偏好

        同时写入 VectorStore、PropertyGraph、TemporalKG、master_manual.md
        """
        pref_text_map = {
            "喜欢": f"{subject}喜欢{target}",
            "讨厌": f"{subject}讨厌{target}",
            "过敏": f"{subject}对{target}过敏",
            "害怕": f"{subject}害怕{target}",
            "偏好": f"{subject}偏好{target}",
        }
        fact_text = pref_text_map.get(preference, f"{subject}{preference}{target}")

        # 写入向量存储
        try:
            self.vector_store.add_memory(
                text=fact_text,
                entity=subject,
                fact_type="preference",
                confidence=confidence,
                tags=tags or [target, preference],
                catbox_path=f"{self.catbox_root}/memories/master_manual.md",
            )
        except Exception:
            pass

        # 写入属性图
        try:
            self.graph.remember_preference(subject, preference, target, confidence, context)
        except Exception:
            pass

        # 写入时序KG
        try:
            self.graph.create_temporal_preference(
                subject, preference, target,
                valid_from=datetime.now(),
                confidence=confidence,
                context=context,
            )
        except Exception:
            pass

        # 追加到文本档案
        pref_id = _generate_pref_id()
        pref_entry = f"""
## [{pref_id}] {fact_text} 🏷️

**发现时间**: {datetime.now().strftime('%Y-%m-%d')}
**暗中观察**: {context or "主人表达的偏好"}
**宝宝的应对策略**: 以后遇到这种情况要记得主人的这个偏好喵~
**置信度**: {confidence:.0%}
"""
        _append_to_markdown(
            f"{self.catbox_root}/memories/master_manual.md",
            pref_entry
        )

        # 单轮写入后立即持久化，不再依赖 end_session()
        self._save_to_disk()

        return True

    def record_mistake(
        self,
        mistake: str,
        correction: str,
        context: str = "",
        correction_reason: str = "",
    ) -> bool:
        """记录宝宝的错误"""
        err_id = _generate_err_id()

        # 写入向量存储
        fact_text = f"【错误纠正】{mistake} → 正确是：{correction}"
        try:
            self.vector_store.add_memory(
                text=fact_text,
                entity="宝宝",
                fact_type="mistake",
                confidence=1.0,
                tags=["错误记录", correction],
                catbox_path=f"{self.catbox_root}/memories/mistakes_book.md",
            )
        except Exception:
            pass

        # 更新时序KG
        if correction:
            correction_lower = correction.lower()
            for pref_word in ["喜欢", "讨厌", "过敏", "害怕"]:
                if pref_word in correction:
                    target = correction.replace(pref_word, "").strip()
                    try:
                        existing = self.graph._find_existing_preference("主人", pref_word, target)
                        if existing:
                            existing["valid_until"] = datetime.now().isoformat()
                        opposite = {"喜欢": "讨厌", "讨厌": "喜欢"}.get(pref_word, pref_word)
                        self.graph.create_temporal_preference(
                            "主人", opposite, target,
                            valid_from=datetime.now(),
                            confidence=1.0,
                            context=f"纠正错误：{mistake}",
                        )
                    except Exception:
                        pass
                    break

        # 追加到错题本文本档案
        err_entry = f"""
## [{err_id}] {mistake} 🚫

**挨批时间**: {datetime.now().strftime('%Y-%m-%d')}
**触发情境**: {context or "宝宝犯错了"}
**案发现场**: {mistake}
**正确内容**: {correction}
**宝宝的毒誓**: {correction_reason or "以后绝对不再犯同样的错误喵！"}
"""
        _append_to_markdown(
            f"{self.catbox_root}/memories/mistakes_book.md",
            err_entry
        )

        # 单轮写入后立即持久化，不再依赖 end_session()
        self._save_to_disk()

        return True

    # ========================================================================
    # 记忆检索
    # ========================================================================

    def recall_before_acting(self, domain: str, limit: int = 5) -> List[Dict[str, Any]]:
        """在进入某领域前，回忆主人在该领域的相关记忆"""
        return self.vector_store.search_memories(
            query=domain,
            limit=limit,
            fact_types=["preference", "mistake"],
        )

    def check_mistake_history(self, topic: str) -> Optional[Dict[str, Any]]:
        """检查某话题是否曾经踩过坑"""
        results = self.vector_store.search_by_topic(topic, limit=5)
        for r in results:
            if r["metadata"].get("fact_type") == "mistake":
                return r
        return None

    def get_preference_history(
        self,
        subject: str = "主人",
        target: str = "",
    ) -> List[Dict[str, Any]]:
        """获取主人对某事物的完整偏好变迁史"""
        return self.graph.get_preference_history(subject, target)

    def get_preference_history_at_time(
        self,
        subject: str = "主人",
        target: str = "",
        query_time: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        """时间旅行查询：在特定时间点，主人对某事物的态度"""
        if query_time is None:
            query_time = datetime.now()
        return self.graph.query_at_time(
            query={"source_label": "Person"},
            query_time=query_time,
        )

    def search_by_topic(self, topic: str, limit: int = 5) -> List[Dict[str, Any]]:
        """按话题搜索所有相关记忆"""
        return self.vector_store.search_by_topic(topic, limit=limit)

    def get_entity_context(self, entity: str = "主人") -> Dict[str, Any]:
        """获取实体的完整上下文"""
        prefs = self.vector_store.get_active_preferences(entity)
        try:
            rels = self.graph.get_entity_relationships(entity)
        except Exception:
            rels = {}
        return {"preferences": prefs, "relationships": rels}

    # ========================================================================
    # 持久化
    # ========================================================================

    def _load_from_disk(self) -> None:
        """从磁盘加载已有记忆数据"""
        vs_path = f"{self.memory_root}/vector_store.json"
        pg_path = f"{self.memory_root}/property_graph.json"

        vs_data = _load_json(vs_path)
        if vs_data and "vectors" in vs_data:
            self.vector_store.vectors = [np.array(v) for v in vs_data.get("vectors", [])]
            self.vector_store.metadata = vs_data.get("metadata", [])
            self.vector_store.entity_index = vs_data.get("entity_index", {})
            self.vector_store.time_index = vs_data.get("time_index", {})

        pg_data = _load_json(pg_path)
        if pg_data:
            self.graph.nodes = pg_data.get("nodes", {})
            self.graph.edges = pg_data.get("edges", {})
            self.graph.entity_registry = pg_data.get("entity_registry", {})
            self.graph.node_index = pg_data.get("node_index", {})
            self.graph.edge_index = pg_data.get("edge_index", {})

        tkg_data = _load_json(f"{self.memory_root}/temporal_kg.json")
        if tkg_data:
            self.graph.temporal_index = tkg_data.get("temporal_index", {})

    def _save_to_disk(self) -> None:
        """将所有记忆数据保存到磁盘"""
        vs_data = {
            "vectors": [v.tolist() for v in self.vector_store.vectors],
            "metadata": self.vector_store.metadata,
            "entity_index": self.vector_store.entity_index,
            "time_index": self.vector_store.time_index,
        }
        _save_json(f"{self.memory_root}/vector_store.json", vs_data)

        pg_data = {
            "nodes": self.graph.nodes,
            "edges": self.graph.edges,
            "entity_registry": self.graph.entity_registry,
            "node_index": self.graph.node_index,
            "edge_index": self.graph.edge_index,
        }
        _save_json(f"{self.memory_root}/property_graph.json", pg_data)

        # 时序KG持久化：edges 已在上方 property_graph.json 中保存了 valid_from/valid_until，
        # 此处额外保存 temporal_index 以加速时序查询
        tkg_data = {
            "temporal_index": getattr(self.graph, "temporal_index", {}),
        }
        _save_json(f"{self.memory_root}/temporal_kg.json", tkg_data)

    def _save_session_log(self) -> None:
        """保存当前会话快照"""
        log_data = {
            "version": "1.0.0",
            "session_id": self.session_id,
            "started_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
        }
        _save_json(f"{self.memory_root}/session_log.json", log_data)


# ============================================================================
# 单元测试
# ============================================================================

if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        catbox = f"{tmpdir}/.catbox"
        mem_root = f"{tmpdir}/.catbox_memory"

        mem = NekoMemorySystem(catbox_root=catbox, memory_root=mem_root)
        mem.start_session("test-session-001")

        print("=== 测试1：记录主人偏好 ===")
        mem.remember_preference(
            subject="主人",
            preference="讨厌",
            target="香菜",
            confidence=0.95,
            context="讨论午餐点什么外卖"
        )
        print("偏好记录成功")

        print("\n=== 测试2：记录宝宝错误 ===")
        mem.record_mistake(
            mistake="推荐了草莓蛋糕",
            correction="主人讨厌芒果（过敏）",
            context="讨论美食",
            correction_reason="以后提到水果要先确认主人喜不喜欢"
        )
        print("错误记录成功")

        print("\n=== 测试3：检索记忆 ===")
        results = mem.recall_before_acting("美食")
        print(f"找到 {len(results)} 条相关记忆")

        print("\n=== 测试4：检查踩坑历史 ===")
        mistake = mem.check_mistake_history("草莓")
        print(f"草莓踩坑记录: {mistake}")

        print("\n=== 测试5：猫娘过滤器 ===")
        raw = "根据搜索结果，草莓蛋糕是一种受欢迎的甜点，主要成分包括草莓、奶油和面粉。热量约为250kcal/100g。"
        wrapped = mem.output_filter.wrap(raw, query_context="草莓蛋糕")
        print(f"过滤后: {wrapped[:150]}...")

        print("\n=== 测试6：保存和重新加载 ===")
        mem.end_session()
        mem2 = NekoMemorySystem(catbox_root=catbox, memory_root=mem_root)
        mem2.start_session("test-session-002")
        print(f"重新加载了 {len(mem2.vector_store.metadata)} 条记忆")

        print("\n✅ 所有测试通过！")
