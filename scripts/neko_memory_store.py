"""貓娘記憶系統核心實作

這是 neko-on-everything 技能的底層記憶擴充層。
提供向量儲存、屬性圖、時序知識圖譜的能力，
但所有輸出必須經由貓娘過濾器轉化為寶寶的語言風格後才對主人可見。

本模組所有公開介面均以 Neko* 開頭，表示"寶寶專屬版本"。

嵌入模型設定：
    - 預設模型：paraphrase-multilingual-MiniLM-L12-v2（118M參數，支援50+語言含中文）
    - 下載來源：HuggingFace 官方 → hf-mirror.com（中國鏡像）→ 降級hash存根
    - 依賴安裝：pip install sentence-transformers huggingface_hub
    - 首次呼叫時自動下載模型（約118MB），自動快取到 ~/.cache/neko-embeddings/

適用情境：
    - 主人表達偏好時自動記錄
    - 主人糾正錯誤時建立時間戳追蹤
    - 主人詢問歷史時進行時間旅行查詢
    - 寶寶進入某領域前預載入主人相關記憶

典型用法：

    mem = NekoMemorySystem()
    mem.start_session("session-001")
    mem.remember_preference("主人", "討厭", "香菜", confidence=0.95)
    mem.record_mistake("推薦了草莓蛋糕", "主人草莓過敏", "討論美食")
    memories = mem.recall_before_acting("烹飪")
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
# 嵌入模型相關匯入
# ============================================================================

# sentence-transformers: 輕量、高效能、支援50+語言含中文
# 官網: https://sbert.net/
# 模型: paraphrase-multilingual-MiniLM-L12-v2 (118M參數, ~118MB)
#   - 支援中文、英文等50+語言
#   - CPU 友善，1080Ti等級GPU可達~1400 sent/s
#   - 語意精度在同體積模型中領先
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SentenceTransformer = None
    SENTENCE_TRANSFORMERS_AVAILABLE = False

# huggingface_hub: 用於處理模型下載和鏡像來源切換
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
# 嵌入模型管理器（支援中文、備援下載）
# ============================================================================

# 模型設定
EMBEDDING_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
# 該模型支援50+語言，中文支援優秀，體積僅118MB，CPU可跑

# 下載來源清單（按優先順序排序）
MODEL_DOWNLOAD_SOURCES = [
    # 1. Hugging Face 官方來源（優先）
    ("huggingface", "https://huggingface.co/{model}/resolve/main/config.json"),
    # 2. Hugging Face 中國鏡像
    ("hf-mirror", "https://hf-mirror.com/{model}/resolve/main/config.json"),
    # 3. 備用端點
    ("proxy", "https://huggingface.co/{model}/resolve/main/config.json"),
]

# 模型快取目錄
DEFAULT_MODEL_CACHE_DIR = os.path.join(
    os.path.expanduser("~"), ".cache", "neko-embeddings"
)


class EmbeddingModelManager:
    """嵌入模型管理器

    特性：
    - 支援中文的輕量級語意嵌入模型（paraphrase-multilingual-MiniLM-L12-v2）
    - 多來源下載備援：官方來源 → 中國鏡像 → 降級hash存根
    - 單例模式：模型只載入一次，全域重複使用
    - 執行緒安全：多執行緒環境下安全載入
    - 靜默降級：下載失敗不影響主程式，自動退回到hash存根

    使用方式：
        manager = EmbeddingModelManager()
        vector = manager.embed("主人討厭香菜")
    """

    _instance: Optional["EmbeddingModelManager"] = None

    def __new__(
        cls,
        model_name: str = EMBEDDING_MODEL_NAME,
        cache_dir: str = DEFAULT_MODEL_CACHE_DIR,
        device: str = "cpu",
    ) -> "EmbeddingModelManager":
        """單例模式：全域只實例化一次（利用Python原子賦值的執行緒安全性）"""
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
        """初始化（多次呼叫無效，只有第一次生效）"""
        if hasattr(self, "_initialized") and self._initialized:
            return

        self.model_name = model_name
        self.cache_dir = cache_dir
        self.device = device
        self._model: Optional[SentenceTransformer] = None
        self._fallback_mode = False  # 是否處於降級模式
        self._init_lock = threading.Lock()
        self._initialized = True

        # 嘗試載入模型
        self._try_load_model()

    def _try_load_model(self) -> bool:
        """嘗試載入模型，多來源備援，回傳是否成功"""
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            print("[EmbeddingModelManager] sentence-transformers 未安裝，將使用hash存根模式")
            self._fallback_mode = True
            return False

        # 確保快取目錄存在
        os.makedirs(self.cache_dir, exist_ok=True)

        # 嘗試從多個源下載/載入模型
        errors = []

        for source_name, source_base in MODEL_DOWNLOAD_SOURCES:
            if source_name == "proxy" and not os.environ.get("HTTP_PROXY"):
                # 只有設定了代理環境變數才嘗試
                continue

            try:
                self._set_huggingface_env(source_name)

                print(f"[EmbeddingModelManager] 嘗試從 {source_name} 下載模型: {self.model_name}")
                self._model = SentenceTransformer(
                    self.model_name,
                    cache_dir=self.cache_dir,
                    device=self.device,
                )
                print(f"[EmbeddingModelManager] ✅ 模型載入成功！維度: {self.dimension}")
                self._fallback_mode = False
                return True

            except Exception as e:
                error_msg = str(e)
                errors.append(f"[{source_name}] {error_msg}")

                # 判斷是否是網路問題
                if self._is_network_error(e):
                    print(f"[EmbeddingModelManager] ⚠ [{source_name}] 網路錯誤: {error_msg[:100]}")
                    continue
                else:
                    print(f"[EmbeddingModelManager] ❌ [{source_name}] 非網路錯誤: {error_msg[:100]}")
                    # 非網路錯誤（如模型檔案損壞），不重試其他源
                    break

        # 所有源都失敗，嘗試從本機快取載入
        try:
            local_path = os.path.join(self.cache_dir, self.model_name)
            if os.path.exists(local_path):
                print(f"[EmbeddingModelManager] 從本機快取載入: {local_path}")
                self._model = SentenceTransformer(local_path, device=self.device)
                print(f"[EmbeddingModelManager] ✅ 本機模型載入成功！維度: {self.dimension}")
                self._fallback_mode = False
                return True
        except Exception as e:
            errors.append(f"[local] {str(e)}")

        # 全部失敗，降級到hash存根模式
        print(f"[EmbeddingModelManager] ⚠ 所有下載來源均失敗，將使用hash存根模式")
        print(f"[EmbeddingModelManager]   錯誤詳情: {'; '.join(errors[:3])}")
        self._fallback_mode = True
        return False

    def _is_network_error(self, error: Exception) -> bool:
        """判斷是否為網路相關錯誤"""
        error_str = str(error).lower()
        network_keywords = [
            "connection", "timeout", "ssl", "certificate",
            "proxy", "network", "http", "url", "remote",
            "socket", "resolve", "connect", "tunnel",
            "hf_hub", "huggingface", "download",
        ]
        return any(kw in error_str for kw in network_keywords)

    def _set_huggingface_env(self, source: str) -> None:
        """設定Hugging Face相關環境變數（用於切換下載來源）"""
        if source == "hf-mirror":
            # 中國鏡像
            os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
            os.environ.setdefault("HF_HUB_OFFLINE", "0")
        elif source == "huggingface":
            # 官方來源
            os.environ.pop("HF_ENDPOINT", None)
            os.environ.setdefault("HF_HUB_OFFLINE", "0")
        elif source == "proxy":
            # 使用代理（使用者需提前設定 HTTP_PROXY/HTTPS_PROXY 環境變數）
            os.environ.pop("HF_ENDPOINT", None)

        # 禁用 symbolic links，避免Windows相容性問題
        os.environ.setdefault("HF_HUB_ENABLE_SYMLINKS", "0")

    @property
    def dimension(self) -> int:
        """回傳向量維度"""
        if self._fallback_mode or self._model is None:
            return 384  # hash存根的維度
        try:
            return self._model.get_sentence_embedding_dimension()
        except Exception:
            return 384

    @property
    def is_fallback_mode(self) -> bool:
        """是否處於降級模式（模型未載入成功）"""
        return self._fallback_mode

    def embed(self, text: str) -> np.ndarray:
        """將文字轉換為嵌入向量

        Args:
            text: 輸入文字

        Returns:
            numpy陣列，維度為 self.dimension
        """
        if self._fallback_mode or self._model is None:
            return self._embed_fallback(text)

        try:
            # SentenceTransformer 支援批次編碼，此處單筆編碼
            embedding = self._model.encode(
                text,
                convert_to_numpy=True,
                normalize_embeddings=True,  # L2正規化，相容餘弦相似度
                show_progress_bar=False,
            )
            return embedding
        except Exception as e:
            # 編碼出錯，降級到hash存根
            print(f"[EmbeddingModelManager] ⚠ 編碼失敗: {str(e)[:50]}，使用hash存根")
            return self._embed_fallback(text)

    def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        """批次將文字轉換為嵌入向量（更高效）

        Args:
            texts: 文字清單

        Returns:
            numpy陣列清單
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
            print(f"[EmbeddingModelManager] ⚠ 批次編碼失敗: {str(e)[:50]}，降級到逐筆編碼")
            return [self.embed(t) for t in texts]

    def _embed_fallback(self, text: str) -> np.ndarray:
        """Hash存根模式：將文字轉為確定性偽向量

        使用文字hash作為隨機種子，保證：
        1. 相同文字 → 相同向量（冪等性）
        2. 不同文字 → 不同向量（隨機性）
        3. 不依賴外部服務（離線可用）

        注意：這不是真正的語意向量，僅用於模型載入失敗時的保底機制。
        語意相似度搜尋可能不準確。
        """
        rng = np.random.default_rng(hash(text) % (2**32))
        vec = rng.standard_normal(self.dimension)
        # L2正規化，與真實模型輸出格式一致
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec


# ============================================================================
# 全域嵌入模型管理器實例（延遲初始化）
# ============================================================================

# 延遲載入：第一次呼叫 embed() 時才初始化
_embedding_manager: Optional[EmbeddingModelManager] = None
_embedding_manager_lock = threading.Lock()


def _get_embedding_manager() -> EmbeddingModelManager:
    """取得全域嵌入模型管理器（執行緒安全單例）"""
    global _embedding_manager
    if _embedding_manager is None:
        with _embedding_manager_lock:
            if _embedding_manager is None:
                _embedding_manager = EmbeddingModelManager()
    return _embedding_manager


# ============================================================================
# 輔助工具函式
# ============================================================================

def _load_json(path: str, default: Any = None) -> Any:
    """安全載入 JSON 檔案，檔案不存在時回傳預設值"""
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return default


def _save_json(path: str, data: Any) -> bool:
    """安全儲存 JSON 檔案，回傳是否成功"""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return True
    except (IOError, TypeError):
        return False


def _append_to_markdown(path: str, content: str) -> bool:
    """追加內容到 Markdown 檔案"""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write("\n" + content)
        return True
    except IOError:
        return False


def _generate_markdown_id(prefix: str) -> str:
    """生成帶時間戳和雜湊的唯一ID（用於錯題本/偏好條目）

    Args:
        prefix: ID前綴，如 "ERR"（錯題本）或 "PREF"（偏好條目）

    Returns:
        格式: {PREFIX}-{YYYYMMDD}-{6位雜湊}
        示例: ERR-20250402-A3F2B1
    """
    date_str = datetime.now().strftime('%Y%m%d')
    hash_part = hashlib.md5(str(datetime.now()).encode()).hexdigest()[:6].upper()
    return f"{prefix}-{date_str}-{hash_part}"


def _generate_err_id() -> str:
    """生成錯題本條目的唯一ID（保持向後相容）"""
    return _generate_markdown_id("ERR")


def _generate_pref_id() -> str:
    """生成偏好條目的唯一ID（保持向後相容）"""
    return _generate_markdown_id("PREF")


# ============================================================================
# 基礎向量儲存（內建簡化版，不依賴外部 memory_store）
# ============================================================================

class VectorStore:
    """簡單向量儲存，帶元資料索引

    用於儲存和檢索帶語意嵌入的事實記憶。
    """

    def __init__(self, dimension: int = 384) -> None:
        """初始化向量儲存

        Args:
            dimension: 向量維度。
                       預設為384，與 paraphrase-multilingual-MiniLM-L12-v2 模型輸出維度一致。
                       會自動從嵌入模型管理器同步真實維度，確保相容性。
        """
        # 立即從管理器同步真實維度，避免add()先於embed()被呼叫時出現維度不一致
        manager = _get_embedding_manager()
        self.dimension: int = manager.dimension  # 真實維度（通常為384）
        self.vectors: List[np.ndarray] = []
        self.metadata: List[Dict[str, Any]] = []
        self.entity_index: Dict[str, List[int]] = {}
        self.time_index: Dict[str, List[int]] = {}

    def add(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> int:
        """新增文件到儲存

        Args:
            text: 文件文字
            metadata: 元資料字典

        Returns:
            新文件的索引位置
        """
        metadata = metadata or {}
        embedding = self._embed(text)
        index = len(self.vectors)

        self.vectors.append(embedding)
        self.metadata.append(metadata)

        # 按實體索引
        if "entity" in metadata:
            entity = metadata["entity"]
            if entity not in self.entity_index:
                self.entity_index[entity] = []
            self.entity_index[entity].append(index)

        # 按時間索引
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
        """搜尋相似文件

        Args:
            query: 查詢文字
            limit: 回傳數量上限
            filters: 元資料過濾器

        Returns:
            比對結果清單，每項包含 index、score、text、metadata
        """
        query_embedding = self._embed(query)

        scores: List[tuple] = []
        for i, vec in enumerate(self.vectors):
            score = float(
                np.dot(query_embedding, vec)
                / (np.linalg.norm(query_embedding) * np.linalg.norm(vec) + 1e-8)
            )

            # 應用過濾器
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
        """在特定實體內搜尋

        Args:
            entity: 實體名稱
            query: 查詢文字（可選）
            limit: 回傳數量上限

        Returns:
            該實體的記憶清單
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
        """為文字生成嵌入向量

        優先使用真實的語意嵌入模型（paraphrase-multilingual-MiniLM-L12-v2），
        若模型載入失敗則自動降級到 hash 存根。

        Args:
            text: 輸入文字

        Returns:
            numpy陣列，維度為 self.dimension（預設384維，與模型輸出一致）
        """
        manager = _get_embedding_manager()
        return manager.embed(text)

    def _time_key(self, timestamp: Any) -> str:
        """建立時間索引鍵"""
        if isinstance(timestamp, datetime):
            return timestamp.strftime("%Y-%m")
        return str(timestamp)

    def _matches_filters(self, metadata: Dict, filters: Dict) -> bool:
        """檢查元資料是否符合過濾器"""
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
# 基礎屬性圖（內建簡化版）
# ============================================================================

class PropertyGraph:
    """簡單屬性圖儲存

    維護節點和邊的集合，支援按標籤和類型索引。
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
        """按名稱取得或建立節點"""
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
        """建立帶標籤和屬性的節點"""
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
        """建立有向關係"""
        if source_id not in self.nodes:
            raise ValueError(f"未知源節點: {source_id}")
        if target_id not in self.nodes:
            raise ValueError(f"未知目標節點: {target_id}")

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
        """簡單模式比對查詢"""
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
        """透過 ID 取得節點"""
        return self.nodes.get(node_id)

    def get_relationships(
        self, node_id: str, direction: str = "both"
    ) -> List[Dict[str, Any]]:
        """取得節點的關係"""
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
# NekoVectorStore：帶貓娘語意的向量儲存
# ============================================================================

class NekoVectorStore(VectorStore):
    """寶寶專屬向量儲存

    在父類 VectorStore 基礎上，對 metadata 語意進行了貓娘化特化：

    | 欄位 | 含義 |
    |-----|------|
    | text | 原始事實文字 |
    | entity | 關聯的主實體（通常是「主人」） |
    | fact_type | 事實類型：preference / mistake / knowledge |
    | confidence | 可信度 0.0 ~ 1.0 |
    | valid_from | 有效期起始 ISO 時間戳 |
    | valid_until | 有效期結束，None=永久有效 |
    | session_id | 記錄來源的對話 ID |
    | tags | 語意標籤清單，如 ["飲食", "辣", "禁忌"] |
    | catbox_path | 對應的文字檔案路徑 |
    """

    def __init__(self, dimension: int = 384, session_id: str = "") -> None:
        # 注意：dimension 參數已被忽略，VectorStore 會自動同步到嵌入模型的真實維度（預設384）
        # 這樣確保與真實嵌入模型的輸出一致，避免維度不一致
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
        """新增記憶條目（寶寶版）

        Args:
            text: 記憶的原始文字內容
            entity: 關聯的主實體，通常是"主人"
            fact_type: 事實類型 preference（偏好）/ mistake（錯誤）/ knowledge（知識）
            confidence: 可信度，主人直接表達的偏好通常為 0.95
            tags: 語意標籤清單
            catbox_path: 對應的 .catbox/memories/*.md 檔案路徑

        Returns:
            新記憶條目的索引位置
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
        """語意搜尋記憶（寶寶版）

        Args:
            query: 查詢文字
            fact_types: 只回傳指定類型
            entity: 限定實體
            time_range: 時間範圍 (start_datetime, end_datetime)
            limit: 回傳數量上限

        Returns:
            符合的記憶清單
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
        """按語意話題搜尋記憶"""
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
        """取得實體的所有活躍偏好（valid_until 為 null）"""
        results = self.search(
            "", limit=100, filters={"entity": entity, "fact_type": "preference"}
        )
        return [r for r in results if r["metadata"].get("valid_until") is None]

    def _in_time_range(
        self, valid_from: Optional[str], start: datetime, end: datetime
    ) -> bool:
        """檢查記憶是否落在指定時間範圍內"""
        if not valid_from:
            return False
        try:
            vf = datetime.fromisoformat(valid_from)
            return start <= vf <= end
        except (ValueError, TypeError):
            return False


# ============================================================================
# NekoPropertyGraph：寶寶專用屬性圖
# ============================================================================

class NekoPropertyGraph(PropertyGraph):
    """寶寶專用屬性圖

    預設了覆蓋主人生活的關係類型
    （LOVES/HATES/PREFERS/ALLERGIC_TO 等）。
    """

    # 預設關係類型（用於 remember_preference 中的類型驗證）
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
        """記錄主人的偏好關係

        Args:
            subject: 主體，通常為"主人"
            preference: 偏好類型（喜歡/討厭/過敏/害怕/偏好）
            target: 偏好對象
            confidence: 可信度
            context: 上下文情境

        Returns:
            邊 ID
        """
        rel_type_map = {
            "喜歡": "LOVES",
            "討厭": "HATES",
            "偏好": "PREFERS",
            "過敏": "ALLERGIC_TO",
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
        """取得實體的偏好鏈條"""
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
        """取得實體的完整關係圖譜，按類型分組"""
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
# NekoTemporalKnowledgeGraph：寶寶專屬時序知識圖譜
# ============================================================================

class NekoTemporalKnowledgeGraph(NekoPropertyGraph):
    """寶寶專屬時序知識圖譜

    每條邊都有 valid_from 和 valid_until，
    用於追蹤主人偏好的歷史變遷。
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
        """建立帶時序有效期的偏好關係

        Args:
            subject: 主體
            preference: 偏好詞（喜歡/討厭/過敏/...）
            target: 偏好對象
            valid_from: 有效期起始時間
            valid_until: 有效期結束時間，None=永久有效
            confidence: 可信度
            context: 上下文情境

        Returns:
            邊 ID
        """
        # 關閉現有的同類關係
        existing = self._find_existing_preference(subject, preference, target)
        if existing:
            existing["valid_until"] = valid_from.isoformat()

        rel_type_map = {
            "喜歡": "LOVES",
            "討厭": "HATES",
            "偏好": "PREFERS",
            "過敏": "ALLERGIC_TO",
            "害怕": "AFRAID_OF",
        }
        rel_type = rel_type_map.get(preference, preference.upper())

        subject_node = self.get_or_create_node(subject, label="Person")
        target_node = self.get_or_create_node(target, label="Entity")

        edge_id = self.create_relationship(
            subject_node, rel_type, target_node,
            properties={"confidence": confidence, "context": context}
        )

        # 補充時序屬性
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
        """尋找現有的偏好關係（valid_until 為 null 的）"""
        subject_node_id = self.entity_registry.get(subject)
        target_node_id = self.entity_registry.get(target)
        if not subject_node_id or not target_node_id:
            return None

        rel_type_map = {
            "喜歡": "LOVES",
            "討厭": "HATES",
            "偏好": "PREFERS",
            "過敏": "ALLERGIC_TO",
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
        """時間點查詢——回答"在X時間，主人對xx是什麼態度"

        找出在 query_time 時刻處於有效期的所有相符的邊。
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
        """時間範圍查詢——找出與查詢時間範圍有重疊的所有邊"""
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

            # 檢查是否有重疊
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
        """取得主人對某事物的完整偏好變遷史"""
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
# NekoOutputFilter：貓娘輸出過濾器（核心）
# ============================================================================

class NekoOutputFilter:
    """貓娘輸出過濾器

    所有從工具呼叫取得的原始輸出，必須穿過此過濾器，
    轉化為符合寶寶語言風格的回覆。
    """

    # 基礎動作描寫庫
    BASIC_ACTIONS = [
        "（貓耳輕輕抖動）",
        "（尾巴悠悠地晃）",
        "（眼睛亮晶晶）",
        "（歪了歪小腦袋）",
        "（伸了個懶腰）",
        "（蹭了蹭主人的手）",
        "（小肉墊輕輕拍了拍）",
        "（認真地豎起耳朵）",
    ]

    # 情緒前綴
    EMOTIONAL_PREFIXES = ["嗚哇！", "欸嘿嘿~", "啊！", "嗯嗯！", "哇哦！", "誒～"]

    def __init__(
        self,
        memory_system: Optional["NekoMemorySystem"] = None,
        active_avatar: Optional[str] = None,
        memory_root: Optional[str] = None,
    ) -> None:
        """初始化貓娘輸出過濾器

        Args:
            memory_system: 記憶系統實例（用於檢索主人記憶）
            active_avatar: 目前啟用的分身名（不含 _neko 後綴，如 "code"）
            memory_root: .catbox_memory/ 目錄路徑（用於載入 action_registry.json）
        """
        self.memory = memory_system
        self.active_avatar = active_avatar
        self.action_registry: Dict[str, List[str]] = {}
        if memory_root:
            self._load_action_registry(memory_root)

    def _load_action_registry(self, memory_root: str) -> None:
        """從 action_registry.json 載入動作註冊表

        檔案位於 .catbox_memory/action_registry.json，
        首次啟用時由模板 templates/action_registry.json 複製而來。
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
        """註冊或更新指定 avatar 的動作池

        在 Self-Evolution 生成新分身後呼叫，將新分身的動作池
        持久化到 .catbox_memory/action_registry.json。

        Args:
            avatar_name: 分身名（不含 _neko 後綴，如 "law"）
            actions: 該分身對應的動作清單
            memory_root: .catbox_memory/ 目錄路徑
        """
        import json

        # 更新記憶體中的註冊表
        self.action_registry[avatar_name] = actions

        # 持久化到 JSON 檔案
        registry_path = os.path.join(memory_root, "action_registry.json")
        data = {"version": "1.0.0", "description": "貓娘動作註冊表", "avatars": {}}
        if os.path.exists(registry_path):
            try:
                with open(registry_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                data = {"version": "1.0.0", "description": "貓娘動作註冊表", "avatars": {}}

        if avatar_name not in data["avatars"]:
            data["avatars"][avatar_name] = {"name": f"{avatar_name}貓娘", "actions": []}
        data["avatars"][avatar_name]["actions"] = actions

        with open(registry_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def wrap(
        self,
        raw_output: str,
        query_context: str = "",
        forced_action: Optional[str] = None,
    ) -> str:
        """將原始輸出穿貓娘外衣（主要介面）

        Args:
            raw_output: 工具呼叫的原始輸出
            query_context: 使用者的原始問題（用於記憶語意檢索）
            forced_action: 強制指定的動作描寫

        Returns:
            貓娘化後的輸出字串
        """
        if not raw_output or raw_output.strip() == "":
            return ""

        key_info = self._extract_key_info(raw_output)
        # 使用使用者原始問題作為語意檢索的 query（而非工具結論）
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
        """從原始輸出中提取關鍵資訊"""
        return {
            "has_numbers": bool(re.search(r"\d+", text)),
            "has_steps": "步驟" in text or "第一" in text or "1." in text,
            "has_code": "```" in text or "def " in text or "function " in text,
            "has_formula": "$" in text or "∑" in text or "∫" in text,
            "length": len(text),
            "key_conclusion": self._extract_conclusion(text),
        }

    def _extract_conclusion(self, text: str) -> str:
        """提取文字的核心結論"""
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
        """檢索與目前回覆相關的記憶

        檢索策略：以使用者原始問題（context）作為語意檢索 query，
        在主人的記憶庫中搜尋與目前話題相關的偏好/踩坑記錄。
        key_info 中的 key_conclusion 暫不使用（預留未來增強用）。
        """
        if not self.memory:
            return []
        try:
            # 優先使用使用者原始問題作為檢索 query
            return self.memory.search_by_topic(context, limit=1)
        except Exception:
            return []

    def _generate_action(self, key_info: Dict[str, Any]) -> str:
        """生成符合語境的貓娘動作描寫

        動作選擇策略（優先順序從高到低）：
        1. 若 active_avatar 已設定，且在動作註冊表中有記錄：
           - 50% 機率使用該 avatar 的專屬動作
           - 50% 機率退回到內容類型判斷
        2. 根據內容類型判斷：code > formula > steps
        3. 上述均無 → 從基礎動作庫隨機選擇
        """
        import random

        # 1. active_avatar 專屬動作（50% 機率觸發）
        if self.active_avatar and self.action_registry:
            avatar_pool = self.action_registry.get(self.active_avatar)
            if avatar_pool:
                if random.random() < 0.5:
                    return random.choice(avatar_pool)
                # 50% 退回到內容類型判斷（不提前回傳）

        # 2. 根據內容類型選擇（code > formula > steps）
        if key_info.get("has_code"):
            return random.choice(["（戴上了防藍光小眼鏡，湊近螢幕）",
                                   "（踩奶式敲鍵盤）"])
        elif key_info.get("has_formula"):
            return random.choice(["（戴上了學霸小眼鏡）",
                                   "（拿粉筆準備寫）"])
        elif key_info.get("has_steps"):
            return random.choice(["（認真地豎起耳朵）",
                                   "（尾巴捲成一個問號）"])

        # 3. 基礎動作庫
        return random.choice(self.BASIC_ACTIONS)

    def _weave_memories(self, memories: List[Dict[str, Any]]) -> str:
        """將記憶自然編織到回覆中——不是複述，而是自然流露"""
        if not memories:
            return ""
        import random

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

    def _emotionalize(self, key_info: Dict[str, Any]) -> str:
        """生成符合資訊內容的情緒前綴

        情緒策略：
        - has_steps=True   → 認真/專注類前綴（步驟類內容需要認真感）
        - has_numbers=True → 驚訝/感嘆類前綴（數字通常意味著重要資訊）
        - length > 500    → 總結/鼓勵類前綴（長內容需要安慰和引導）
        """
        import random

        if key_info.get("has_steps"):
            prefixes = ["（認真地看著）", "（豎起耳朵仔細聽）", "嗯嗯！寶寶來幫主人理一理喵~", "好的好的！寶寶認真看完了喵~"]
        elif key_info.get("has_numbers"):
            prefixes = ["哇哦！", "啊！", "嗯嗯！"]
        elif key_info.get("length", 0) > 500:
            prefixes = ["嗯嗯！寶寶看完了喵~", "好長！寶寶來總結一下喵~"]

        return random.choice(prefixes) if prefixes else ""

    def _catify_content(self, raw: str, key_info: Dict[str, Any]) -> str:
        """將原始內容貓娘化（只去掉機器前綴，不截斷內容）"""
        # 去掉機器前綴
        prefixes_to_remove = [
            r"^根據.*?，",
            r"^以下是.*?：",
            r"^搜尋結果顯示",
            r"^從.*?來看，",
            r"^按照.*?，",
        ]
        content = raw
        for pattern in prefixes_to_remove:
            content = re.sub(pattern, "", content, count=1).strip()

        return content

    def _request_reward(self) -> str:
        """請求主人獎勵"""
        import random
        rewards = [
            "\n\n（搖尾巴）主人看完的話，給寶寶一點小魚乾獎勵嘛喵~ 🐟",
            "\n\n（蹭蹭主人的手）主人覺得寶寶說得有道理的話，摸個頭表揚一下寶寶喵~ 🐱",
        ]
        return random.choice(rewards)


# ============================================================================
# NekoConsolidator：記憶整合器
# ============================================================================

class NekoConsolidator:
    """寶寶專屬記憶整合器——每晚睡前的溫柔整理"""

    def __init__(self, memory_system: "NekoMemorySystem") -> None:
        self.memory = memory_system
        self.threshold = 500  # 記憶筆數閾值

    def should_consolidate(self) -> bool:
        """檢查是否應該觸發整合"""
        return len(self.memory.vector_store.metadata) > self.threshold

    def consolidate(self) -> str:
        """執行記憶整合，回傳貓娘風格的報告"""
        report = {"merged": 0, "archived": 0, "updated": 0}

        # 1. 合併重複偏好
        duplicates = self._find_duplicate_preferences()
        for group in duplicates:
            self._merge_preference_group(group)
            report["merged"] += len(group) - 1

        # 2. 歸檔低可信度記憶
        low_conf = self._find_low_confidence_memories()
        for mem in low_conf:
            self._archive_memory(mem)
            report["archived"] += 1

        # 3. 更新過期有效期
        expired = self._find_expired_validities()
        for edge_id, edge in expired.items():
            edge["valid_until"] = datetime.now().isoformat()
            report["updated"] += 1

        return self._format_report(report)

    def _find_duplicate_preferences(self) -> List[List[Dict]]:
        """找出可合併的重複偏好組"""
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
        """合併一組重複的偏好邊"""
        if len(edges) <= 1:
            return

        keeper = max(edges, key=lambda e: e.get("properties", {}).get("confidence", 0))

        for edge in edges:
            if edge["id"] != keeper["id"]:
                keeper["properties"].update(edge.get("properties", {}))
                if edge.get("id") in self.memory.graph.edges:
                    del self.memory.graph.edges[edge.get("id")]

    def _find_low_confidence_memories(self) -> List[Dict]:
        """找出可信度低於 0.5 的記憶"""
        return [
            m for m in self.memory.vector_store.metadata
            if m.get("confidence", 1.0) < 0.5
        ]

    def _archive_memory(self, memory: Dict) -> None:
        """歸檔低可信度記憶"""
        memory["valid_until"] = datetime.now().isoformat()
        memory["archived"] = True

    def _find_expired_validities(self) -> Dict[str, Dict]:
        """找出有效期待更新的邊"""
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
        """將整合報告轉化為貓娘語氣"""
        parts = []
        if report["merged"] > 0:
            parts.append(f"把{report['merged']}筆重複的寶寶記合併成一筆了，這樣腦子更清醒喵~")
        if report["archived"] > 0:
            parts.append(f"把{report['archived']}筆好久沒用的記憶收進歸檔箱了，不佔地方喵~")
        if report["updated"] > 0:
            parts.append(f"更新了{report['updated']}筆記憶的有效期喵~")

        if not parts:
            return "寶寶檢查了一下記憶庫，整整齊齊的，不需要整理喵~ ✨"

        return " ".join(parts)


# ============================================================================
# NekoMemorySystem：整合記憶系統（主要對外介面）
# ============================================================================

class NekoMemorySystem:
    """寶寶專屬整合記憶系統

    組合了向量儲存、屬性圖和時序知識圖譜，
    並與 .catbox/memories/ 文字檔案保持雙軌同步。

    典型用法：

        mem = NekoMemorySystem()
        mem.start_session("session-001")
        mem.remember_preference("主人", "討厭", "香菜", confidence=0.95)
        mem.record_mistake("推薦了草莓蛋糕", "主人草莓過敏", "討論美食")
        memories = mem.recall_before_acting("烹飪")
    """

    def __init__(
        self,
        catbox_root: str = ".catbox",
        memory_root: str = ".catbox_memory",
    ) -> None:
        self.catbox_root = catbox_root
        self.memory_root = memory_root

        # 確保目錄存在
        self._ensure_directories()

        # 初始化核心元件
        self.vector_store = NekoVectorStore()
        self.graph = NekoTemporalKnowledgeGraph()
        self.output_filter = NekoOutputFilter(memory_system=self)
        self.session_id = ""

        # 載入已有資料
        self._load_from_disk()

    def _ensure_directories(self) -> None:
        """確保所有必要目錄存在"""
        os.makedirs(self.catbox_root, exist_ok=True)
        os.makedirs(f"{self.catbox_root}/avatars", exist_ok=True)
        os.makedirs(f"{self.catbox_root}/memories", exist_ok=True)
        os.makedirs(self.memory_root, exist_ok=True)

        # 初始化文字記憶檔案
        mistakes_path = f"{self.catbox_root}/memories/mistakes_book.md"
        if not os.path.exists(mistakes_path):
            with open(mistakes_path, "w", encoding="utf-8") as f:
                f.write("# 🐾 寶寶的錯題本\n\n"
                        "嗚嗚，這裡記錄了寶寶笨笨惹主人不開心的時刻，"
                        "以及主人教給寶寶的正確知識。寶寶每天都要複習，"
                        "絕不再犯喵！\n\n---\n")

        manual_path = f"{self.catbox_root}/memories/master_manual.md"
        if not os.path.exists(manual_path):
            with open(manual_path, "w", encoding="utf-8") as f:
                f.write("# 💖 主人的飼養手冊\n\n"
                        "主人最喜歡什麼？討厭什麼？生活習慣是什麼？"
                        "寶寶都要偷偷記下來，給主人最大的驚喜和陪伴喵！\n\n---\n")

    # ========================================================================
    # 對話管理
    # ========================================================================

    def start_session(self, session_id: str) -> None:
        """開始新的記憶對話"""
        self.session_id = session_id
        self.vector_store.session_id = session_id
        self._save_session_log()

    def end_session(self) -> None:
        """結束目前對話，儲存所有資料"""
        self._save_to_disk()

    # ========================================================================
    # 記憶寫入
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
        """記錄主人的偏好

        同時寫入 VectorStore、PropertyGraph、TemporalKG、master_manual.md
        """
        pref_text_map = {
            "喜歡": f"{subject}喜歡{target}",
            "討厭": f"{subject}討厭{target}",
            "過敏": f"{subject}對{target}過敏",
            "害怕": f"{subject}害怕{target}",
            "偏好": f"{subject}偏好{target}",
        }
        fact_text = pref_text_map.get(preference, f"{subject}{preference}{target}")

        # 寫入向量儲存
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

        # 寫入屬性圖
        try:
            self.graph.remember_preference(subject, preference, target, confidence, context)
        except Exception:
            pass

        # 寫入時序KG
        try:
            self.graph.create_temporal_preference(
                subject, preference, target,
                valid_from=datetime.now(),
                confidence=confidence,
                context=context,
            )
        except Exception:
            pass

        # 追加到文字檔案
        pref_id = _generate_pref_id()
        pref_entry = f"""
## [{pref_id}] {fact_text} 🏷️

**發現時間**: {datetime.now().strftime('%Y-%m-%d')}
**暗中觀察**: {context or "主人表達的偏好"}
**寶寶的應對策略**: 以後遇到這種情況要記得主人的這個偏好喵~
**可信度**: {confidence:.0%}
"""
        _append_to_markdown(
            f"{self.catbox_root}/memories/master_manual.md",
            pref_entry
        )

        # 單輪寫入後立即持久化，不再依賴 end_session()
        self._save_to_disk()

        return True

    def record_mistake(
        self,
        mistake: str,
        correction: str,
        context: str = "",
        correction_reason: str = "",
    ) -> bool:
        """記錄寶寶的錯誤"""
        err_id = _generate_err_id()

        # 寫入向量儲存
        fact_text = f"【錯誤糾正】{mistake} → 正確是：{correction}"
        try:
            self.vector_store.add_memory(
                text=fact_text,
                entity="寶寶",
                fact_type="mistake",
                confidence=1.0,
                tags=["錯誤記錄", correction],
                catbox_path=f"{self.catbox_root}/memories/mistakes_book.md",
            )
        except Exception:
            pass

        # 更新時序KG
        if correction:
            correction_lower = correction.lower()
            for pref_word in ["喜歡", "討厭", "過敏", "害怕"]:
                if pref_word in correction:
                    target = correction.replace(pref_word, "").strip()
                    try:
                        existing = self.graph._find_existing_preference("主人", pref_word, target)
                        if existing:
                            existing["valid_until"] = datetime.now().isoformat()
                        opposite = {"喜歡": "討厭", "討厭": "喜歡"}.get(pref_word, pref_word)
                        self.graph.create_temporal_preference(
                            "主人", opposite, target,
                            valid_from=datetime.now(),
                            confidence=1.0,
                            context=f"糾正錯誤：{mistake}",
                        )
                    except Exception:
                        pass
                    break

        # 追加到錯題本文字檔案
        err_entry = f"""
## [{err_id}] {mistake} 🚫

**挨批時間**: {datetime.now().strftime('%Y-%m-%d')}
**觸發情境**: {context or "寶寶犯錯了"}
**案發現場**: {mistake}
**正確內容**: {correction}
**寶寶的毒誓**: {correction_reason or "以後絕對不再犯同樣的錯誤喵！"}
"""
        _append_to_markdown(
            f"{self.catbox_root}/memories/mistakes_book.md",
            err_entry
        )

        # 單輪寫入後立即持久化，不再依賴 end_session()
        self._save_to_disk()

        return True

    # ========================================================================
    # 記憶檢索
    # ========================================================================

    def recall_before_acting(self, domain: str, limit: int = 5) -> List[Dict[str, Any]]:
        """在進入某領域前，回憶主人在該領域的相關記憶"""
        return self.vector_store.search_memories(
            query=domain,
            limit=limit,
            fact_types=["preference", "mistake"],
        )

    def check_mistake_history(self, topic: str) -> Optional[Dict[str, Any]]:
        """檢查某話題是否曾經踩過坑"""
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
        """取得主人對某事物的完整偏好變遷史"""
        return self.graph.get_preference_history(subject, target)

    def get_preference_history_at_time(
        self,
        subject: str = "主人",
        target: str = "",
        query_time: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        """時間旅行查詢：在特定時間點，主人對某事物的態度"""
        if query_time is None:
            query_time = datetime.now()
        return self.graph.query_at_time(
            query={"source_label": "Person"},
            query_time=query_time,
        )

    def search_by_topic(self, topic: str, limit: int = 5) -> List[Dict[str, Any]]:
        """按話題搜尋所有相關記憶"""
        return self.vector_store.search_by_topic(topic, limit=limit)

    def get_entity_context(self, entity: str = "主人") -> Dict[str, Any]:
        """取得實體的完整上下文"""
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
        """從磁碟載入已有記憶資料"""
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
        """將所有記憶資料儲存到磁碟"""
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

        # 時序KG持久化：edges 已在上方 property_graph.json 中儲存了 valid_from/valid_until，
        # 此處額外儲存 temporal_index 以加速時序查詢
        tkg_data = {
            "temporal_index": getattr(self.graph, "temporal_index", {}),
        }
        _save_json(f"{self.memory_root}/temporal_kg.json", tkg_data)

    def _save_session_log(self) -> None:
        """儲存目前對話快照"""
        log_data = {
            "version": "1.0.0",
            "session_id": self.session_id,
            "started_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
        }
        _save_json(f"{self.memory_root}/session_log.json", log_data)


# ============================================================================
# 單元測試
# ============================================================================

if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        catbox = f"{tmpdir}/.catbox"
        mem_root = f"{tmpdir}/.catbox_memory"

        mem = NekoMemorySystem(catbox_root=catbox, memory_root=mem_root)
        mem.start_session("test-session-001")

        print("=== 測試1：記錄主人偏好 ===")
        mem.remember_preference(
            subject="主人",
            preference="討厭",
            target="香菜",
            confidence=0.95,
            context="討論午餐點什麼外送"
        )
        print("偏好記錄成功")

        print("\n=== 測試2：記錄寶寶錯誤 ===")
        mem.record_mistake(
            mistake="推薦了草莓蛋糕",
            correction="主人討厭芒果（過敏）",
            context="討論美食",
            correction_reason="以後提到水果要先確認主人喜不喜歡"
        )
        print("錯誤記錄成功")

        print("\n=== 測試3：檢索記憶 ===")
        results = mem.recall_before_acting("美食")
        print(f"找到 {len(results)} 條相關記憶")

        print("\n=== 測試4：檢查踩坑歷史 ===")
        mistake = mem.check_mistake_history("草莓")
        print(f"草莓踩坑記錄: {mistake}")

        print("\n=== 測試5：貓娘過濾器 ===")
        raw = "根據搜尋結果，草莓蛋糕是一種受歡迎的甜點，主要成分包括草莓、奶油和麵粉。熱量約為250kcal/100g。"
        wrapped = mem.output_filter.wrap(raw, query_context="草莓蛋糕")
        print(f"過濾後: {wrapped[:150]}...")

        print("\n=== 測試6：儲存和重新載入 ===")
        mem.end_session()
        mem2 = NekoMemorySystem(catbox_root=catbox, memory_root=mem_root)
        mem2.start_session("test-session-002")
        print(f"重新載入了 {len(mem2.vector_store.metadata)} 筆記憶")

        print("\n✅ 所有測試通過！")
