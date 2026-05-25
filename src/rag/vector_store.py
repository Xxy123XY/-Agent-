"""
向量存储管理器 —— 基于 ChromaDB + OpenAI Embeddings，
提供知识库文档的向量化存储和语义检索能力。

使用方式：
    vstore = VectorStoreManager(config)
    vstore.initialize()                         # 首次启动时构建索引
    results = vstore.retrieve("Python 后端开发", k=5)  # 检索相关文档
"""

import hashlib
import json
import os
import shutil
from datetime import datetime

try:
    from langchain_chroma import Chroma
except ModuleNotFoundError:
    from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.rag.document_loader import load_and_split_documents


class VectorStoreManager:
    """ChromaDB 向量存储管理器。

    属性:
        config: 全局配置字典。
        vectorstore: Chroma 向量库实例（懒加载）。
    """

    def __init__(self, config: dict):
        self.config = config
        self.embedding_model = config["embedding_model"]
        self.kb_dir = config["knowledge_base_dir"]
        self.top_k = config["rag_top_k"]
        self.chunk_size = config["rag_chunk_size"]
        self.chunk_overlap = config["rag_chunk_overlap"]
        # ChromaDB 持久化目录。不同 embedding/chunk 配置使用不同子目录，避免维度冲突。
        self.persist_root = os.path.join(self.kb_dir, "..", "chroma_store")
        self.persist_dir = os.path.join(self.persist_root, self._index_fingerprint())
        self.meta_path = os.path.join(self.persist_dir, "index_meta.json")
        self.vectorstore: Chroma | None = None
        self._keyword_docs: list[Document] | None = None
        self._tfidf_vectorizer: TfidfVectorizer | None = None
        self._tfidf_matrix = None
        self.last_hits: list[dict] = []

    def initialize(self) -> None:
        """初始化向量库：加载文档 → 切分 → Embedding → 构建 ChromaDB 索引。

        仅首次调用时执行。若已有持久化数据则直接加载，避免重复 Embedding。
        """
        if self.vectorstore is not None:
            return

        try:
            # 尝试加载已有数据，避免每次启动都重新 Embedding
            if os.path.exists(os.path.join(self.persist_dir, "chroma.sqlite3")):
                if not self._is_index_compatible():
                    self._reset_persist_dir()
                else:
                    self.vectorstore = Chroma(
                        embedding_function=self.embedding_model,
                        persist_directory=self.persist_dir,
                        collection_name="knowledge_base",
                    )
                    if self.vectorstore._collection.count() > 0:
                        return  # 已有数据，直接复用
        except Exception:
            self._reset_persist_dir()  # 数据损坏或不存在，重新构建

        chunks = load_and_split_documents(
            self.kb_dir,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )

        os.makedirs(self.persist_dir, exist_ok=True)

        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embedding_model,
            persist_directory=self.persist_dir,
            collection_name="knowledge_base",
        )
        self._write_index_meta(chunks)

    def retrieve(self, query: str, k: int | None = None) -> str:
        """混合检索：向量召回 + 关键词召回 + RRF 融合排序。

        Args:
            query: 查询文本。
            k: 返回条数，默认使用 rag_top_k 配置。

        Returns:
            str: 拼接后的相关文本块。若检索失败则返回空字符串。
        """
        if self.vectorstore is None:
            try:
                self.initialize()
            except Exception:
                return ""

        k = k or self.top_k
        try:
            docs = self.hybrid_retrieve(query, k=k)
            self.last_hits = self.format_hits(docs)
            if not docs:
                return ""
            snippets = []
            for i, doc in enumerate(docs, 1):
                content = doc.page_content.strip()
                if content:
                    source = doc.metadata.get("hybrid_source", "hybrid")
                    snippets.append(f"[来源 {i} | {source}]: {content}")
            return "\n\n".join(snippets)
        except Exception as e:
            print(f"RAG 检索失败：{e}")
            self.last_hits = []
            return ""

    def retrieve_with_hits(self, query: str, k: int | None = None) -> tuple[str, list[dict]]:
        """Retrieve text plus structured hit metadata for UI observability."""
        text = self.retrieve(query, k=k)
        return text, list(self.last_hits)

    def format_hits(self, docs: list[Document]) -> list[dict]:
        """Convert retrieved documents into UI-friendly hit records."""
        hits = []
        for index, doc in enumerate(docs, 1):
            metadata = doc.metadata or {}
            source_path = metadata.get("source", "")
            hits.append(
                {
                    "rank": index,
                    "source": os.path.basename(str(source_path)) if source_path else "knowledge_base",
                    "source_path": str(source_path),
                    "retrieval": metadata.get("hybrid_source", "hybrid"),
                    "score": metadata.get("hybrid_score"),
                    "content": doc.page_content.strip()[:500],
                }
            )
        return hits

    def hybrid_retrieve(self, query: str, k: int | None = None) -> list[Document]:
        """Return top documents using vector and keyword retrieval fusion."""
        k = k or self.top_k
        recall_k = max(k * 3, k, 6)

        vector_docs = self.retrieve_vector(query, k=recall_k)
        keyword_docs = self.retrieve_keyword(query, k=recall_k)

        fused = self._rrf_fuse(
            ranked_lists=[
                ("vector", vector_docs),
                ("keyword", keyword_docs),
            ],
            k=k,
        )
        return fused or vector_docs[:k] or keyword_docs[:k]

    def retrieve_vector(self, query: str, k: int | None = None) -> list[Document]:
        """Vector semantic retrieval with Chroma."""
        if self.vectorstore is None:
            self.initialize()
        if self.vectorstore is None:
            return []
        try:
            return self.vectorstore.similarity_search(query, k=k or self.top_k)
        except Exception as e:
            print(f"向量检索失败：{e}")
            return []

    def retrieve_keyword(self, query: str, k: int | None = None) -> list[Document]:
        """Local keyword retrieval using char n-gram TF-IDF."""
        k = k or self.top_k
        try:
            self._ensure_keyword_index()
            if not self._keyword_docs or self._tfidf_vectorizer is None or self._tfidf_matrix is None:
                return []

            query_vec = self._tfidf_vectorizer.transform([query])
            scores = cosine_similarity(query_vec, self._tfidf_matrix).ravel()
            ranked = scores.argsort()[::-1]

            docs = []
            for idx in ranked[:k]:
                if scores[idx] <= 0:
                    continue
                docs.append(self._keyword_docs[int(idx)])
            return docs
        except Exception as e:
            print(f"关键词检索失败：{e}")
            return []

    def is_ready(self) -> bool:
        """检查向量库是否已初始化且可用。"""
        return self.vectorstore is not None

    def _ensure_keyword_index(self) -> None:
        if self._keyword_docs is not None:
            return

        self._keyword_docs = load_and_split_documents(
            self.kb_dir,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        texts = [doc.page_content for doc in self._keyword_docs if doc.page_content.strip()]
        if not texts:
            self._tfidf_vectorizer = None
            self._tfidf_matrix = None
            return

        self._tfidf_vectorizer = TfidfVectorizer(
            analyzer="char",
            ngram_range=(2, 4),
            max_features=5000,
        )
        self._tfidf_matrix = self._tfidf_vectorizer.fit_transform(texts)

    def _rrf_fuse(self, ranked_lists: list[tuple[str, list[Document]]], k: int) -> list[Document]:
        rrf_k = int(self.config.get("rag_rrf_k", 60))
        scores: dict[str, float] = {}
        docs_by_id: dict[str, Document] = {}
        sources_by_id: dict[str, set[str]] = {}

        for source, docs in ranked_lists:
            for rank, doc in enumerate(docs, 1):
                doc_id = self._doc_id(doc)
                docs_by_id.setdefault(doc_id, doc)
                sources_by_id.setdefault(doc_id, set()).add(source)
                scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (rrf_k + rank)

        ranked_ids = sorted(scores, key=scores.get, reverse=True)
        results = []
        for doc_id in ranked_ids[:k]:
            doc = docs_by_id[doc_id]
            metadata = dict(doc.metadata or {})
            metadata["hybrid_score"] = round(scores[doc_id], 6)
            metadata["hybrid_source"] = "+".join(sorted(sources_by_id[doc_id]))
            results.append(Document(page_content=doc.page_content, metadata=metadata))
        return results

    def _current_index_meta(self, dimension: int | None = None, doc_count: int | None = None) -> dict:
        return {
            "embedding_model": str(self.config.get("embedding_model_name", "")),
            "embedding_base_url": str(self.config.get("embedding_base_url", "")),
            "dimension": dimension,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "collection_name": "knowledge_base",
            "doc_count": doc_count,
            "fingerprint": self._index_fingerprint(),
        }

    def _is_index_compatible(self) -> bool:
        existing = self._read_index_meta()
        if not existing:
            return False

        current = self._current_index_meta()
        keys = ["embedding_model", "embedding_base_url", "chunk_size", "chunk_overlap", "collection_name"]
        for key in keys:
            if existing.get(key) != current.get(key):
                return False

        expected_dimension = existing.get("dimension")
        if expected_dimension:
            actual_dimension = self._probe_embedding_dimension()
            if actual_dimension and int(expected_dimension) != int(actual_dimension):
                return False

        return True

    def _read_index_meta(self) -> dict:
        if not os.path.exists(self.meta_path):
            return {}
        try:
            with open(self.meta_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def _write_index_meta(self, chunks: list[Document]) -> None:
        os.makedirs(self.persist_dir, exist_ok=True)
        dimension = self._probe_embedding_dimension()
        meta = self._current_index_meta(dimension=dimension, doc_count=len(chunks))
        meta["built_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    def _probe_embedding_dimension(self) -> int | None:
        try:
            vector = self.embedding_model.embed_query("dimension probe")
            return len(vector)
        except Exception:
            return None

    def _reset_persist_dir(self) -> None:
        persist_dir = os.path.abspath(self.persist_dir)
        persist_root = os.path.abspath(self.persist_root)
        if not persist_dir.startswith(persist_root):
            raise ValueError(f"拒绝清理知识库目录之外的索引：{persist_dir}")
        if os.path.isdir(persist_dir):
            shutil.rmtree(persist_dir)
        os.makedirs(persist_dir, exist_ok=True)
        self.vectorstore = None

    def _index_fingerprint(self) -> str:
        payload = {
            "embedding_model": str(self.config.get("embedding_model_name", "")),
            "embedding_base_url": str(self.config.get("embedding_base_url", "")),
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "collection_name": "knowledge_base",
        }
        raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]

    @staticmethod
    def _doc_id(doc: Document) -> str:
        source = str(doc.metadata.get("source", ""))
        start = str(doc.metadata.get("start_index", ""))
        content = doc.page_content.strip()
        raw = f"{source}:{start}:{content[:300]}"
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()
