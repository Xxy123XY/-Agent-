"""Vector-backed interview question bank with hybrid retrieval.

The bank stores reusable interview questions in Chroma, then retrieves them with
vector similarity plus local TF-IDF keyword matching. It is intentionally
separate from the document RAG store so generated/web-discovered questions can
evolve without polluting the knowledge-base index.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    from langchain_chroma import Chroma
except ModuleNotFoundError:
    from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class InterviewQuestionBankManager:
    """Manage a persistent, hybrid-searchable interview question bank."""

    COLLECTION_NAME = "interview_question_bank"

    def __init__(self, config: dict):
        self.config = config
        self.embedding_model = config["embedding_model"]
        self.max_size = int(config.get("interview_bank_max_size", 1000))
        self.top_k = int(config.get("interview_bank_top_k", 6))
        self.rrf_k = int(config.get("interview_bank_rrf_k", config.get("rag_rrf_k", 60)))

        root = Path(config.get("interview_bank_dir") or Path(config["knowledge_base_dir"]).parent / "interview_question_bank")
        self.persist_root = str(root)
        self.persist_dir = str(root / self._index_fingerprint())
        self.meta_path = os.path.join(self.persist_dir, "index_meta.json")

        self.vectorstore: Chroma | None = None
        self._keyword_docs: list[Document] | None = None
        self._tfidf_vectorizer: TfidfVectorizer | None = None
        self._tfidf_matrix = None

    def initialize(self) -> None:
        """Load or create the Chroma collection."""
        if self.vectorstore is not None:
            return

        os.makedirs(self.persist_dir, exist_ok=True)
        self.vectorstore = Chroma(
            embedding_function=self.embedding_model,
            persist_directory=self.persist_dir,
            collection_name=self.COLLECTION_NAME,
        )
        self._write_index_meta()

    def retrieve_questions(self, query: str, k: int | None = None) -> tuple[str, list[dict]]:
        """Return formatted question text plus structured hit records."""
        docs = self.hybrid_retrieve(query, k=k or self.top_k)
        hits = self.format_hits(docs)
        if not docs:
            return "", hits

        lines = []
        for index, doc in enumerate(docs, 1):
            meta = doc.metadata or {}
            points = meta.get("expected_points", "")
            suffix = f"；考察点：{points}" if points else ""
            lines.append(
                f"[题库 {index} | {meta.get('hybrid_source', 'hybrid')} | "
                f"{meta.get('category', '未分类')}] {doc.page_content.strip()}{suffix}"
            )
        return "\n".join(lines), hits

    def hybrid_retrieve(self, query: str, k: int | None = None) -> list[Document]:
        """Retrieve questions with vector + keyword RRF fusion."""
        if not query.strip():
            return []

        k = k or self.top_k
        recall_k = max(k * 3, k, 8)
        vector_docs = self.retrieve_vector(query, k=recall_k)
        keyword_docs = self.retrieve_keyword(query, k=recall_k)
        fused = self._rrf_fuse([("vector", vector_docs), ("keyword", keyword_docs)], k=k)
        return fused or vector_docs[:k] or keyword_docs[:k]

    def retrieve_vector(self, query: str, k: int | None = None) -> list[Document]:
        """Semantic retrieval from Chroma."""
        try:
            self.initialize()
            if not self.vectorstore or self._count() == 0:
                return []
            return self.vectorstore.similarity_search(query, k=k or self.top_k)
        except Exception as e:
            print(f"面试题库向量检索失败：{e}")
            return []

    def retrieve_keyword(self, query: str, k: int | None = None) -> list[Document]:
        """Keyword retrieval using char n-gram TF-IDF over stored questions."""
        try:
            self._ensure_keyword_index()
            if not self._keyword_docs or self._tfidf_vectorizer is None or self._tfidf_matrix is None:
                return []
            query_vec = self._tfidf_vectorizer.transform([query])
            scores = cosine_similarity(query_vec, self._tfidf_matrix).ravel()
            ranked = scores.argsort()[::-1]
            docs = []
            for idx in ranked[: k or self.top_k]:
                if scores[idx] <= 0:
                    continue
                docs.append(self._keyword_docs[int(idx)])
            return docs
        except Exception as e:
            print(f"面试题库关键词检索失败：{e}")
            return []

    def add_questions(self, questions: list[dict], default_source: str = "web_search") -> int:
        """Insert new questions and prune the bank if it exceeds max_size."""
        if not questions:
            return 0

        self.initialize()
        if self.vectorstore is None:
            return 0

        existing_ids = set(self._all_ids())
        documents: list[Document] = []
        ids: list[str] = []

        for item in questions:
            question = str(item.get("question", "")).strip()
            if not question or len(question) < 8:
                continue

            question_id = item.get("question_id") or self._question_id(question)
            if question_id in existing_ids:
                continue

            expected_points = item.get("expected_points", [])
            if isinstance(expected_points, list):
                expected_points_text = "；".join(str(p) for p in expected_points if str(p).strip())
            else:
                expected_points_text = str(expected_points or "")

            metadata = {
                "question_id": question_id,
                "category": str(item.get("category") or "技术基础"),
                "topic": str(item.get("topic") or ""),
                "source": str(item.get("source") or default_source),
                "source_url": str(item.get("source_url") or ""),
                "expected_points": expected_points_text,
                "quality_score": float(item.get("quality_score") or 0.7),
                "created_at": str(item.get("created_at") or self._now()),
                "last_used_at": str(item.get("last_used_at") or ""),
                "used_count": int(item.get("used_count") or 0),
                "hash": hashlib.sha1(question.encode("utf-8")).hexdigest(),
            }
            documents.append(Document(page_content=question, metadata=metadata))
            ids.append(question_id)
            existing_ids.add(question_id)

        if not documents:
            return 0

        self.vectorstore.add_documents(documents, ids=ids)
        self._invalidate_keyword_index()
        self.prune_if_needed()
        self._write_index_meta()
        return len(documents)

    def add_web_search_results(self, results: list[dict], topic: str = "") -> int:
        """Extract question-like lines from web snippets and store them."""
        candidates: list[dict] = []
        for result in results or []:
            title = str(result.get("title") or "")
            body = str(result.get("body") or "")
            href = str(result.get("href") or "")
            for question in self.extract_questions_from_text(f"{title}\n{body}"):
                candidates.append(
                    {
                        "question": question,
                        "category": "技术基础",
                        "topic": topic,
                        "source": "网上真实面经",
                        "source_url": href,
                        "quality_score": 0.65,
                    }
                )
        return self.add_questions(candidates, default_source="网上真实面经")

    def extract_questions_from_text(self, text: str) -> list[str]:
        """Best-effort extraction of interview-question-like sentences."""
        cleaned = re.sub(r"\s+", " ", text or "").strip()
        if not cleaned:
            return []

        patterns = [
            r"[^。！？?]*?(?:什么|如何|怎么|为什么|区别|原理|流程|机制|场景|优化|排查|设计)[^。！？?]{4,80}[？?]",
            r"[^。！？?]{4,80}(?:是什么|有哪些|怎么做|如何实现|如何优化|如何排查|有什么区别)",
        ]
        found: list[str] = []
        for pattern in patterns:
            for match in re.findall(pattern, cleaned):
                question = match.strip(" -:：;；,.，。")
                if question and not question.endswith(("?", "？")):
                    question = f"{question}？"
                if 8 <= len(question) <= 120:
                    found.append(question)

        deduped = []
        seen = set()
        for question in found:
            key = self._question_id(question)
            if key not in seen:
                deduped.append(question)
                seen.add(key)
        return deduped[:20]

    def prune_if_needed(self) -> int:
        """Delete old/low-value questions when the bank exceeds max_size."""
        self.initialize()
        count = self._count()
        if count <= self.max_size:
            return 0

        overflow = count - self.max_size
        records = self._get_all(include_documents=False)
        ids = records.get("ids", [])
        metadatas = records.get("metadatas", [])
        ranked = sorted(
            zip(ids, metadatas),
            key=lambda pair: (
                float((pair[1] or {}).get("quality_score") or 0),
                int((pair[1] or {}).get("used_count") or 0),
                str((pair[1] or {}).get("last_used_at") or (pair[1] or {}).get("created_at") or ""),
            ),
        )
        delete_ids = [item_id for item_id, _ in ranked[:overflow]]
        if delete_ids:
            self.vectorstore._collection.delete(ids=delete_ids)
            self._invalidate_keyword_index()
            self._write_index_meta()
        return len(delete_ids)

    def update_usage(self, question_ids: list[str]) -> None:
        """Best-effort usage tracking for retrieved questions."""
        if not question_ids:
            return
        self.initialize()
        records = self._get_all(include_documents=True)
        docs_by_id = dict(zip(records.get("ids", []), records.get("documents", [])))
        meta_by_id = dict(zip(records.get("ids", []), records.get("metadatas", [])))
        for question_id in question_ids:
            if question_id not in docs_by_id:
                continue
            metadata = dict(meta_by_id.get(question_id) or {})
            metadata["used_count"] = int(metadata.get("used_count") or 0) + 1
            metadata["last_used_at"] = self._now()
            # Only metadata changes here. Passing documents would make Chroma
            # re-embed through its default embedding function, which can create
            # dimension conflicts when the app uses a custom provider.
            self.vectorstore._collection.update(ids=[question_id], metadatas=[metadata])
        self._invalidate_keyword_index()

    def format_hits(self, docs: list[Document]) -> list[dict]:
        """Convert retrieved questions into UI-friendly hit records."""
        hits = []
        for rank, doc in enumerate(docs, 1):
            meta = doc.metadata or {}
            hits.append(
                {
                    "rank": rank,
                    "question_id": meta.get("question_id", self._question_id(doc.page_content)),
                    "question": doc.page_content.strip(),
                    "category": meta.get("category", "未分类"),
                    "topic": meta.get("topic", ""),
                    "source": meta.get("source", "题库"),
                    "source_url": meta.get("source_url", ""),
                    "retrieval": meta.get("hybrid_source", "hybrid"),
                    "score": meta.get("hybrid_score"),
                    "expected_points": meta.get("expected_points", ""),
                }
            )
        return hits

    def _ensure_keyword_index(self) -> None:
        if self._keyword_docs is not None:
            return
        self.initialize()
        records = self._get_all(include_documents=True)
        docs = []
        for document, metadata in zip(records.get("documents", []), records.get("metadatas", [])):
            if document and str(document).strip():
                docs.append(Document(page_content=str(document), metadata=metadata or {}))

        self._keyword_docs = docs
        texts = [doc.page_content for doc in docs]
        if not texts:
            self._tfidf_vectorizer = None
            self._tfidf_matrix = None
            return

        self._tfidf_vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 4), max_features=5000)
        self._tfidf_matrix = self._tfidf_vectorizer.fit_transform(texts)

    def _rrf_fuse(self, ranked_lists: list[tuple[str, list[Document]]], k: int) -> list[Document]:
        scores: dict[str, float] = {}
        docs_by_id: dict[str, Document] = {}
        sources_by_id: dict[str, set[str]] = {}

        for source, docs in ranked_lists:
            for rank, doc in enumerate(docs, 1):
                doc_id = str((doc.metadata or {}).get("question_id") or self._question_id(doc.page_content))
                docs_by_id.setdefault(doc_id, doc)
                sources_by_id.setdefault(doc_id, set()).add(source)
                scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (self.rrf_k + rank)

        results = []
        for doc_id in sorted(scores, key=scores.get, reverse=True)[:k]:
            doc = docs_by_id[doc_id]
            metadata = dict(doc.metadata or {})
            metadata["question_id"] = doc_id
            metadata["hybrid_score"] = round(scores[doc_id], 6)
            metadata["hybrid_source"] = "+".join(sorted(sources_by_id[doc_id]))
            results.append(Document(page_content=doc.page_content, metadata=metadata))
        return results

    def _get_all(self, include_documents: bool = True) -> dict[str, Any]:
        self.initialize()
        include = ["metadatas"]
        if include_documents:
            include.append("documents")
        return self.vectorstore._collection.get(include=include)

    def _all_ids(self) -> list[str]:
        self.initialize()
        return list(self.vectorstore._collection.get(include=[])["ids"])

    def _count(self) -> int:
        self.initialize()
        return int(self.vectorstore._collection.count())

    def _invalidate_keyword_index(self) -> None:
        self._keyword_docs = None
        self._tfidf_vectorizer = None
        self._tfidf_matrix = None

    def _write_index_meta(self) -> None:
        os.makedirs(self.persist_dir, exist_ok=True)
        meta = {
            "collection_name": self.COLLECTION_NAME,
            "embedding_model": str(self.config.get("embedding_model_name", "")),
            "embedding_base_url": str(self.config.get("embedding_base_url", "")),
            "max_size": self.max_size,
            "doc_count": self._safe_count(),
            "fingerprint": self._index_fingerprint(),
            "updated_at": self._now(),
        }
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    def _safe_count(self) -> int:
        try:
            return self._count()
        except Exception:
            return 0

    def _index_fingerprint(self) -> str:
        payload = {
            "embedding_model": str(self.config.get("embedding_model_name", "")),
            "embedding_base_url": str(self.config.get("embedding_base_url", "")),
            "collection_name": self.COLLECTION_NAME,
        }
        raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]

    @staticmethod
    def _question_id(question: str) -> str:
        normalized = re.sub(r"\s+", "", question.strip().lower())
        return hashlib.sha1(normalized.encode("utf-8")).hexdigest()

    @staticmethod
    def _now() -> str:
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
