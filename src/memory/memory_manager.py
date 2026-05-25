"""
长期记忆管理器 —— 基于 ChromaDB 向量库存储面试历史摘要，
支持跨会话的记忆存取。

原理：每次面试结束后，将评估摘要 + 时间戳存入 ChromaDB。
下次面试时，检索相关历史记录注入到 System Prompt 中。
"""

import os
from datetime import datetime

from langchain_core.documents import Document
try:
    from langchain_chroma import Chroma
except ModuleNotFoundError:
    from langchain_community.vectorstores import Chroma


class MemoryManager:
    """长期记忆管理器。

    属性:
        config: 全局配置字典。
        store_dir: ChromaDB 持久化目录。
        vectorstore: Chroma 向量库实例。
    """

    def __init__(self, config: dict):
        self.config = config
        self.embedding_model = config["embedding_model"]
        self.store_dir = config["memory_store_dir"]
        self.top_k = config["memory_top_k"]
        self.vectorstore: Chroma | None = None
        self.collection_name = "interview_memory"

        # 确保存储目录存在
        os.makedirs(self.store_dir, exist_ok=True)

    def save_memory(
        self,
        session_summary: str,
        scores: dict | None = None,
    ) -> None:
        """保存一段面试记忆。

        Args:
            session_summary: 本次面试的总结文本。
            scores: 可选的评分字典。
        """
        scores = scores or {}

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        metadata = {"timestamp": timestamp, **scores}
        memory_text = f"[{timestamp}] {session_summary}"

        doc = Document(page_content=memory_text, metadata=metadata)

        try:
            if self.vectorstore is None:
                self.vectorstore = Chroma.from_documents(
                    documents=[doc],
                    embedding=self.embedding_model,
                    persist_directory=self.store_dir,
                    collection_name=self.collection_name,
                )
            else:
                self.vectorstore.add_documents([doc])
        except Exception as e:
            print(f"记忆保存失败：{e}")

    def load_memory(self, query: str = "") -> str:
        """加载相关历史记忆。

        Args:
            query: 检索查询文本，留空则返回最近记忆。

        Returns:
            str: 拼接后的历史记忆文本。
        """
        if self.vectorstore is None:
            # 尝试从磁盘加载已有集合
            try:
                self.vectorstore = Chroma(
                    embedding_function=self.embedding_model,
                    persist_directory=self.store_dir,
                    collection_name=self.collection_name,
                )
            except Exception as e:
                print(f"记忆加载失败：{e}")
                return ""

        # 检查集合中是否有数据
        try:
            count = self.vectorstore._collection.count()
            if count == 0:
                return ""
        except Exception:
            return ""

        query = query or "面试表现 技术能力 沟通"
        try:
            docs = self.vectorstore.similarity_search(query, k=self.top_k)
            if not docs:
                return ""

            memories = []
            for doc in docs:
                ts = doc.metadata.get("timestamp", "未知时间")
                memories.append(f"- {doc.page_content}")

            return "## 历史面试记录\n" + "\n".join(memories)
        except Exception as e:
            print(f"记忆检索失败：{e}")
            return ""
