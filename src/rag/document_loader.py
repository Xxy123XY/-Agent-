"""
文档加载器 —— 加载 data/knowledge_base/ 下的 txt/md 文档，
使用 RecursiveCharacterTextSplitter 切分为适合 Embedding 的块。
"""

import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_and_split_documents(
    knowledge_base_dir: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> list:
    """加载知识库目录下的所有 txt/md 文件并切分为文本块。

    Args:
        knowledge_base_dir: 知识库文件目录路径。
        chunk_size: 每块最大字符数。
        chunk_overlap: 相邻块重叠字符数。

    Returns:
        list[Document]: LangChain Document 对象列表。
    """
    if not os.path.isdir(knowledge_base_dir):
        raise FileNotFoundError(f"知识库目录不存在：{knowledge_base_dir}")

    # 使用 DirectoryLoader 批量加载 .txt 和 .md 文件
    loader = DirectoryLoader(
        knowledge_base_dir,
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        show_progress=False,
    )

    try:
        documents = loader.load()
    except Exception as e:
        print(f"文档加载警告：{e}")
        documents = []

    if not documents:
        raise ValueError(f"知识库目录 {knowledge_base_dir} 中未找到可加载的文档")

    # 切分文档
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", "，", " ", ""],
        length_function=len,
    )

    chunks = text_splitter.split_documents(documents)
    return chunks
