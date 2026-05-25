"""
文件存储与缓存管理器 —— 上传文件永久保存 + 分析结果缓存。

- 文件存储：data/uploads/{jd,resume}/{hash}_{name}
- 缓存索引：data/uploads/cache.json（file_hash → structured_jd）
- 相同内容（SHA256 一致）自动去重，直接复用缓存
"""

import os
import json
import hashlib
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPLOADS_DIR = os.path.join(PROJECT_ROOT, "data", "uploads")
CACHE_FILE = os.path.join(UPLOADS_DIR, "cache.json")


def _ensure_dirs():
    os.makedirs(os.path.join(UPLOADS_DIR, "jd"), exist_ok=True)
    os.makedirs(os.path.join(UPLOADS_DIR, "resume"), exist_ok=True)


def _load_cache() -> dict:
    """加载缓存索引。"""
    if not os.path.exists(CACHE_FILE):
        return {}
    try:
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        # 清理不存在的文件引用
        cleaned = {}
        for h, entry in data.items():
            file_path = entry.get("file_path")
            if not file_path or os.path.exists(file_path):
                cleaned[h] = entry
        return cleaned
    except Exception:
        return {}


def _save_cache(cache: dict):
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def _hash_content(content: str | bytes) -> str:
    """计算 SHA256。"""
    if isinstance(content, str):
        content = content.encode("utf-8")
    return hashlib.sha256(content).hexdigest()[:16]


# ── 文件存储 ──

def save_file(content: str | bytes, filename: str, category: str) -> dict:
    """保存文本/字节到磁盘。

    Args:
        content: 文件文本或字节。
        filename: 原始文件名。
        category: "jd" 或 "resume"。

    Returns:
        {"hash": str, "path": str, "name": str, "size": int, "saved_at": str}
    """
    _ensure_dirs()
    if isinstance(content, str):
        content = content.encode("utf-8")
    file_hash = _hash_content(content)
    safe_name = f"{file_hash}_{filename}"

    target_dir = os.path.join(UPLOADS_DIR, category)
    target_path = os.path.join(target_dir, safe_name)

    # 同 hash 文件不重复写
    if not os.path.exists(target_path):
        with open(target_path, "wb") as f:
            f.write(content)

    return {
        "hash": file_hash,
        "path": target_path,
        "name": filename,
        "size": len(content),
        "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }


def list_files(category: str) -> list[dict]:
    """列出某类别下所有已存储文件。

    Returns:
        [{"hash": str, "name": str, "path": str, "size": int, "saved_at": str}, ...]
    """
    _ensure_dirs()
    target_dir = os.path.join(UPLOADS_DIR, category)
    files = []
    for fname in sorted(os.listdir(target_dir), reverse=True):
        fpath = os.path.join(target_dir, fname)
        if os.path.isfile(fpath):
            parts = fname.split("_", 1)
            file_hash = parts[0] if parts else ""
            original_name = parts[1] if len(parts) > 1 else fname
            files.append({
                "hash": file_hash,
                "name": original_name,
                "path": fpath,
                "size": os.path.getsize(fpath),
                "saved_at": datetime.fromtimestamp(os.path.getmtime(fpath)).strftime("%Y-%m-%d %H:%M"),
            })
    return files


def read_file(file_hash: str, category: str) -> str | None:
    """根据 hash 读取文件内容。"""
    target_dir = os.path.join(UPLOADS_DIR, category)
    for fname in os.listdir(target_dir):
        if fname.startswith(file_hash):
            fpath = os.path.join(target_dir, fname)
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    return f.read()
            except (UnicodeDecodeError, UnicodeError):
                with open(fpath, "rb") as f:
                    raw = f.read()
                try:
                    return raw.decode("utf-8")
                except UnicodeDecodeError:
                    try:
                        return raw.decode("gbk")
                    except UnicodeDecodeError:
                        return raw.decode("utf-8", errors="replace")
    return None


def delete_file(file_hash: str, category: str) -> bool:
    """根据 hash 删除已保存文件，并清理对应缓存。

    Args:
        file_hash: 文件内容 hash。
        category: "jd" 或 "resume"。

    Returns:
        bool: 是否删除了至少一个文件。
    """
    if category not in {"jd", "resume"}:
        raise ValueError("category 只能是 jd 或 resume")
    if not file_hash:
        return False

    _ensure_dirs()
    target_dir = os.path.abspath(os.path.join(UPLOADS_DIR, category))
    uploads_root = os.path.abspath(UPLOADS_DIR)
    if not target_dir.startswith(uploads_root):
        raise ValueError("拒绝删除 uploads 目录之外的文件")

    deleted = False
    for fname in list(os.listdir(target_dir)):
        if not fname.startswith(file_hash):
            continue
        fpath = os.path.abspath(os.path.join(target_dir, fname))
        if not fpath.startswith(target_dir):
            raise ValueError("拒绝删除目标目录之外的文件")
        if os.path.isfile(fpath):
            os.remove(fpath)
            deleted = True

    cache = _load_cache()
    if file_hash in cache:
        cache.pop(file_hash, None)
        _save_cache(cache)

    return deleted


# ── 分析缓存 ──

def get_cached_analysis(file_hash: str) -> dict | None:
    """获取缓存的 JD 分析结果。"""
    cache = _load_cache()
    return cache.get(file_hash, {}).get("result")


def save_cached_analysis(file_hash: str, file_path: str | None, result: dict):
    """保存 JD 分析结果到缓存。"""
    cache = _load_cache()
    cache[file_hash] = {
        "file_path": file_path,
        "analyzed_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "result": result,  # structured_jd dict
    }
    _save_cache(cache)


def is_cached(file_hash: str) -> bool:
    """检查是否已有分析缓存。"""
    return get_cached_analysis(file_hash) is not None


def list_cached() -> list[dict]:
    """列出所有已缓存的分析记录。"""
    cache = _load_cache()
    result = []
    for h, entry in cache.items():
        result.append({
            "hash": h,
            "analyzed_at": entry.get("analyzed_at", ""),
            "role_summary": entry.get("result", {}).get("role_summary", "") if entry.get("result") else "",
        })
    return result
