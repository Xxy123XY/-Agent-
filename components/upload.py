"""文件解析工具 —— 从 PDF / DOCX / DOC / TXT / MD 中提取文本。"""
import io, os, re, tempfile


def _extract_docx_text(file_bytes: bytes) -> str:
    from docx import Document
    doc = Document(io.BytesIO(file_bytes))
    return "\n".join(p.text.strip() for p in doc.paragraphs if p.text.strip())


def _extract_doc_text(file_bytes: bytes) -> str:
    try:
        return _extract_docx_text(file_bytes)
    except Exception:
        pass
    try:
        tmp = tempfile.NamedTemporaryFile(suffix=".doc", delete=False)
        tmp.write(file_bytes)
        tmp.close()
        try:
            import win32com.client
            word = win32com.client.Dispatch("Word.Application")
            word.Visible = False
            doc = word.Documents.Open(tmp.name)
            text = doc.Content.Text
            doc.Close()
            word.Quit()
            return text.strip()
        finally:
            os.unlink(tmp.name)
    except Exception:
        pass
    try:
        import olefile
        ole = olefile.OleFileIO(io.BytesIO(file_bytes))
        if ole.exists("WordDocument"):
            raw = ole.openstream("WordDocument").read()
            text = raw.decode("utf-8", errors="replace")
            cleaned = re.sub(r'[^\x20-\x7e一-鿿　-〿＀-￯\n\r\t]', '', text)
            ole.close()
            if len(cleaned.strip()) > 100:
                return cleaned.strip()
        ole.close()
    except Exception:
        pass
    return "⚠️ 无法解析 .doc 文件。请用 Word 另存为 .docx 格式后重新上传。"


def parse_uploaded_file(uploaded_file) -> str:
    filename = (uploaded_file.name or "").lower()
    file_bytes = uploaded_file.read()
    if not file_bytes:
        return "⚠️ 文件为空，请重新上传。"
    try:
        if filename.endswith(".pdf"):
            from PyPDF2 import PdfReader
            reader = PdfReader(io.BytesIO(file_bytes))
            parts = []
            for page in reader.pages:
                t = page.extract_text()
                if t:
                    parts.append(t)
            r = "\n".join(parts).strip()
            return r or "⚠️ PDF 解析无文本内容，可能为扫描件或图片型 PDF。"
        elif filename.endswith(".docx"):
            r = _extract_docx_text(file_bytes)
            return r or "⚠️ DOCX 文件无文本内容。"
        elif filename.endswith(".doc"):
            r = _extract_doc_text(file_bytes)
            return r or "⚠️ DOC 文件无文本内容。"
        elif filename.endswith((".txt", ".md", ".markdown")):
            try:
                return file_bytes.decode("utf-8").strip()
            except UnicodeDecodeError:
                try:
                    return file_bytes.decode("gbk").strip()
                except UnicodeDecodeError:
                    return file_bytes.decode("utf-8", errors="replace").strip()
        else:
            return f"⚠️ 不支持的格式：.{filename.split('.')[-1]}。支持 PDF/DOCX/DOC/TXT/MD。"
    except ImportError as e:
        return f"⚠️ 缺少解析库：{e}"
    except Exception as e:
        return f"⚠️ 文件解析失败：{str(e)}"
