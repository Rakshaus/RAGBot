from pathlib import Path
from langchain_core.documents import Document


def load_document(file_path="sample.txt"):
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Document not found: {file_path}")

    text = path.read_text(encoding="utf-8")

    if not text.strip():
        raise ValueError("The document is empty.")

    return [
        Document(
            page_content=text,
            metadata={
                "source": path.name
            }
        )
    ]