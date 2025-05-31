import os
from typing import List, Dict, Tuple

import faiss
import numpy as np
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer


class PDFRetriever:
    def __init__(self, pdf_dir: str, embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.pdf_dir = pdf_dir
        self.model = SentenceTransformer(embedding_model)
        self.docs: List[Dict[str, str]] = []
        self.index = None
        self._load_documents()
        self._build_index()

    def _load_documents(self) -> None:
        for name in os.listdir(self.pdf_dir):
            if not name.lower().endswith(".pdf"):
                continue
            path = os.path.join(self.pdf_dir, name)
            try:
                reader = PdfReader(path)
                text = "\n".join(page.extract_text() or "" for page in reader.pages)
                self.docs.append({"text": text, "source": name})
            except Exception:
                # ignore unreadable PDFs
                continue

    def _build_index(self) -> None:
        if not self.docs:
            return
        embeddings = self.model.encode([d["text"] for d in self.docs])
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(np.array(embeddings).astype("float32"))

    def query(self, question: str, k: int = 3) -> List[Tuple[str, str]]:
        if self.index is None:
            return []
        q_emb = self.model.encode([question])
        distances, indices = self.index.search(np.array(q_emb).astype("float32"), k)
        results = []
        for idx in indices[0]:
            doc = self.docs[idx]
            results.append((doc["text"], doc["source"]))
        return results
