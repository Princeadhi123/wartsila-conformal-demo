"""Knowledge base loader and TF-IDF retrieval for the demo RAG layer."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

DISCLAIMER = (
    "> Illustrative synthetic document written for this demo from general, "
    "publicly known engineering practice. Not Wärtsilä documentation."
)

_FM_RE = re.compile(r"\A---[ \t]*\n(.*?)\n---[ \t]*\n?", re.DOTALL)
_SECTION_RE = re.compile(r"^## +", re.MULTILINE)


@dataclass
class Hit:
    doc_id: str
    title: str
    doc_type: str
    domain: str
    section: str
    text: str
    score: float


def _parse_front_matter(text: str) -> tuple[dict, str]:
    """Hand-parse the leading ``--- ... ---`` YAML-ish block (no PyYAML)."""
    m = _FM_RE.match(text)
    if not m:
        return {}, text
    meta: dict = {}
    for line in m.group(1).splitlines():
        line = line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, _, val = line.partition(":")
        key = key.strip()
        val = val.strip()
        # strip an optional trailing "# comment"
        if "#" in val:
            val = val.split("#", 1)[0].strip()
        if val.startswith("[") and val.endswith("]"):
            meta[key] = [t.strip().strip("\"'") for t in val[1:-1].split(",") if t.strip()]
        else:
            meta[key] = val.strip("\"'")
    return meta, text[m.end():]


def _split_sections(body: str) -> list[tuple[str, str]]:
    """Split markdown body into (heading, text) pairs at ``## `` boundaries."""
    matches = list(_SECTION_RE.finditer(body))
    sections: list[tuple[str, str]] = []
    if not matches:
        cleaned = body.strip()
        return [("", cleaned)] if cleaned else []
    pre = body[: matches[0].start()].strip()
    if pre:
        sections.append(("", pre))
    for i, m in enumerate(matches):
        heading_line_end = body.index("\n", m.end()) if "\n" in body[m.end():] else len(body)
        heading = body[m.end():heading_line_end].strip()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
        text = body[heading_line_end:end].strip()
        if text:
            sections.append((heading, text))
    return sections


class KnowledgeBase:
    """Markdown knowledge base with per-`## `-section TF-IDF retrieval."""

    def __init__(self, kb_dir: str | Path):
        self.kb_dir = Path(kb_dir)
        self.docs: dict[str, dict] = {}
        self.chunks: list[dict] = []
        for path in sorted(self.kb_dir.glob("*.md")):
            raw = path.read_text(encoding="utf-8")
            meta, body = _parse_front_matter(raw)
            doc_id = meta.get("id") or path.stem
            tags = meta.get("tags") or []
            if isinstance(tags, str):
                tags = [t.strip() for t in tags.split(",") if t.strip()]
            meta["tags"] = tags
            self.docs[doc_id] = {"meta": meta, "text": raw, "path": path}
            clean_body = "\n".join(
                ln for ln in body.splitlines() if ln.strip() != DISCLAIMER
            )
            for heading, text in _split_sections(clean_body):
                self.chunks.append(
                    {
                        "doc_id": doc_id,
                        "title": meta.get("title", doc_id),
                        "doc_type": meta.get("type", "manual"),
                        "domain": meta.get("domain", ""),
                        "section": heading,
                        "text": text,
                    }
                )
        corpus = [
            " ".join([c["title"], " ".join(self.docs[c["doc_id"]]["meta"]["tags"]),
                      c["section"], c["text"]])
            for c in self.chunks
        ]
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2), sublinear_tf=True, stop_words="english"
        )
        self.matrix = self.vectorizer.fit_transform(corpus) if corpus else None

    @property
    def n_docs(self) -> int:
        return len(self.docs)

    @property
    def n_chunks(self) -> int:
        return len(self.chunks)

    def get_doc(self, doc_id: str) -> dict:
        d = self.docs[doc_id]
        return {"id": doc_id, **d["meta"], "text": d["text"], "path": str(d["path"])}

    def retrieve(
        self,
        query: str,
        k: int = 4,
        domain: str | None = None,
        doc_types: list[str] | None = None,
    ) -> list[Hit]:
        """Cosine-similarity retrieval; filters applied before ranking."""
        if not self.chunks:
            return []
        idx = [
            i
            for i, c in enumerate(self.chunks)
            if (domain is None or c["domain"] == domain)
            and (doc_types is None or c["doc_type"] in doc_types)
        ]
        if not idx:
            return []
        q = self.vectorizer.transform([query])
        scores = linear_kernel(q, self.matrix[idx]).ravel()
        order = sorted(range(len(idx)), key=lambda j: scores[j], reverse=True)
        hits: list[Hit] = []
        per_doc: dict[str, int] = {}
        for j in order:
            s = float(scores[j])
            if s <= 0:
                break
            c = self.chunks[idx[j]]
            if per_doc.get(c["doc_id"], 0) >= 2:
                continue
            per_doc[c["doc_id"]] = per_doc.get(c["doc_id"], 0) + 1
            hits.append(
                Hit(c["doc_id"], c["title"], c["doc_type"], c["domain"],
                    c["section"], c["text"], s)
            )
            if len(hits) >= k:
                break
        return hits
