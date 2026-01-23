from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Set

from .docstore import DocRecord, DocStore
from .wikipedia_client import WikipediaClient
from .wikidata_client import WikidataClient
from .schemas import sha256_text


@dataclass(frozen=True)
class Budgets:
    max_pages_per_question: int = 12
    max_hops: int = 2
    max_list_pages: int = 2
    max_country_pages: int = 1
    max_context_qids: int = 8
    max_outgoing: int = 8


@dataclass(frozen=True)
class RawDoc:
    doc_key: str
    title: Optional[str]
    url: Optional[str]
    html: Optional[str]
    anchors: dict
    source: str
    dataset_origin: str
    page_id: Optional[str] = None
    revision_id: Optional[str] = None


def _cached_wiki_page(
    title: str,
    source: str,
    dataset_origin: str,
    docstore: DocStore,
    wiki: WikipediaClient,
) -> RawDoc:
    cache_key = f"title:{title}"

    def _fetch() -> DocRecord:
        page = wiki.fetch_html(title)
        return DocRecord(
            title=page.title,
            page_id=page.page_id,
            revision_id=page.revision_id,
            url=page.url,
            html=page.html,
            cleaned_text=None,
            anchors={"outgoing_titles": page.outgoing_titles, "incoming_stub": []},
            source=source,
            dataset_origin=dataset_origin,
        )

    record = docstore.get_or_fetch(cache_key, _fetch)
    doc_key = record.page_id or sha256_text(record.title)
    return RawDoc(
        doc_key=doc_key,
        title=record.title,
        url=record.url,
        html=record.html,
        anchors=record.anchors,
        source=source,
        dataset_origin=dataset_origin,
        page_id=record.page_id,
        revision_id=record.revision_id,
    )


def _dedupe_docs(docs: Iterable[RawDoc]) -> List[RawDoc]:
    seen = set()
    unique: List[RawDoc] = []
    for doc in docs:
        if doc.doc_key in seen:
            continue
        seen.add(doc.doc_key)
        unique.append(doc)
    return unique


def _propose_list_pages(titles: Sequence[str], question: str, limit: int) -> List[str]:
    candidates: List[str] = []
    for title in titles:
        candidates.append(f"List of {title}")
        candidates.append(f"Timeline of {title}")
    if any(word in question.lower() for word in ["list", "timeline", "history", "year"]):
        candidates.extend([f"List of {question}", f"Timeline of {question}"])
    deduped = []
    seen = set()
    for title in candidates:
        if title not in seen:
            seen.add(title)
            deduped.append(title)
        if len(deduped) >= limit:
            break
    return deduped


def _normalize_qid(value: str) -> Optional[str]:
    if not value:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.upper().startswith("Q"):
        return text.upper()
    if text.isdigit():
        return f"Q{text}"
    return text


def ingest_complextempqa(
    sample: dict,
    budgets: Budgets,
    docstore: DocStore,
    wiki: WikipediaClient,
    wikidata: WikidataClient,
) -> List[RawDoc]:
    source = "complextempqa"
    dataset_origin = "complextempqa"
    seed_qids: Set[str] = set()
    for field in ("question_entity", "answer_entity", "question_country_entity"):
        value = sample.get(field)
        if isinstance(value, list):
            seed_qids.update(
                normalized
                for v in value
                for normalized in [_normalize_qid(v)]
                if normalized
            )
        elif value:
            normalized = _normalize_qid(value)
            if normalized:
                seed_qids.add(normalized)

    pages: Set[str] = set()
    for qid in seed_qids:
        title = wikidata.get_wikipedia_title(qid)
        if title:
            pages.add(title)
        if len(pages) >= budgets.max_pages_per_question:
            break

    context_props = ["P17", "P131", "P463", "P361", "P571", "P585"]
    context_qids: Set[str] = set()
    for qid in seed_qids:
        context_qids.update(
            wikidata.get_claim_qids(qid, context_props, limit=budgets.max_context_qids)
        )
        if len(context_qids) >= budgets.max_context_qids:
            break

    for qid in list(context_qids):
        if len(pages) >= budgets.max_pages_per_question:
            break
        title = wikidata.get_wikipedia_title(qid)
        if title:
            pages.add(title)

    if budgets.max_country_pages > 0:
        for qid in list(seed_qids):
            for country_qid in wikidata.get_claim_qids(qid, ["P17"], limit=1):
                if len(pages) >= budgets.max_pages_per_question:
                    break
                title = wikidata.get_wikipedia_title(country_qid)
                if title:
                    pages.add(title)
            if len(pages) >= budgets.max_pages_per_question:
                break

    question_text = sample.get("question", "")
    list_pages = _propose_list_pages(list(pages), question_text, budgets.max_list_pages)
    for title in list_pages:
        if len(pages) >= budgets.max_pages_per_question:
            break
        pages.add(title)

    docs = [
        _cached_wiki_page(title, source, dataset_origin, docstore, wiki)
        for title in pages
    ]
    return _dedupe_docs(docs)


def ingest_wikiwhy(
    sample: dict,
    budgets: Budgets,
    docstore: DocStore,
    wiki: WikipediaClient,
) -> List[RawDoc]:
    source = "wikiwhy"
    dataset_origin = "wikiwhy"
    title = sample.get("title")
    if not title:
        return []

    pages: List[str] = [title]
    docs: List[RawDoc] = []
    for page_title in pages:
        docs.append(_cached_wiki_page(page_title, source, dataset_origin, docstore, wiki))
    if docs:
        outgoing = docs[0].anchors.get("outgoing_titles", [])
        for out_title in outgoing[: budgets.max_outgoing]:
            if len(docs) >= budgets.max_pages_per_question:
                break
            docs.append(
                _cached_wiki_page(out_title, source, dataset_origin, docstore, wiki)
            )

    list_pages = _propose_list_pages(
        [doc.title for doc in docs if doc.title],
        sample.get("question", ""),
        budgets.max_list_pages,
    )
    for title in list_pages:
        if len(docs) >= budgets.max_pages_per_question:
            break
        docs.append(_cached_wiki_page(title, source, dataset_origin, docstore, wiki))
    return _dedupe_docs(docs)


def ingest_nq(
    sample: dict,
    budgets: Budgets,
    docstore: DocStore,
    wiki: WikipediaClient,
) -> List[RawDoc]:
    source = "nq"
    dataset_origin = "nq"
    document = sample.get("document", {}) if isinstance(sample.get("document"), dict) else {}
    html = document.get("html") or sample.get("document_html")
    title = document.get("title") or sample.get("document_title") or sample.get("title")
    url = document.get("url")
    doc_key = sha256_text(html or title or sample.get("id", "nq"))
    docs = [
        RawDoc(
            doc_key=doc_key,
            title=title,
            url=url,
            html=html,
            anchors={"outgoing_titles": [], "incoming_stub": []},
            source=source,
            dataset_origin=dataset_origin,
            page_id=None,
            revision_id=None,
        )
    ]

    if title and budgets.max_pages_per_question > 1:
        outgoing = wiki.fetch_html(title).outgoing_titles
        for out_title in outgoing[: budgets.max_outgoing]:
            if len(docs) >= budgets.max_pages_per_question:
                break
            docs.append(
                _cached_wiki_page(out_title, source, dataset_origin, docstore, wiki)
            )
    return _dedupe_docs(docs)
