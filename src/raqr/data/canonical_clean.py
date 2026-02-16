from __future__ import annotations

from typing import List

from bs4 import BeautifulSoup

from .corpus_schemas import Block, StructuredDoc


DROP_TAGS = ["script", "style", "nav", "footer", "aside"]


def _table_to_text(table, max_rows: int = 30, max_cols: int = 8) -> str:
    rows = []
    for row in table.find_all("tr", recursive=False)[:max_rows]:
        cells = row.find_all(["th", "td"], recursive=False)[:max_cols]
        rows.append(" | ".join(cell.get_text(" ", strip=True) for cell in cells))
    return "\n".join(rows)


def _update_section_path(path: List[str], heading: str, level: str) -> List[str]:
    try:
        depth = int(level.replace("h", ""))
    except ValueError:
        depth = 2
    new_path = path[: max(1, depth - 1)]
    new_path.append(heading)
    return new_path


def clean_html_to_structured_doc(
    html: str,
    doc_id: str,
    title: str | None,
    url: str | None,
    anchors: dict,
    source: str,
    dataset_origin: str,
    page_id: str | None = None,
    revision_id: str | None = None,
) -> StructuredDoc:
    soup = BeautifulSoup(html or "", "lxml")
    for tag in DROP_TAGS:
        for node in soup.find_all(tag):
            node.decompose()

    blocks: List[Block] = []
    current_path = ["Lead"]

    body = soup.body if soup.body else soup
    for node in body.descendants:
        if not getattr(node, "name", None):
            continue
        if node.name in ["h1", "h2", "h3", "h4", "h5", "h6"]:
            current_path = _update_section_path(
                current_path, node.get_text(" ", strip=True), node.name
            )
        elif node.name == "p":
            text = node.get_text(" ", strip=True)
            if text:
                blocks.append(
                    Block(text=text, section_path=list(current_path), block_type="paragraph")
                )
        elif node.name in ["ul", "ol"]:
            items = [
                li.get_text(" ", strip=True)
                for li in node.find_all("li", recursive=False)
            ]
            if items:
                text = "\n".join(f"- {item}" for item in items)
                blocks.append(
                    Block(text=text, section_path=list(current_path), block_type="list")
                )
        elif node.name == "table":
            text = _table_to_text(node)
            if text:
                blocks.append(
                    Block(text=text, section_path=list(current_path), block_type="table")
                )

    return StructuredDoc(
        doc_id=doc_id,
        title=title,
        url=url,
        blocks=blocks,
        anchors=anchors or {"outgoing_titles": [], "incoming_stub": []},
        source=source,
        dataset_origin=dataset_origin,
        page_id=page_id,
        revision_id=revision_id,
    )
