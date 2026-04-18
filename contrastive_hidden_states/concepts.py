from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re


@dataclass(frozen=True)
class ContrastivePair:
    category_name: str
    category_key: str
    positive: str
    negative: str
    index_within_category: int
    slug: str


def _slugify(text: str) -> str:
    text = text.strip().lower().replace("&", "and")
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def _normalize_category(raw_header: str) -> tuple[str, str]:
    header = re.sub(r"\s*\(\d+\)\s*$", "", raw_header.strip())
    key = _slugify(header)
    return header, key


def parse_contrastive_concepts(concepts_path: str | Path) -> dict[str, list[ContrastivePair]]:
    concepts_path = Path(concepts_path)
    current_category_name: str | None = None
    current_category_key: str | None = None
    counts: dict[str, int] = {}
    parsed: dict[str, list[ContrastivePair]] = {}

    for raw_line in concepts_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue

        if " vs " not in line:
            current_category_name, current_category_key = _normalize_category(line)
            parsed.setdefault(current_category_key, [])
            counts.setdefault(current_category_key, 0)
            continue

        if current_category_name is None or current_category_key is None:
            raise ValueError(
                f"Encountered concept pair before category header in {concepts_path}: {line}"
            )

        positive, negative = [part.strip() for part in line.split(" vs ", maxsplit=1)]
        counts[current_category_key] += 1
        pair_slug = f"{current_category_key}__{_slugify(positive)}__{_slugify(negative)}"
        parsed[current_category_key].append(
            ContrastivePair(
                category_name=current_category_name,
                category_key=current_category_key,
                positive=positive,
                negative=negative,
                index_within_category=counts[current_category_key],
                slug=pair_slug,
            )
        )

    return parsed

