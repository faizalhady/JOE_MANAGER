"""
CSV and Text File Processors
CSV: Similar to XLSX but for comma-separated files.
Text: Simple plain text extraction.
"""

import csv
import logging
from typing import List, Dict, Any
from pathlib import Path
from .base import BaseProcessor

logger = logging.getLogger(__name__)


class CSVProcessor(BaseProcessor):

    supported_extensions = [".csv"]
    ROWS_PER_GROUP = 10

    def extract(self, file_path: str) -> List[Dict[str, Any]]:
        """Extract data from CSV, grouping rows with column context."""
        self.validate_file(file_path)

        file_name = Path(file_path).name
        sections = []

        # Detect encoding and delimiter
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            sample = f.read(4096)
            dialect = csv.Sniffer().sniff(sample) if sample.strip() else None

        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            if dialect:
                reader = csv.reader(f, dialect)
            else:
                reader = csv.reader(f)

            rows = list(reader)

        if not rows:
            return []

        headers = [h.strip() if h.strip() else f"Column_{i}" for i, h in enumerate(rows[0])]
        data_rows = rows[1:]

        logger.info(f"Processing CSV: {file_name} ({len(data_rows)} rows)")

        for group_start in range(0, len(data_rows), self.ROWS_PER_GROUP):
            group_end = min(group_start + self.ROWS_PER_GROUP, len(data_rows))
            group = data_rows[group_start:group_end]

            row_texts = []
            for row in group:
                pairs = []
                for header, value in zip(headers, row):
                    if value and value.strip():
                        pairs.append(f"{header}: {value.strip()}")
                if pairs:
                    row_texts.append(". ".join(pairs))

            if not row_texts:
                continue

            sections.append({
                "content": "\n".join(row_texts),
                "metadata": {
                    "source": file_name,
                    "row_range": f"{group_start + 2}-{group_end + 1}",
                    "columns": headers,
                },
                "chunk_type": "row_group",
            })

        logger.info(f"Extracted {len(sections)} sections from {file_name}")
        return sections


class TextProcessor(BaseProcessor):

    supported_extensions = [".txt"]

    def extract(self, file_path: str) -> List[Dict[str, Any]]:
        """Extract text from plain text files."""
        self.validate_file(file_path)

        file_name = Path(file_path).name

        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()

        if not content.strip():
            return []

        # Split by double newlines (paragraph breaks) for natural sections
        paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]

        sections = []
        for i, para in enumerate(paragraphs):
            sections.append({
                "content": para,
                "metadata": {
                    "source": file_name,
                    "paragraph_index": i,
                },
                "chunk_type": "text",
            })

        logger.info(f"Extracted {len(sections)} sections from {file_name}")
        return sections
