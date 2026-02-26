"""
XLSX Document Processor
Extracts data from Excel files, converting rows into natural language chunks
with column headers as context. This makes the data much more searchable.
"""

import logging
from typing import List, Dict, Any
from pathlib import Path
from .base import BaseProcessor

logger = logging.getLogger(__name__)


class XLSXProcessor(BaseProcessor):

    supported_extensions = [".xlsx", ".xls"]

    # How many rows to group into one section
    ROWS_PER_GROUP = 10

    def extract(self, file_path: str) -> List[Dict[str, Any]]:
        """Extract data from Excel, one section per row group per sheet."""
        self.validate_file(file_path)
        import openpyxl

        file_name = Path(file_path).name
        wb = openpyxl.load_workbook(file_path, read_only=True, data_only=True)
        sections = []

        for sheet_name in wb.sheetnames:
            ws = wb[sheet_name]
            rows = list(ws.iter_rows(values_only=True))

            if not rows:
                continue

            # First row is headers
            headers = [str(h).strip() if h else f"Column_{i}" for i, h in enumerate(rows[0])]
            data_rows = rows[1:]

            if not data_rows:
                continue

            logger.info(f"Processing sheet: {sheet_name} ({len(data_rows)} rows, {len(headers)} columns)")

            # Group rows into batches for chunking
            for group_start in range(0, len(data_rows), self.ROWS_PER_GROUP):
                group_end = min(group_start + self.ROWS_PER_GROUP, len(data_rows))
                group = data_rows[group_start:group_end]

                # Convert each row to natural language with column context
                row_texts = []
                for row in group:
                    pairs = []
                    for header, value in zip(headers, row):
                        if value is not None and str(value).strip():
                            pairs.append(f"{header}: {value}")
                    if pairs:
                        row_texts.append(". ".join(pairs))

                if not row_texts:
                    continue

                content = "\n".join(row_texts)

                sections.append({
                    "content": content,
                    "metadata": {
                        "source": file_name,
                        "sheet": sheet_name,
                        "row_range": f"{group_start + 2}-{group_end + 1}",  # +2 for header + 0-index
                        "columns": headers,
                    },
                    "chunk_type": "row_group",
                })

        wb.close()
        logger.info(f"Extracted {len(sections)} sections from {file_name}")
        return sections
