"""
PDF Document Processor
Extracts text from PDFs page by page, preserving page numbers as metadata.
"""

import re
import logging
from typing import List, Dict, Any
from pathlib import Path
from .base import BaseProcessor

logger = logging.getLogger(__name__)


class PDFProcessor(BaseProcessor):
    
    supported_extensions = [".pdf"]

    def extract(self, file_path: str) -> List[Dict[str, Any]]:
        """Extract text from PDF, one section per page."""
        self.validate_file(file_path)
        
        # Try pdfplumber first (better for tables), fall back to PyPDF2
        try:
            return self._extract_with_pdfplumber(file_path)
        except ImportError:
            logger.info("pdfplumber not available, using PyPDF2")
            return self._extract_with_pypdf2(file_path)

    def _extract_with_pdfplumber(self, file_path: str) -> List[Dict[str, Any]]:
        """Extract using pdfplumber (better table extraction)."""
        import pdfplumber

        sections = []
        file_name = Path(file_path).name

        with pdfplumber.open(file_path) as pdf:
            total_pages = len(pdf.pages)
            logger.info(f"Processing PDF: {file_name} ({total_pages} pages)")

            # Extract Jabil header metadata once from the first page
            doc_meta = self._extract_jabil_header_metadata(pdf)
            logger.info(
                f"Doc metadata → doc_name={doc_meta['doc_name']}, "
                f"plant_code={doc_meta['plant_code']}, "
                f"revision={doc_meta['revision']}, "
                f"released_date={doc_meta['released_date']}, "
                f"author={doc_meta['author']}"
            )

            for i, page in enumerate(pdf.pages):
                text = page.extract_text()
                if not text or not text.strip():
                    continue

                sections.append({
                    "content": text.strip(),
                    "metadata": {
                        "page": i + 1,
                        "total_pages": total_pages,
                        "source": file_name,
                        "doc_name": doc_meta["doc_name"],
                        "plant_code": doc_meta["plant_code"],
                        "revision": doc_meta["revision"],
                        "released_date": doc_meta["released_date"],
                        "author": doc_meta["author"],
                    },
                    "chunk_type": "text",
                })

                # Also extract tables separately if present
                tables = page.extract_tables()
                for t_idx, table in enumerate(tables):
                    if not table:
                        continue
                    # Convert table to readable text
                    table_text = self._table_to_text(table)
                    if table_text.strip():
                        sections.append({
                            "content": table_text,
                            "metadata": {
                                "page": i + 1,
                                "total_pages": total_pages,
                                "source": file_name,
                                "table_index": t_idx,
                                "doc_name": doc_meta["doc_name"],
                                "plant_code": doc_meta["plant_code"],
                                "revision": doc_meta["revision"],
                                "released_date": doc_meta["released_date"],
                                "author": doc_meta["author"],
                            },
                            "chunk_type": "table",
                        })

        logger.info(f"Extracted {len(sections)} sections from {file_name}")
        return sections

    def _extract_jabil_header_metadata(self, pdf) -> Dict[str, str]:
        """
        Extract doc_name, plant_code, revision, released_date, and author
        from Jabil WI PDF documents.

        Header table structure (page 1, top of every Jabil doc):
            Row 0: [JABIL logo cell,  "07-TE30-GEN-153"         ]  ← doc_name
            Row 1: [JABIL logo cell,  "WI,DRACULA USER GUIDELINES"]
            Row 2: [JABIL logo cell,  "Revision", "G", "Released Date", "MAR/09/2022"]

        - doc_name   : the cell matching pattern DD-XXXX-XXX-XXX (e.g. "07-TE30-GEN-153")
        - plant_code : first segment of doc_name before "-"        (e.g. "07")
        - revision   : cell after "Revision" label                 (e.g. "G")
        - released_date : cell after "Released Date" label         (e.g. "MAR/09/2022")
        - author     : from revision history table on page 1,
                       row where Rev column == revision letter      (e.g. "KC Kuan")
        """
        revision = "unknown"
        released_date = "unknown"
        author = "unknown"
        doc_name = "unknown"
        plant_code = "unknown"

        # Regex: Jabil doc IDs look like "07-TE30-GEN-153" or "00-AT80-00004"
        DOC_ID_PATTERN = re.compile(r'^\d{2}-[A-Z0-9]+-[A-Z0-9]+-[A-Z0-9]+$')

        try:
            first_page = pdf.pages[0]
            tables = first_page.extract_tables()

            # ── Pass 1: doc_name, Revision, Released Date from the header table ──
            for table in tables:
                for row in table:
                    clean = [str(c).strip() if c else "" for c in row]

                    # doc_name — find the cell that matches the Jabil doc ID pattern
                    if doc_name == "unknown":
                        for cell in clean:
                            if DOC_ID_PATTERN.match(cell):
                                doc_name = cell
                                plant_code = cell.split("-")[0]  # e.g. "07"
                                break

                    # Revision + Released Date — the row that contains both labels
                    if "Revision" in clean and "Released Date" in clean:
                        rev_idx = clean.index("Revision")
                        rel_idx = clean.index("Released Date")
                        if rev_idx + 1 < len(clean) and clean[rev_idx + 1]:
                            revision = clean[rev_idx + 1]
                        if rel_idx + 1 < len(clean) and clean[rel_idx + 1]:
                            released_date = clean[rel_idx + 1]
                        break  # header table done

                # Stop scanning tables once we have revision
                if revision != "unknown":
                    break

            # ── Pass 2: Author from the revision history table ──
            # Columns: Rev | Date Released | Author(S) | Change Details
            # Match the row where Rev == current revision letter
            if revision != "unknown":
                for table in tables:
                    header_row = None
                    for row in table:
                        clean_lower = [str(c).strip().lower() if c else "" for c in row]
                        if "rev" in clean_lower and any("author" in h for h in clean_lower):
                            header_row = clean_lower
                            break

                    if header_row is None:
                        continue

                    rev_col = next((i for i, h in enumerate(header_row) if h == "rev"), None)
                    author_col = next((i for i, h in enumerate(header_row) if "author" in h), None)

                    if rev_col is None or author_col is None:
                        continue

                    for row in table:
                        clean = [str(c).strip() if c else "" for c in row]
                        if len(clean) > max(rev_col, author_col):
                            if clean[rev_col].upper() == revision.upper():
                                author = clean[author_col] or "unknown"
                                break

                    if author != "unknown":
                        break

        except Exception as e:
            logger.warning(f"Could not extract Jabil header metadata: {e}")

        return {
            "doc_name": doc_name,
            "plant_code": plant_code,
            "revision": revision,
            "released_date": released_date,
            "author": author,
        }

    def _extract_with_pypdf2(self, file_path: str) -> List[Dict[str, Any]]:
        """Fallback extraction using PyPDF2."""
        from PyPDF2 import PdfReader

        sections = []
        file_name = Path(file_path).name
        reader = PdfReader(file_path)
        total_pages = len(reader.pages)

        logger.info(f"Processing PDF: {file_name} ({total_pages} pages)")

        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if not text or not text.strip():
                continue

            sections.append({
                "content": text.strip(),
                "metadata": {
                    "page": i + 1,
                    "total_pages": total_pages,
                    "source": file_name,
                },
                "chunk_type": "text",
            })

        logger.info(f"Extracted {len(sections)} sections from {file_name}")
        return sections

    def _table_to_text(self, table: list) -> str:
        """Convert a pdfplumber table (list of lists) to readable text."""
        if not table or not table[0]:
            return ""

        headers = table[0]
        rows = table[1:]
        lines = []

        for row in rows:
            pairs = []
            for header, value in zip(headers, row):
                if header and value:
                    pairs.append(f"{header}: {value}")
            if pairs:
                lines.append(". ".join(pairs))

        return "\n".join(lines)
