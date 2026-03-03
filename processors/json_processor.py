"""
JSON Document Processor
Extracts searchable text from JSON files by flattening nested structures
into key-value pairs. Handles arrays, nested objects, and mixed types.
"""

import json
import logging
from typing import List, Dict, Any
from pathlib import Path
from .base import BaseProcessor

logger = logging.getLogger(__name__)


class JSONProcessor(BaseProcessor):

    supported_extensions = [".json"]

    # How many records to group per section (for JSON arrays)
    RECORDS_PER_GROUP = 5

    def extract(self, file_path: str) -> List[Dict[str, Any]]:
        """Extract searchable text from JSON files."""
        self.validate_file(file_path)

        file_name = Path(file_path).name

        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON in {file_name}: {e}")
                return []

        sections = []

        if isinstance(data, list):
            # JSON array — group records into sections
            sections = self._process_array(data, file_name)
        elif isinstance(data, dict):
            # JSON object — flatten into readable text
            sections = self._process_object(data, file_name)
        else:
            # Scalar value
            sections = [{
                "content": str(data),
                "metadata": {"source": file_name},
                "chunk_type": "text",
            }]

        logger.info(f"Extracted {len(sections)} sections from {file_name}")
        return sections

    def _process_array(self, data: list, file_name: str) -> List[Dict[str, Any]]:
        """Process a JSON array, grouping records into sections."""
        sections = []

        for group_start in range(0, len(data), self.RECORDS_PER_GROUP):
            group_end = min(group_start + self.RECORDS_PER_GROUP, len(data))
            group = data[group_start:group_end]

            record_texts = []
            for item in group:
                if isinstance(item, dict):
                    flat = self._flatten_dict(item)
                    pairs = [f"{k}: {v}" for k, v in flat.items() if v]
                    if pairs:
                        record_texts.append(". ".join(pairs))
                elif isinstance(item, (str, int, float, bool)):
                    record_texts.append(str(item))

            if not record_texts:
                continue

            sections.append({
                "content": "\n".join(record_texts),
                "metadata": {
                    "source": file_name,
                    "record_range": f"{group_start + 1}-{group_end}",
                    "total_records": len(data),
                },
                "chunk_type": "row_group",
            })

        return sections

    def _process_object(self, data: dict, file_name: str) -> List[Dict[str, Any]]:
        """Process a JSON object by flattening it."""
        sections = []

        # Check if it's a dict of arrays (common API format like {"results": [...]})
        for key, value in data.items():
            if isinstance(value, list) and len(value) > 0 and isinstance(value[0], dict):
                # This key contains an array of records — process it separately
                sub_sections = self._process_array(value, file_name)
                for s in sub_sections:
                    s["metadata"]["json_key"] = key
                sections.extend(sub_sections)
            elif isinstance(value, dict):
                # Nested object — flatten it
                flat = self._flatten_dict(value)
                pairs = [f"{k}: {v}" for k, v in flat.items() if v]
                if pairs:
                    sections.append({
                        "content": f"{key}\n" + ". ".join(pairs),
                        "metadata": {
                            "source": file_name,
                            "json_key": key,
                        },
                        "chunk_type": "text",
                    })
            else:
                # Simple key-value — accumulate
                pass

        # If no nested structures were found, flatten the whole thing
        if not sections:
            flat = self._flatten_dict(data)
            pairs = [f"{k}: {v}" for k, v in flat.items() if v]
            if pairs:
                sections.append({
                    "content": ". ".join(pairs),
                    "metadata": {"source": file_name},
                    "chunk_type": "text",
                })

        return sections

    def _flatten_dict(self, d: dict, prefix: str = "") -> dict:
        """
        Flatten a nested dict into dot-notation keys.
        {"a": {"b": 1}} → {"a.b": "1"}
        """
        flat = {}
        for key, value in d.items():
            full_key = f"{prefix}.{key}" if prefix else key

            if isinstance(value, dict):
                flat.update(self._flatten_dict(value, full_key))
            elif isinstance(value, list):
                # Convert short lists to string, skip very long ones
                if len(value) <= 10:
                    str_items = [str(v) for v in value if not isinstance(v, (dict, list))]
                    if str_items:
                        flat[full_key] = ", ".join(str_items)
                else:
                    flat[full_key] = f"[{len(value)} items]"
            elif value is not None:
                str_val = str(value).strip()
                if str_val and str_val.lower() != "none":
                    flat[full_key] = str_val

        return flat
