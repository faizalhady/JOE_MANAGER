"""
Document Processors Package
Provides a factory function to get the right processor for any file type.
"""

from pathlib import Path
from .base import BaseProcessor
from .pdf_processor import PDFProcessor
from .docx_processor import DOCXProcessor
from .xlsx_processor import XLSXProcessor
from .csv_processor import CSVProcessor, TextProcessor

# Registry of all available processors
_PROCESSORS = [
    PDFProcessor(),
    DOCXProcessor(),
    XLSXProcessor(),
    CSVProcessor(),
    TextProcessor(),
]


def get_processor(file_path: str) -> BaseProcessor:
    """
    Factory: returns the correct processor for a given file.
    
    Usage:
        processor = get_processor("report.pdf")
        sections = processor.extract("report.pdf")
    """
    ext = Path(file_path).suffix.lower()

    for proc in _PROCESSORS:
        if ext in proc.supported_extensions:
            return proc

    supported = [ext for p in _PROCESSORS for ext in p.supported_extensions]
    raise ValueError(
        f"Unsupported file type: {ext}. Supported: {', '.join(supported)}"
    )


__all__ = [
    "get_processor",
    "BaseProcessor",
    "PDFProcessor",
    "DOCXProcessor",
    "XLSXProcessor",
    "CSVProcessor",
    "TextProcessor",
]
