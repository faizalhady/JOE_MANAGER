"""
Base Document Processor
All file-type processors (PDF, DOCX, XLSX, CSV) inherit from this.
Defines the contract every processor must follow.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any
from pathlib import Path


class BaseProcessor(ABC):
    """
    Abstract base class for document processors.
    
    Every processor must implement extract(), which takes a file path
    and returns a list of extracted text sections with metadata.
    """

    # Each subclass sets this
    supported_extensions: List[str] = []

    def can_process(self, file_path: str) -> bool:
        """Check if this processor can handle the given file."""
        ext = Path(file_path).suffix.lower()
        return ext in self.supported_extensions

    @abstractmethod
    def extract(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Extract text content from a file.
        
        Args:
            file_path: path to the source file
            
        Returns:
            List of dicts, each containing:
            {
                "content": str,        # the extracted text
                "metadata": dict,      # file-specific context (page, sheet, section, etc.)
                "chunk_type": str,     # "text", "table", "row_group", "header", etc.
            }
            
        Note: These are NOT final chunks yet. The chunker will further split
        large sections. But small sections (like individual Excel rows) 
        may already be chunk-sized.
        """
        pass

    def validate_file(self, file_path: str) -> bool:
        """Check if the file exists and is readable."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        if not path.is_file():
            raise ValueError(f"Not a file: {file_path}")
        if path.stat().st_size == 0:
            raise ValueError(f"File is empty: {file_path}")
        return True
