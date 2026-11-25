"""
PDF Processing module for Reading Passage Evidence Locator.
Extracts text page by page from PDF files using PyMuPDF (fitz),
preserving original 1-based page numbers and metadata.
Gracefully handles scanned, corrupted, or empty PDFs.
"""

import os
import logging
from typing import List, Dict, Any, Union
from pathlib import Path

# Use pymupdf directly (fitz has deprecation notice in newer versions)
try:
    import pymupdf
except ImportError:
    import fitz as pymupdf

logger = logging.getLogger(__name__)


class PDFProcessor:
    """Extracts text and metadata from PDF files while preserving page numbers."""

    @staticmethod
    def extract_from_file(file_path: Union[str, Path]) -> List[Dict[str, Any]]:
        """
        Extracts text from a single PDF document page by page.

        Args:
            file_path: Path to the PDF file.

        Returns:
            List of page dictionaries:
            [
                {
                    "document_name": str,
                    "file_path": str,
                    "page_number": int,   # 1-indexed
                    "text": str,
                    "char_count": int,
                    "is_empty": bool
                },
                ...
            ]

        Raises:
            FileNotFoundError: If the specified file does not exist.
            ValueError: If the file is not a valid or readable PDF.
        """
        path_obj = Path(file_path)
        if not path_obj.exists():
            raise FileNotFoundError(f"PDF file not found: {file_path}")

        if not path_obj.is_file():
            raise ValueError(f"Path is not a regular file: {file_path}")

        doc_name = path_obj.name
        pages: List[Dict[str, Any]] = []

        try:
            doc = pymupdf.open(str(path_obj))
        except Exception as e:
            logger.error(f"Failed to open PDF '{doc_name}': {e}")
            raise ValueError(f"Could not open '{doc_name}' as a PDF: {str(e)}")

        try:
            if doc.is_encrypted:
                logger.warning(f"PDF '{doc_name}' is password-protected or encrypted.")
                raise ValueError(f"PDF '{doc_name}' is encrypted or password-protected.")

            total_pages = len(doc)
            if total_pages == 0:
                logger.warning(f"PDF '{doc_name}' has 0 pages.")
                return pages

            for page_index in range(total_pages):
                page = doc.load_page(page_index)
                page_num = page_index + 1  # 1-based page numbering

                # Extract plain text
                raw_text = page.get_text("text") or ""

                # Normalize text formatting
                cleaned_text = PDFProcessor._clean_page_text(raw_text)
                is_empty = len(cleaned_text.strip()) == 0

                if is_empty:
                    logger.debug(f"Page {page_num} in '{doc_name}' contains no readable text (may be blank or image-only).")

                pages.append({
                    "document_name": doc_name,
                    "file_path": str(path_obj.resolve()),
                    "page_number": page_num,
                    "text": cleaned_text,
                    "char_count": len(cleaned_text),
                    "is_empty": is_empty
                })

        finally:
            doc.close()

        # Check if entire document is empty or scanned without OCR text
        total_text_length = sum(p["char_count"] for p in pages)
        if total_text_length == 0:
            logger.warning(
                f"PDF '{doc_name}' yielded 0 characters of text across all {len(pages)} pages. "
                f"It may be a scanned image-only PDF without OCR layer."
            )

        return pages

    @staticmethod
    def extract_from_multiple(file_paths: List[Union[str, Path]]) -> Dict[str, Any]:
        """
        Extracts text from multiple PDF documents.

        Args:
            file_paths: List of file paths to process.

        Returns:
            Dictionary with aggregated pages and processing statistics:
            {
                "pages": List[Dict[str, Any]],
                "processed_files": List[str],
                "failed_files": Dict[str, str],
                "total_pages": int,
                "empty_pages_count": int
            }
        """
        all_pages: List[Dict[str, Any]] = []
        processed_files: List[str] = []
        failed_files: Dict[str, str] = {}
        empty_pages_count = 0

        for fp in file_paths:
            try:
                pages = PDFProcessor.extract_from_file(fp)
                all_pages.extend(pages)
                processed_files.append(Path(fp).name)
                empty_pages_count += sum(1 for p in pages if p["is_empty"])
            except Exception as e:
                failed_files[str(fp)] = str(e)
                logger.error(f"Error extracting '{fp}': {e}")

        return {
            "pages": all_pages,
            "processed_files": processed_files,
            "failed_files": failed_files,
            "total_pages": len(all_pages),
            "empty_pages_count": empty_pages_count
        }

    @staticmethod
    def _clean_page_text(text: str) -> str:
        """Cleans carriage returns and excessive whitespace while preserving structure."""
        if not text:
            return ""
        # Normalize CRLF / CR to LF
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        # Strip null bytes
        text = text.replace("\x00", "")
        # Remove consecutive blank lines
        lines = [line.strip() for line in text.split("\n")]
        # Recombine: preserve non-empty lines with single newline separation
        non_empty_lines = [l for l in lines if l]
        return " ".join(non_empty_lines)
