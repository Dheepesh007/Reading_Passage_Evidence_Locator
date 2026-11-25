"""
Text chunking module for Reading Passage Evidence Locator.
Splits page-level text into meaningful, overlapping chunks while
strictly preserving document names and original page numbers.
Preserves sentence boundaries wherever possible to maintain semantic context.
"""

import re
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


class TextChunker:
    """Segments page-level text into context-preserving, metadata-tagged chunks."""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100):
        """
        Args:
            chunk_size: Target maximum character length for each chunk.
            chunk_overlap: Overlap in characters between consecutive chunks.
        """
        if chunk_size <= 50:
            raise ValueError("chunk_size must be greater than 50 characters.")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be strictly less than chunk_size.")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_pages(self, pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Chunks a list of page objects extracted by PDFProcessor.

        Args:
            pages: List of page dicts with keys 'document_name', 'page_number', 'text'.

        Returns:
            List of chunk dicts:
            [
                {
                    "chunk_id": int,
                    "document_name": str,
                    "page_number": int,
                    "text": str,
                    "char_count": int,
                    "chunk_index": int
                },
                ...
            ]
        """
        all_chunks: List[Dict[str, Any]] = []
        global_chunk_id = 0

        for page in pages:
            text = page.get("text", "").strip()
            if not text:
                continue

            doc_name = page.get("document_name", "unknown.pdf")
            page_num = page.get("page_number", 1)

            page_chunks = self._chunk_single_page(text, doc_name, page_num, start_id=global_chunk_id)
            all_chunks.extend(page_chunks)
            global_chunk_id += len(page_chunks)

        logger.info(f"Chunked {len(pages)} pages into {len(all_chunks)} chunks.")
        return all_chunks

    def _chunk_single_page(
        self,
        text: str,
        document_name: str,
        page_number: int,
        start_id: int = 0
    ) -> List[Dict[str, Any]]:
        """Splits single-page text into chunks respecting sentence boundaries."""
        sentences = self._split_into_sentences(text)
        if not sentences:
            return []

        chunks: List[Dict[str, Any]] = []
        current_chunk_sentences: List[str] = []
        current_length = 0
        chunk_index = 0

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            sentence_len = len(sentence)

            # If a single sentence is longer than chunk_size, split by words
            if sentence_len > self.chunk_size:
                # First flush accumulated sentences
                if current_chunk_sentences:
                    chunk_text = " ".join(current_chunk_sentences)
                    chunks.append(self._create_chunk_dict(
                        start_id + len(chunks), document_name, page_number, chunk_text, chunk_index
                    ))
                    chunk_index += 1
                    current_chunk_sentences = []
                    current_length = 0

                # Split long sentence by word boundary
                word_chunks = self._split_long_sentence(sentence)
                for w_text in word_chunks:
                    chunks.append(self._create_chunk_dict(
                        start_id + len(chunks), document_name, page_number, w_text, chunk_index
                    ))
                    chunk_index += 1
                continue

            # If adding this sentence exceeds chunk_size, finalize current chunk
            if current_length + sentence_len + 1 > self.chunk_size and current_chunk_sentences:
                chunk_text = " ".join(current_chunk_sentences)
                chunks.append(self._create_chunk_dict(
                    start_id + len(chunks), document_name, page_number, chunk_text, chunk_index
                ))
                chunk_index += 1

                # Carry over trailing sentences for overlap
                overlap_sentences = []
                overlap_len = 0
                for s in reversed(current_chunk_sentences):
                    if overlap_len + len(s) + 1 <= self.chunk_overlap:
                        overlap_sentences.insert(0, s)
                        overlap_len += len(s) + 1
                    else:
                        break

                current_chunk_sentences = overlap_sentences
                current_length = sum(len(s) + 1 for s in current_chunk_sentences)

            current_chunk_sentences.append(sentence)
            current_length += sentence_len + 1

        # Flush remaining sentences
        if current_chunk_sentences:
            chunk_text = " ".join(current_chunk_sentences)
            chunks.append(self._create_chunk_dict(
                start_id + len(chunks), document_name, page_number, chunk_text, chunk_index
            ))

        return chunks

    @staticmethod
    def _split_into_sentences(text: str) -> List[str]:
        """Splits text on punctuation boundaries while preserving sentence content."""
        # Match punctuation (. ! ?) followed by whitespace, avoiding common abbreviations
        sentence_endings = re.compile(r'(?<=[.!?])\s+(?=[A-Z0-9"\'])')
        sentences = sentence_endings.split(text)
        if len(sentences) == 1 and "\n" in text:
            # Fallback to newline separation if no punctuation endings matched
            sentences = [s.strip() for s in text.split("\n") if s.strip()]
        return sentences

    def _split_long_sentence(self, sentence: str) -> List[str]:
        """Splits an oversized sentence at word boundaries."""
        words = sentence.split()
        sub_chunks = []
        cur_words: List[str] = []
        cur_len = 0

        for w in words:
            if cur_len + len(w) + 1 > self.chunk_size and cur_words:
                sub_chunks.append(" ".join(cur_words))
                cur_words = []
                cur_len = 0
            cur_words.append(w)
            cur_len += len(w) + 1

        if cur_words:
            sub_chunks.append(" ".join(cur_words))
        return sub_chunks

    @staticmethod
    def _create_chunk_dict(
        chunk_id: int,
        document_name: str,
        page_number: int,
        text: str,
        chunk_index: int
    ) -> Dict[str, Any]:
        return {
            "chunk_id": chunk_id,
            "document_name": document_name,
            "page_number": page_number,
            "text": text.strip(),
            "char_count": len(text.strip()),
            "chunk_index": chunk_index
        }
