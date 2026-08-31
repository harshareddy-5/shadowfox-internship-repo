import uuid
from typing import List, Dict, Any
from app.core.config import settings
from app.schemas.document import DocumentChunkMetadata


class DocumentChunker:
    """
    Splits document pages into text chunks with configurable size and overlap.
    Preserves document metadata and page numbers per chunk.
    """

    def __init__(self, chunk_size: int = None, chunk_overlap: int = None):
        self.chunk_size = chunk_size or settings.DEFAULT_CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or settings.DEFAULT_CHUNK_OVERLAP

        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be strictly less than chunk_size")

    def chunk_document(
        self,
        document_id: str,
        filename: str,
        page_records: List[Dict[str, Any]]
    ) -> List[DocumentChunkMetadata]:
        """
        Takes page records [{"page": int, "text": str}] and generates DocumentChunkMetadata objects.
        """
        all_chunks: List[DocumentChunkMetadata] = []
        global_chunk_index = 0

        for record in page_records:
            page_num = record.get("page")
            text = record.get("text", "")

            if not text.strip():
                continue

            raw_chunks = self._split_text(text)

            for chunk_text in raw_chunks:
                if not chunk_text.strip():
                    continue

                chunk_id = f"{document_id}_c{global_chunk_index}"
                chunk_meta = DocumentChunkMetadata(
                    chunk_id=chunk_id,
                    document_id=document_id,
                    filename=filename,
                    page=page_num,
                    chunk_index=global_chunk_index,
                    text=chunk_text.strip()
                )
                all_chunks.append(chunk_meta)
                global_chunk_index += 1

        return all_chunks

    def _split_text(self, text: str) -> List[str]:
        """
        Recursive character/paragraph/sentence splitter enforcing chunk_size and chunk_overlap.
        Separators tried in order: double newline, single newline, sentence end, space, character.
        """
        if len(text) <= self.chunk_size:
            return [text]

        separators = ["\n\n", "\n", ". ", "? ", "! ", " ", ""]
        return self._recursive_split(text, separators)

    def _recursive_split(self, text: str, separators: List[str]) -> List[str]:
        """Recursively split text using the first effective separator."""
        if len(text) <= self.chunk_size:
            return [text]

        if not separators:
            # Hard character slice if no separators left
            return self._slice_hard(text)

        sep = separators[0]
        splits = text.split(sep) if sep else list(text)
        
        # If splitting by current separator doesn't refine the text, try next separator
        if len(splits) == 1:
            return self._recursive_split(text, separators[1:])

        chunks = []
        current_chunk = []
        current_length = 0

        for piece in splits:
            piece_len = len(piece) + (len(sep) if current_chunk else 0)

            if current_length + piece_len > self.chunk_size:
                if current_chunk:
                    joined = sep.join(current_chunk)
                    chunks.append(joined)

                    # Compute overlap from end of current chunk
                    overlap_size = 0
                    overlap_pieces = []
                    for p in reversed(current_chunk):
                        if overlap_size + len(p) + len(sep) <= self.chunk_overlap:
                            overlap_pieces.insert(0, p)
                            overlap_size += len(p) + len(sep)
                        else:
                            break

                    current_chunk = overlap_pieces
                    current_length = sum(len(p) for p in current_chunk) + len(sep) * max(0, len(current_chunk) - 1)

            current_chunk.append(piece)
            current_length += piece_len

        if current_chunk:
            joined = sep.join(current_chunk)
            chunks.append(joined)

        # Recursively reduce any oversized chunk if separator was too coarse
        final_chunks = []
        for c in chunks:
            if len(c) > self.chunk_size and len(separators) > 1:
                final_chunks.extend(self._recursive_split(c, separators[1:]))
            else:
                final_chunks.append(c)

        return final_chunks

    def _slice_hard(self, text: str) -> List[str]:
        """Fallback hard character slicing with overlap."""
        step = self.chunk_size - self.chunk_overlap
        chunks = []
        for i in range(0, len(text), step):
            chunks.append(text[i : i + self.chunk_size])
        return chunks
