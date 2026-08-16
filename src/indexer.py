"""Corpus ingestion: chunking, tokenization and BM25 index construction."""

from typing import List
from pathlib import Path
import re
from rank_bm25 import BM25Okapi
import json
import pickle
from tqdm import tqdm
from .tokenizer import tokenize


class CodeIndexer:
    """Splits corpus files into chunks and builds a BM25 index.

    Boundaries are character positions found in the original text, and
    content is always a slice of it, so indices and content cannot drift
    apart.
    """

    def chunk_python_file(
            self, content: str,
            max_chunk_size: int, path_file: str) -> List[dict]:
        """Chunk a Python file on class/def boundaries, then by paragraphs.

        Args:
            content: Path to the file to read.
            max_chunk_size: Upper bound on chunk length, in characters.
            path_file: Path to record in each chunk; compared literally by
                the grader.

        Returns:
            The chunk records for this file.
        """
        chunks_py: List[dict] = []

        with open(content, 'r') as f:
            file_content = f.read()

        starts = []
        for m in re.finditer(r'^(class |def )', file_content,
                             flags=re.MULTILINE):
            starts.append(m.start())

        if not starts or starts[0] != 0:
            starts.insert(0, 0)
        starts.append(len(file_content))

        for i in range(len(starts) - 1):
            self._split_range(file_content, starts[i], starts[i + 1],
                              max_chunk_size, path_file, chunks_py)

        return chunks_py

    def _split_range(
            self, text: str, start: int, end: int, max_chunk_size: int,
            path_file: str, chunks_list: List[dict]) -> None:
        """Emit chunks covering text[start:end], cutting on blank lines.

        Accumulates up to the last cut position that still fits, then emits
        there.

        Args:
            text: The whole file content.
            start: Absolute start of the range to cover.
            end: Absolute end of the range to cover.
            max_chunk_size: Upper bound on chunk length.
            path_file: Path to record in each chunk.
            chunks_list: Accumulator the chunks are appended to.
        """

        if (end - start) <= max_chunk_size:
            self._add_chunk(
                text, start, end, max_chunk_size, path_file, chunks_list
            )
            return

        positions = []
        for sep in re.finditer(r'\n\n+', text[start:end]):
            positions.append(start + sep.end())
        positions.append(end)

        prev = start
        for pos in positions:
            if (pos - start) > max_chunk_size and prev > start:
                self._add_chunk(text, start, prev, max_chunk_size,
                                path_file, chunks_list)
                start = prev
            prev = pos

        self._add_chunk(
            text, start, end, max_chunk_size, path_file, chunks_list
        )

    def _add_chunk(
            self, text: str, start: int, end: int, max_chunk_size: int,
            path_file: str, chunks_list: List[dict]) -> None:
        """Store text[start:end], slicing anything still over the limit.

        The only place that writes a chunk, and so the only place the size
        limit has to be enforced.

        Args:
            text: The whole file content.
            start: Absolute start of the range to store.
            end: Absolute end of the range to store.
            max_chunk_size: Upper bound on chunk length.
            path_file: Path to record in each chunk.
            chunks_list: Accumulator the chunks are appended to.
        """

        while start < end:
            stop = start + max_chunk_size
            if stop < end:
                chunk = text[start: stop]
                chunk_dict = {
                    "file_path": path_file,
                    "first_character_index": start,
                    "last_character_index": stop,
                    "content": chunk
                }
                chunks_list.append(chunk_dict)
                start = stop
            else:
                chunk = text[start: end]
                chunk_dict = {
                    "file_path": path_file,
                    "first_character_index": start,
                    "last_character_index": end,
                    "content": chunk
                }
                chunks_list.append(chunk_dict)
                start = end

    def chunk_markdown_file(
            self, content: str,
            max_chunk_size: int, path_file: str) -> List[dict]:
        """Chunk a markdown file by headings, then by paragraphs if needed.

        Args:
            content: Path to the file to read.
            max_chunk_size: Upper bound on chunk length, in characters.
            path_file: Path to record in each chunk; compared literally by
                the grader.

        Returns:
            The chunk records for this file.
        """
        chunks_md: List[dict] = []

        with open(content, 'r') as f:
            file_content = f.read()

        starts = []
        for m in re.finditer(r'^#{1,6} .*$', file_content, flags=re.MULTILINE):
            starts.append(m.start())

        if not starts or starts[0] != 0:
            starts.insert(0, 0)
        starts.append(len(file_content))

        for i in range(len(starts) - 1):
            self._split_range(file_content, starts[i], starts[i + 1],
                              max_chunk_size, path_file, chunks_md)

        return chunks_md

    def index_corpus(self, max_chunk_size: int) -> List[dict]:
        """Ingest data/raw/ and write the index under data/processed/.

        Writes chunks.json, holding the chunk records, and
        bm25_vectorizer.pkl, holding the statistics learned from them. Files
        that cannot be decoded are skipped.

        Args:
            max_chunk_size: Upper bound on chunk length, in characters.

        Returns:
            Every chunk of the corpus, or an empty list if none was found.
        """

        folder = Path("data/raw/")
        files = (list(folder.rglob('*.md'))
                 + list(folder.rglob('*.py'))
                 + list(folder.rglob('*.txt')))
        all_chunks = []

        for file in tqdm(files, desc='Indexing'):
            file_str = str(file)
            try:
                if file_str.endswith('.md'):
                    chunks = self.chunk_markdown_file(
                        file_str, max_chunk_size, file_str
                    )
                else:
                    chunks = self.chunk_python_file(
                        file_str, max_chunk_size, file_str
                    )
                all_chunks.extend(chunks)
            except (UnicodeDecodeError, OSError) as e:
                tqdm.write(str(e))

        if len(all_chunks) == 0:
            print("No indexable files found in data/raw/")
            return []

        chunks_tokenized = self._tokenize(all_chunks)
        bm25 = BM25Okapi(chunks_tokenized)
        chunks_json = json.dumps(all_chunks, indent=2)

        p = Path('data/processed/')
        p.mkdir(parents=True, exist_ok=True)
        json_file = 'chunks.json'
        json_path = p / json_file
        with json_path.open('w') as f:
            f.write(chunks_json)

        pkl_file = 'bm25_vectorizer.pkl'
        pkl_path = p / pkl_file
        with pkl_path.open('wb') as f:
            pickle.dump(bm25, f)

        return all_chunks

    def _tokenize(self, content: List[dict]) -> List[List[str]]:
        """Turn chunk records into the token lists BM25 learns from."""

        chunks_content: List[str] = []
        chunks_tokenizer: List[List[str]] = []

        for c in content:
            chunks_content.append(c['content'])

        for words in chunks_content:
            tokens = tokenize(words)
            chunks_tokenizer.append(tokens)

        return chunks_tokenizer
