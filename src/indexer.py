from typing import List
from pathlib import Path
import re
from rank_bm25 import BM25Okapi
import json
import pickle

class CodeIndexer:

    def chunk_python_file(self, content: str, max_chunk_size: int, path_file: str) -> List[dict]:
        chunks_py = []

        with open(content, 'r') as f:
            lines = f.readlines()

        current_chunk_lines = []
        char_position = 0

        for line in lines:
            if re.match('^(class |def )', line):
                if current_chunk_lines:
                    chunk_content = ''.join(current_chunk_lines)
                    self._process_chunk(chunk_content, max_chunk_size, path_file, char_position, chunks_py)
                    char_position += len(chunk_content)
                current_chunk_lines = [line]
            else:
                current_chunk_lines.append(line)

        if current_chunk_lines:
            chunk_content = ''.join(current_chunk_lines)
            self._process_chunk(chunk_content, max_chunk_size, path_file, char_position, chunks_py)

        return chunks_py

    def _process_chunk(self, chunk_content: str, max_chunk_size: int, path_file: str, start_pos: int, chunks_list: List[dict]) -> None:

        if len(chunk_content) <= max_chunk_size:
            chunk_dict = {
                "file_path": path_file,
                "first_character_index": start_pos,
                "last_character_index": start_pos + len(chunk_content),
                "content": chunk_content
            }
            chunks_list.append(chunk_dict)
        else:
            paragraphs = re.split(r'\n\n+', chunk_content)
            current_chunk = ""
            char_pos = start_pos

            for para in paragraphs:
                if len(current_chunk) + len(para) <= max_chunk_size:
                    current_chunk += para + "\n\n"
                else:
                    if current_chunk:
                        chunk_dict = {
                            "file_path": path_file,
                            "first_character_index": char_pos,
                            "last_character_index": char_pos + len(current_chunk),
                            "content": current_chunk
                        }
                        chunks_list.append(chunk_dict)
                        char_pos += len(current_chunk)
                    current_chunk = para + "\n\n"

            if current_chunk:
                chunk_dict = {
                    "file_path": path_file,
                    "first_character_index": char_pos,
                    "last_character_index": char_pos + len(current_chunk),
                    "content": current_chunk
                }
                chunks_list.append(chunk_dict)



    def chunk_markdown_file(self, content: str, max_chunk_size: int, path_file: str) -> List[dict]:
        """Chunk a markdown file by headings, then by paragraphs if needed."""
        chunks_md = []

        with open(content, 'r') as f:
            file_content = f.read()

        parts = re.split('^(#{1,6} .*)$', file_content, flags=re.MULTILINE)

        sections = []
        i = 1
        while i < len(parts):
            heading = parts[i]
            section_content = parts[i + 1] if i + 1 < len(parts) else ""
            sections.append(heading + section_content)
            i += 2

        char_position = 0
        for section in sections:
            if len(section) <= max_chunk_size:
                chunk_dict = {
                    "file_path": path_file,
                    "first_character_index": char_position,
                    "last_character_index": char_position + len(section),
                    "content": section
                }
                chunks_md.append(chunk_dict)
                char_position += len(section)
            else:
                paragraphs = re.split(r'\n\n+', section)
                current_chunk = ""
                for para in paragraphs:
                    if len(current_chunk) + len(para) <= max_chunk_size:
                        current_chunk += para + "\n\n"
                    else:
                        if current_chunk:
                            chunk_dict = {
                                "file_path": path_file,
                                "first_character_index": char_position,
                                "last_character_index": char_position + len(current_chunk),
                                "content": current_chunk
                            }
                            chunks_md.append(chunk_dict)
                            char_position += len(current_chunk)
                        current_chunk = para + "\n\n"

                if current_chunk:
                    chunk_dict = {
                        "file_path": path_file,
                        "first_character_index": char_position,
                        "last_character_index": char_position + len(current_chunk),
                        "content": current_chunk
                    }
                    chunks_md.append(chunk_dict)
                    char_position += len(current_chunk)

        return chunks_md

    def index_corpus(self, max_chunk_size: int) -> List[dict]:

        folder = Path("data/raw/vllm-0.10.1")

        all_chunks = []
        for file in folder.rglob("*.md"):
            chunks = self.chunk_markdown_file(str(file), max_chunk_size, str(file))
            all_chunks.extend(chunks)

        for file in folder.rglob('*.py'):
            chunks = self.chunk_python_file(str(file), max_chunk_size, str(file))
            all_chunks.extend(chunks)

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

        chunks_content: List[str] = []
        chunks_tokenizer: list[str] = []

        for c in content:
            chunks_content.append(c['content'])

        for words in chunks_content:
            tokens = words.lower().split()
            chunks_tokenizer.append(tokens)

        return chunks_tokenizer

