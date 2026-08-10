from typing import List
import pickle
import json
from rank_bm25 import BM25Okapi


class Retriever:

    def __init__(self, index_path: str, chunks_path: str) -> None:
        """Load BM25 index and chunks from disk."""

        with open(index_path, 'rb') as f:
            self.bm25 = pickle.load(f)

        with open(chunks_path, 'r') as f:
            self.chunks = json.load(f)


    def retrieve(self, query: str, k: int = 10) -> List[dict]:
        """Retrieve top-k chunks for a query."""

        tokenize_query = query.lower().split()
        result = self.bm25.get_top_n(tokenize_query, self.chunks, k)

        print(result)

        return result

if __name__ == '__main__':

    retrieval = Retriever('data/processed/bm25_vectorizer.pkl', 'data/processed/chunks.json')

    retrieval.retrieve("How to install vLLM")
