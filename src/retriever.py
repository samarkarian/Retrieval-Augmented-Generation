"""Lexical retrieval over the BM25 index built by the indexer."""

from typing import List
import pickle
import json
from .tokenizer import tokenize


class Retriever:
    """Ranks corpus chunks against a question using BM25."""

    def __init__(self, bm25_path: str, chunks_path: str) -> None:
        """Load the BM25 index and the chunk records from disk."""

        with open(bm25_path, 'rb') as f:
            self.bm25 = pickle.load(f)

        with open(chunks_path, 'r') as f:
            self.chunks = json.load(f)

    def retrieve(self, query: str, k: int = 5) -> List[dict]:
        """Retrieve the top-k chunk records for a query, ranked by score.

        Args:
            query: The question, in plain text.
            k: How many chunks to return.

        Returns:
            The k highest-scoring chunk records.
        """

        tokenize_query = tokenize(query)
        result: List[dict] = self.bm25.get_top_n(
            tokenize_query, self.chunks, k)

        return result
