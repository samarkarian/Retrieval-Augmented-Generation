from typing import List
import re


def tokenize(text: str) -> List[str]:
    """Split text into lowercase tokens for BM25.

    Punctuation is dropped, and a compound identifier is kept whole as well
    as split into its parts.

    Example:
        "fused_batched_moe" -> ['fused_batched_moe', 'fused', 'batched', 'moe']
        "getNumTokens"      -> ['getnumtokens', 'get', 'num', 'tokens']
        "serve()"           -> ['serve']

    Args:
        text: Raw text, from a chunk or from a query.

    Returns:
        The list of tokens.
    """
    tokens: List[str] = []

    for word in re.findall(r'[A-Za-z0-9_]+', text):

        spaced = re.sub(r'([a-z0-9])([A-Z])', r'\1 \2', word).lower()

        parts = spaced.replace('_', ' ').split()

        if len(parts) > 1:
            tokens.append(word.lower())

        tokens.extend(parts)

    return tokens
