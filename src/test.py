from src.indexer import CodeIndexer
from src.retriever import Retriever
from src.generator import Generator

if __name__ == '__main__':

    indexer = CodeIndexer()
    all_chunks = indexer.index_corpus(2000)

    retrieval = Retriever('data/processed/bm25_vectorizer.pkl', 'data/processed/chunks.json')
    query = "How to install vLLM"
    result = retrieval.retrieve(query)

    generation = Generator("Qwen/Qwen3-0.6B")
    answer = generation.generate(query, result)

    print(answer)
