"""Command line entry point, exposed with Python Fire.

Run as `uv run python -m src <command>`.
"""

import fire
from .models import (
        MinimalSearchResults, MinimalAnswer, MinimalSource,
        RagDataset, StudentSearchResults, StudentSearchResultsAndAnswer
    )
from .indexer import CodeIndexer
from .retriever import Retriever
from .generator import Generator
from pathlib import Path
from tqdm import tqdm
import uuid
import json


class RAGPipeline():
    """The commands exposed by the CLI, one per method.

    Components are built inside the methods that need them, so that indexing
    never pays for loading the language model.
    """

    def index(self, max_chunk_size: int = 2000) -> None:
        """Ingest data/raw/ and build the index under data/processed/.

        Args:
            max_chunk_size: Upper bound on chunk length, in characters.
                Raising it above 2000 invalidates the output.
        """

        indexer = CodeIndexer()
        indexer.index_corpus(max_chunk_size)
        print('Ingestion complete! Indexes are saved under data/processed/')

    def search(self, query: str, k: int = 10) -> MinimalSearchResults:
        """Return the top-k source locations for a single query.

        The question id is generated: a query typed on the command line
        belongs to no dataset.

        Args:
            query: The question, in plain text.
            k: How many sources to return.

        Returns:
            The question, a generated id, and the retrieved locations.
        """

        retrieval = Retriever(
            'data/processed/bm25_vectorizer.pkl',
            'data/processed/chunks.json'
        )
        result = retrieval.retrieve(query, k)
        minimal_src_list = []
        for res in result:
            path = res['file_path']
            first = res['first_character_index']
            last = res['last_character_index']
            minimal_src_list.append(
                MinimalSource(
                    file_path=path,
                    first_character_index=first,
                    last_character_index=last
                )
            )

        return MinimalSearchResults(
            question_id=str(uuid.uuid4()),
            question=query,
            retrieved_sources=minimal_src_list
        )

    def search_dataset(
            self, dataset_path: str,
            save_directory: str, k: int = 10) -> None:
        """Run retrieval over a whole dataset and write the results.

        Each question keeps the id it carries in the dataset: that is how the
        grader matches results against ground truth. The output file reuses
        the input file name, so scope save_directory per dataset.

        Args:
            dataset_path: Path to a RagDataset JSON file.
            save_directory: Where to write; created if missing.
            k: How many sources to retrieve per question.
        """

        with open(dataset_path, 'r') as f:
            questions = json.load(f)

        dataset = RagDataset.model_validate(questions)

        retrieval = Retriever('data/processed/bm25_vectorizer.pkl',
                              'data/processed/chunks.json')
        search_results = []

        for q in tqdm(dataset.rag_questions, desc='Searching'):
            result = retrieval.retrieve(q.question, k)
            minimal_src_list = []
            for res in result:
                minimal_src_list.append(MinimalSource(
                    file_path=res['file_path'],
                    first_character_index=res['first_character_index'],
                    last_character_index=res['last_character_index'],
                ))
            search_results.append(
                MinimalSearchResults(
                    question_id=q.question_id,
                    question=q.question,
                    retrieved_sources=minimal_src_list,
                )
            )

        student_results = StudentSearchResults(
            search_results=search_results,
            k=k
        )

        directory = Path(save_directory)
        directory.mkdir(parents=True, exist_ok=True)
        output_path = directory / Path(dataset_path).name

        with output_path.open('w') as f:
            f.write(student_results.model_dump_json(indent=2))

        print(f'Saved student_search_results to {output_path}')

    def answer(self, query: str, k: int = 10) -> MinimalAnswer:
        """Answer a single query from the retrieved context.

        Works on the raw chunks, which carry the text search() drops. Only
        the first five reach the model, however large k is.

        Args:
            query: The question, in plain text.
            k: How many sources to retrieve.

        Returns:
            The question, the retrieved locations, and the answer.
        """

        retrieval = Retriever(
            'data/processed/bm25_vectorizer.pkl',
            'data/processed/chunks.json'
        )

        top_k = retrieval.retrieve(query, k)
        gen = Generator("Qwen/Qwen3-0.6B")
        answer = gen.generate(query, top_k, 5)

        minimal_src_list = []
        for src in top_k:
            minimal_src_list.append(
                MinimalSource(
                    file_path=src['file_path'],
                    first_character_index=src['first_character_index'],
                    last_character_index=src['last_character_index']
                )
            )

        return MinimalAnswer(
            question_id=str(uuid.uuid4()),
            question=query,
            retrieved_sources=minimal_src_list,
            answer=answer)

    def answer_dataset(
            self, student_search_results_path: str,
            save_directory: str) -> None:
        """Generate an answer for every question of a search results file.

        Chunk text is recovered by re-reading each source file at the
        recorded range, not by searching again: the answers must stand on the
        sources the file declares. Files are read once and cached.

        Args:
            student_search_results_path: Path to a StudentSearchResults file.
            save_directory: Where to write; created if missing.
        """

        with open(student_search_results_path, 'r') as f:
            student_res = json.load(f)
        student = StudentSearchResults.model_validate(student_res)

        gen = Generator("Qwen/Qwen3-0.6B")
        file_cache = {}
        answers = []

        for res in tqdm(student.search_results, desc='Answering'):
            chunks = []
            for src in res.retrieved_sources:
                if src.file_path not in file_cache:
                    try:
                        with open(src.file_path, 'r') as f:
                            file_cache[src.file_path] = f.read()
                    except (OSError, UnicodeDecodeError):
                        file_cache[src.file_path] = ''
                txt = file_cache[src.file_path]
                chunks.append({
                    'file_path': src.file_path,
                    'content': txt[src.first_character_index:
                                   src.last_character_index]
                })

            answer = gen.generate(res.question, chunks, 5)
            answers.append(MinimalAnswer(
                question_id=res.question_id,
                question=res.question,
                retrieved_sources=res.retrieved_sources,
                answer=answer
            ))

        out = StudentSearchResultsAndAnswer(
            search_results=answers, k=student.k
        )

        directory = Path(save_directory)
        directory.mkdir(parents=True, exist_ok=True)
        output_path = directory / Path(student_search_results_path).name

        with output_path.open('w') as f:
            f.write(out.model_dump_json(indent=2))

        print(
            f'Saved student_search_results_and_answer to {output_path}'
        )

    def evaluate(
            self, student_search_results_path: str,
            dataset_path: str) -> None:
        """Report recall@k against a ground-truth dataset.

        A reference source counts as found when one of the top-k results is
        in the same file and overlaps its range by at least 0.05 IoU. For
        local iteration only; the official score comes from the grading tool.

        Args:
            student_search_results_path: Path to a StudentSearchResults file.
            dataset_path: Path to the matching AnsweredQuestions dataset.
        """
        pass


if __name__ == "__main__":

    try:
        fire.Fire(RAGPipeline)
    except KeyboardInterrupt:
        print('\nInterrupted by user.')
    except Exception as e:
        print(f'Error: {e}')
