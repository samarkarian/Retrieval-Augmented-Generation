import fire
from .models import MinimalSearchResults, MinimalAnswer

class RAGPipeline():
    def index(self, max_chunk_size: int = 2000) -> None:
        print('todo')


    def search(self, query: str, k: int = 10) -> MinimalSearchResults:
        print('todo')


    def search_dataset(self, dataset_path: str, save_directory: str, k: int = 10) -> None:
        print('todo')


    def answer(self, query: str, k: int = 10) -> MinimalAnswer:
        print('todo')


    def answer_dataset(self, student_search_results_path: str,
    save_directory: str) -> None:
        print('todo')


    def evaluate(self, student_search_results_path: str, dataset_path: str) -> None:
        print('todo')


if __name__=="__main__":
    fire.Fire(RAGPipeline)