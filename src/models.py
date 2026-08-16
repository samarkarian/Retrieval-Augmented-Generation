"""Pydantic models exchanged between the stages of the pipeline."""

from pydantic import BaseModel, Field
from typing import List
import uuid


class MinimalSource(BaseModel):
    """A location in the corpus: a file path and a character range."""

    file_path: str
    first_character_index: int
    last_character_index: int


class UnansweredQuestion(BaseModel):
    """A question with no answer attached."""

    question_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str


class AnsweredQuestion(UnansweredQuestion):
    """A question with its reference answer and sources."""

    sources: List[MinimalSource]
    answer: str


class RagDataset(BaseModel):
    """A dataset file, holding questions with or without their answers."""

    rag_questions: List[AnsweredQuestion | UnansweredQuestion]


class MinimalSearchResults(BaseModel):
    """The retrieval result for one question."""

    question_id: str
    question: str
    retrieved_sources: List[MinimalSource]


class MinimalAnswer(MinimalSearchResults):
    """A retrieval result with the generated answer added."""

    answer: str


class StudentSearchResults(BaseModel):
    """The output file of a dataset-wide search."""

    search_results: List[MinimalSearchResults]
    k: int


class StudentSearchResultsAndAnswer(BaseModel):
    """The output file of a dataset-wide answer generation."""

    search_results: List[MinimalAnswer]
    k: int
