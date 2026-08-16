"""Answer generation from retrieved chunks, using a local causal LM."""

from typing import Any, List
from transformers import AutoTokenizer, AutoModelForCausalLM


class Generator:
    """Turns retrieved chunks into an answer with a local causal LM."""

    def __init__(self, model_name: str) -> None:
        """Load the tokenizer and the model, once.

        Args:
            model_name: A HuggingFace model id, e.g. "Qwen/Qwen3-0.6B".
        """

        self.model_name = model_name
        self.tokenizer: Any = AutoTokenizer.from_pretrained(model_name)
        self.model: Any = AutoModelForCausalLM.from_pretrained(model_name)

    def build_prompt(self, query: str, chunks: List[dict],
                     max_sources: int = 5) -> str:
        """Assemble the question and its context into a prompt.

        Args:
            query: The question, in plain text.
            chunks: Retrieved chunks. Only file_path and content are read.
            max_sources: How many chunks to include in the context.

        Returns:
            The prompt text.
        """

        setup: str = ('Answer the following question based ONLY on the '
                      'provided context. Do not make up information or use '
                      'external knowledge.\n\n')

        question = f'Question: {query}\n\n'
        context = 'Context:\n'

        prompt = setup + question + context

        sources = []
        i = 1
        for chunk in chunks:
            content: str = ''
            if i == max_sources + 1:
                break
            content += f'\nSource {i}:\n'
            content += chunk['file_path']
            content += '\n\n'
            content += chunk['content']
            sources.append(content)
            i += 1

        sources_text = ''.join(sources)
        prompt += sources_text

        return prompt

    def generate(self, query: str, chunks: List[dict],
                 max_sources: int = 5) -> str:
        """Generate an answer grounded in the retrieved chunks.

        The prompt goes through the model's chat template, which supplies the
        end-of-turn token generation stops on.

        Args:
            query: The question, in plain text.
            chunks: Retrieved chunks.
            max_sources: How many chunks to put in the context.

        Returns:
            The answer text, without the prompt or special tokens.
        """

        prompt = self.build_prompt(query, chunks, max_sources)

        text = self.tokenizer.apply_chat_template(
            [{'role': 'user', 'content': prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

        tokens = self.tokenizer.encode(text, return_tensors='pt')
        answer_tokens = self.model.generate(tokens, max_new_tokens=256)
        generated = answer_tokens[0][tokens.shape[1]:]
        answer: str = self.tokenizer.decode(
            generated, skip_special_tokens=True)

        return answer.strip()
