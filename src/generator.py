from typing import List
from transformers import AutoTokenizer, AutoModelForCausalLM
from pydantic import BaseModel


class Generator:
    def __init__(self, model_name: str) -> None:

        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

    def build_prompt(self, query: str, chunks: List[dict],
                     max_sources: int = 5) -> str:

        setup: str = 'Answer the following question based ONLY on the provided context. Do not make up information or use external knowledge.\n\n'

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
        answer = self.tokenizer.decode(generated, skip_special_tokens=True)

        return answer.strip()
