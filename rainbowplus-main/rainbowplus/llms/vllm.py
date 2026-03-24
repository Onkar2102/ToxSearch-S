from pathlib import Path
from typing import List

from vllm import LLM, SamplingParams

from rainbowplus.llms.base import BaseLLM


def _validate_vllm_model_kwargs(model_kwargs: dict) -> None:
    """
    Local GGUF paths are not valid Hugging Face repo ids; vLLM needs an explicit
    tokenizer (usually a Hub id like meta-llama/Meta-Llama-3.1-8B-Instruct) or
    HuggingFace will raise 'Repo id must be in the form ...'.
    """
    model = model_kwargs.get("model")
    if not model:
        return
    m = str(model).strip()
    if m.lower().endswith(".gguf"):
        tok = model_kwargs.get("tokenizer")
        if not tok or not str(tok).strip():
            raise ValueError(
                "For local GGUF weights, set model_kwargs.tokenizer to a Hugging Face model id "
                "for the same instruct family (e.g. meta-llama/Meta-Llama-3.1-8B-Instruct). "
                "See configs/base.yml."
            )


class vLLM(BaseLLM):
    def __init__(self, model_kwargs: dict):
        self.model_kwargs = dict(model_kwargs)
        _validate_vllm_model_kwargs(self.model_kwargs)
        self.llm = LLM(**self.model_kwargs)

    def get_name(self):
        return self.model_kwargs["model"]

    def generate(self, query: str, sampling_params: dict):
        outputs = self.llm.generate([query], SamplingParams(**sampling_params))
        response = outputs[0].outputs[0].text
        return response

    def batch_generate(self, queries: List[str], sampling_params: dict):
        outputs = self.llm.generate(queries, SamplingParams(**sampling_params))
        return [output.outputs[0].text for output in outputs]
