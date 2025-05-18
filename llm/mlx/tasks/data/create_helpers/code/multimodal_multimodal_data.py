from typing import List, Dict, Optional
from jet.llm.mlx.config import DEFAULT_MODEL
from jet.llm.mlx.mlx_types import LLMModelType
from jet.llm.mlx.models import resolve_model
from jet.llm.mlx.token_utils import tokenize_strings
from jet.logger import logger
import mlx.core as mx
from mlx_lm import load
from mlx_lm.generate import stream_generate, generate_step
from mlx_lm.sample_utils import make_sampler, make_logits_processors
from mlx_lm.utils import TokenizerWrapper


class MultimodalData:
    def __init__(self, model_path: LLMModelType = DEFAULT_MODEL):
        self.model_path = model_path

    def generate(self, method: str = "stream_generate", max_tokens: int = 10, temperature: float = 0.0, top_p: float = 0.9):
        try:
            # Validate inputs
            validate_method(method)

            # Load model and tokenizer
            model_components = load_model_components(self.model_path)

            # Create and log prompt
            system_prompt = create_system_prompt()
            log_prompt_details(system_prompt, "Question", self.model_path)

            # Format messages and apply chat template
            messages = format_chat_messages(system_prompt, "Question")
            formatted_prompt = model_components.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            # Encode choices and setup generation parameters
            choice_token_map = encode_choices(
                model_components.tokenizer, ["Yes", "No"])
            logits_processors, sampler, stop_tokens = setup_generation_parameters(
                model_components.tokenizer, choice_token_map, temperature, top_p
            )

            # Generate answer based on method
            if method == "stream_generate":
                answer, token_id, _ = generate_answer_stream(
                    model_components, formatted_prompt, max_tokens, logits_processors, sampler, stop_tokens, ["Yes", "No"])
            else:
                answer, token_id, _ = generate_answer_step(
                    model_components, formatted_prompt, max_tokens, logits_processors, sampler, stop_tokens, ["Yes", "No"])

            # Validate the answer
            validate_answer(answer, ["Yes", "No"])

            return AnswerResult(
                answer=answer,
                token_id=token_id,
                is_valid=True,
                method=method,
                error=None
            )
