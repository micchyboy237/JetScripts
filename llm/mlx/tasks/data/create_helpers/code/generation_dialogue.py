from typing import Dict, List
from jet.llm.mlx.config import DEFAULT_MODEL
from jet.models.model_types import LLMModelType
from jet.llm.mlx.models import resolve_model
from jet.llm.mlx.generate import stream_generate, generate_step
from jet.llm.mlx.sample_utils import make_sampler, make_logits_processors
from jet.logger import logger
import mlx.core as mx
from mlx_lm import load
from mlx_lm.generate import stream_generate, generate_step
from mlx_lm.sample_utils import make_sampler, make_logits_processors
from mlx_lm.utils import TokenizerWrapper

# Set the seed for reproducibility
mx.random.seed(42)


# Custom exceptions for specific error cases
class ModelLoadError(Exception):
    pass


class InvalidMethodError(Exception):
    pass


class InvalidOutputError(Exception):
    pass


# Type definitions for structured data
class ChatMessage(TypedDict):
    role: str
    content: str


class AnswerResult(TypedDict):
    answer: str
    token_id: int
    is_valid: bool
    method: str
    error: Optional[str]


def answer_yes_no(
    question: str,
    model_path: LLMModelType = DEFAULT_MODEL,
    method: str = "stream_generate",
    max_tokens: int = 1,
    temperature: float = 0.1,
    top_p: float = 0.1
) -> AnswerResult:
    model_path = resolve_model(model_path)

    try:
        try:
            model, tokenizer = load(model_path)
        except Exception as e:
            raise ModelLoadError(f"Error loading model or tokenizer: {e}")

        try:
            model, tokenizer = load(model_path)
        except Exception as e:
            raise ModelLoadError(f"Error loading model or tokenizer: {e}")

        if method not in ["stream_generate", "generate_step"]:
            raise InvalidMethodError(
                f"Invalid method specified: {method}. Valid methods: {['stream_generate', 'generate_step']}"
            )

        messages: List[ChatMessage] = [
            {"role": "system", "content": "Answer the following question with only 'Yes' or 'No'. Ensure accuracy."},
            {"role": "user", "content": question}
        ]

        try:
            formatted_prompt: str = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception as e:
            raise InvalidMethodError(f"Error applying chat template: {e}")

        try:
            yes_tokens = tokenizer.encode("Yes", add_special_tokens=False)
            no_tokens = tokenizer.encode("No", add_special_tokens=False)

            yes_token = yes_tokens[0]
            no_token = no_tokens[0]

            # Print the tokens for "Yes" and "No"
            logger.log("Token for 'Yes':",
                       yes_tokens[0], colors=["GRAY", "ORANGE"])
            logger.log("Token for 'No':",
                       no_tokens[0], colors=["GRAY", "ORANGE"])
        except Exception as e:
            raise TokenEncodingError(f"Error encoding tokens: {e}")

        logit_bias: Dict[int, float] = {
            yes_token: 0.0,
            no_token: 0.0,
            **{i: -1e9 for i in range(tokenizer.vocab_size) if i not in [yes_token, no_token]}
        }
        logits_processors = make_logits_processors(logit_bias=logit_bias)
        sampler = make_sampler(temp=temperature, top_p=top_p)
        stop_tokens = tokenizer.encode("\n") + list(tokenizer.eos_token_ids)

        answer = ""
        token_id = -1

        if method == "stream_generate":
            try:
                for output in stream_generate(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=formatted_prompt,
                    max_tokens=max_tokens,
                    logits_processors=logits_processors,
                    sampler=sampler
                ):
                    if output.token in stop_tokens:
                        raise InvalidOutputError(
                            "Generated token is a stop token.")
                    answer += output.text
                    token_id = output.token
                    break
            except Exception as e:
                raise GenerationError(f"Error during stream_generate: {e}")
        else:
            try:
                input_ids = mx.array(tokenizer.encode(
                    formatted_prompt, add_special_tokens=False))
                prompt_cache = None
                for token, _ in generate_step(
                    model=model,
                    prompt=input_ids,
                    max_tokens=max_tokens,
                    logits_processors=logits_processors,
                    sampler=sampler,
                    prompt_cache=prompt_cache
                )
                if token in stop_tokens:
                    raise InvalidOutputError(
                        "Generated token is a stop token.")
                answer = tokenizer.decode([token])
                token_id = token
                break
            except Exception as e:
                raise GenerationError(f"Error during generate_step: {e}")

        if answer.lower() not in ["yes", "no"]:
            raise InvalidOutputError("Output is not 'Yes' or 'No'.")

        return AnswerResult(answer=answer, token_id=token_id, is_valid=True, method=method, error=None)

    except Exception as e:
        return AnswerResult(answer="", token_id=-1, is_valid=False, method=method, error=str(e))
