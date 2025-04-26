from mlx_lm import stream_generate
from mlx_lm.tokenizer_utils import TokenizerWrapper
from transformers import PreTrainedTokenizer
from typing import Union, List, Dict, Optional
import mlx.core as mx


def get_max_context_length(model: 'nn.Module', max_kv_size: Optional[int] = None) -> int:
    """
    Retrieve the maximum context length of the model (input + output tokens).

    Args:
        model (nn.Module): The MLX model.
        max_kv_size (Optional[int]): The maximum key-value cache size, if specified.

    Returns:
        int: The maximum context length (in tokens).
    """
    max_context_length = get_hidden_size(model)

    # If max_kv_size is specified and smaller, it limits the context length
    if max_kv_size is not None and max_kv_size < max_context_length:
        max_context_length = max_kv_size
        print(
            f"Max context length limited by max_kv_size: {max_context_length}")

    return max_context_length


def get_hidden_size(model: 'nn.Module') -> int:
    """
    Retrieve the hidden size (embedding dimension) of the model.

    Args:
        model (nn.Module): The MLX model.

    Returns:
        int: The hidden size of the model.

    Raises:
        AttributeError: If neither hidden_size nor n_embd is found in the model configuration.
    """
    try:
        hidden_size = model.model.args.hidden_size
        return hidden_size
    except AttributeError:
        raise AttributeError(
            "Neither hidden_size nor n_embd found in model configuration.")


def get_prompt_token_count(
    tokenizer: Union[PreTrainedTokenizer, TokenizerWrapper],
    prompt: Union[str, mx.array, List[int]],
    add_special_tokens: bool = True
) -> int:
    """
    Calculate the token count for a given prompt.

    Args:
        tokenizer (Union[PreTrainedTokenizer, TokenizerWrapper]): The tokenizer.
        prompt (Union[str, mx.array, List[int]]): The input prompt (string, token array, or token list).
        add_special_tokens (bool): Whether to add special tokens (e.g., BOS) during encoding.

    Returns:
        int: The number of tokens in the prompt.
    """
    if not isinstance(tokenizer, TokenizerWrapper):
        tokenizer = TokenizerWrapper(tokenizer)

    if isinstance(prompt, str):
        tokens = tokenizer.encode(
            prompt, add_special_tokens=add_special_tokens)
    elif isinstance(prompt, mx.array):
        tokens = prompt
    else:
        tokens = mx.array(prompt)

    return tokens.size if isinstance(tokens, mx.array) else len(tokens)


def get_messages_token_count(
    tokenizer: Union[PreTrainedTokenizer, TokenizerWrapper],
    messages: List[Dict[str, str]],
    chat_template_config: Optional[Dict] = None,
    add_special_tokens: bool = False,
    continue_final_message: bool = False,
    add_generation_prompt: bool = True
) -> int:
    """
    Calculate the token count for a list of messages, applying the chat template if available.

    Args:
        tokenizer (Union[PreTrainedTokenizer, TokenizerWrapper]): The tokenizer.
        messages (List[Dict[str, str]]): List of messages with 'role' and 'content' keys.
        chat_template_config (Optional[Dict]): Additional config for chat template.
        add_special_tokens (bool): Whether to add special tokens during encoding.
        continue_final_message (bool): Whether to continue the final message (for prefill).
        add_generation_prompt (bool): Whether to add a generation prompt.

    Returns:
        int: The total number of tokens for the messages.
    """
    if not isinstance(tokenizer, TokenizerWrapper):
        tokenizer = TokenizerWrapper(tokenizer)

    chat_template_config = chat_template_config or {}

    if tokenizer.chat_template is not None:
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            continue_final_message=continue_final_message,
            add_generation_prompt=add_generation_prompt,
            **chat_template_config
        )
        tokens = tokenizer.encode(
            prompt, add_special_tokens=add_special_tokens)
    else:
        prompt = "".join(message["content"] for message in messages)
        tokens = tokenizer.encode(
            prompt, add_special_tokens=add_special_tokens)

    return tokens.size if isinstance(tokens, mx.array) else len(tokens)


def get_individual_message_token_counts(
    tokenizer: Union[PreTrainedTokenizer, TokenizerWrapper],
    messages: List[Dict[str, str]],
    add_special_tokens: bool = False
) -> List[Dict[str, Union[str, int]]]:
    """
    Calculate the token count for each message individually.

    Args:
        tokenizer (Union[PreTrainedTokenizer, TokenizerWrapper]): The tokenizer.
        messages (List[Dict[str, str]]): List of messages with 'role' and 'content' keys.
        add_special_tokens (bool): Whether to add special tokens during encoding.

    Returns:
        List[Dict[str, Union[str, int]]]: List of dictionaries with 'role', 'content', and 'token_count'.
    """
    if not isinstance(tokenizer, TokenizerWrapper):
        tokenizer = TokenizerWrapper(tokenizer)

    result = []
    for message in messages:
        tokens = tokenizer.encode(
            message["content"], add_special_tokens=add_special_tokens)
        token_count = tokens.size if isinstance(
            tokens, mx.array) else len(tokens)
        result.append({
            "role": message["role"],
            "content": message["content"],
            "token_count": token_count
        })
    return result


def get_response_token_count(
    model: 'nn.Module',
    tokenizer: Union[PreTrainedTokenizer, TokenizerWrapper],
    prompt: Union[str, mx.array, List[int]],
    max_tokens: int = 100,
    **kwargs
) -> tuple[str, int]:
    """
    Calculate the token count for the generated response.

    Args:
        model (nn.Module): The MLX model.
        tokenizer (Union[PreTrainedTokenizer, TokenizerWrapper]): The tokenizer.
        prompt (Union[str, mx.array, List[int]]): The input prompt.
        max_tokens (int): Maximum number of tokens to generate.
        **kwargs: Additional arguments passed to stream_generate (e.g., sampler, draft_model).

    Returns:
        tuple[str, int]: The generated text and the number of tokens in the response.
    """
    if not isinstance(tokenizer, TokenizerWrapper):
        tokenizer = TokenizerWrapper(tokenizer)

    text = ""
    response_token_count = 0

    for response in stream_generate(model, tokenizer, prompt, max_tokens=max_tokens, **kwargs):
        text += response.text
        response_token_count = response.generation_tokens
        if response.finish_reason in ["stop", "length"]:
            break

    return text, response_token_count


def main1():
    from mlx_lm import load
    from jet.logger import logger

    # Load model and tokenizer
    model_path = "mlx-community/gemma-3-1b-it-qat-4bit"
    model, tokenizer = load(model_path)

    def sample_get_max_context_length():
        # Get max context length
        max_context = get_max_context_length(model)
        print(f"Maximum context length: {max_context} tokens")

        # Get max context length with a specified max_kv_size
        max_kv_size = 2048
        max_context_limited = get_max_context_length(
            model, max_kv_size=max_kv_size)
        print(
            f"Maximum context length (limited by max_kv_size): {max_context_limited} tokens")

        # Validate prompt
        prompt = "Write a detailed summary of the history of artificial intelligence."
        prompt_tokens = get_prompt_token_count(
            tokenizer, prompt, add_special_tokens=True)
        max_tokens_to_generate = 500
        total_tokens = prompt_tokens + max_tokens_to_generate

        if total_tokens > max_context:
            print(
                f"Warning: Total tokens ({total_tokens}) exceed max context length ({max_context}).")
        else:
            print(
                f"Total tokens ({total_tokens}) are within max context length ({max_context}).")

    def sample_get_prompt_token_count():
        # Example 1: String prompt with special tokens
        prompt_str = "Hello, how can I assist you today?"
        token_count_str = get_prompt_token_count(
            tokenizer, prompt_str, add_special_tokens=True)
        print(
            f"Token count for string prompt (with special tokens): {token_count_str}")

        # Example 2: String prompt without special tokens
        token_count_str_no_special = get_prompt_token_count(
            tokenizer, prompt_str, add_special_tokens=False)
        print(
            f"Token count for string prompt (without special tokens): {token_count_str_no_special}")

        # Example 3: Token list prompt
        prompt_tokens = tokenizer.encode(
            prompt_str, add_special_tokens=False)  # Encode to list
        token_count_list = get_prompt_token_count(tokenizer, prompt_tokens)
        print(f"Token count for token list prompt: {token_count_list}")

        # Example 4: mx.array prompt
        prompt_array = mx.array(prompt_tokens)
        token_count_array = get_prompt_token_count(tokenizer, prompt_array)
        print(f"Token count for mx.array prompt: {token_count_array}")

    def sample_get_messages_token_count():
        # Define sample messages
        messages = [
            {"role": "system", "content": "You are a helpful AI assistant specializing in history."},
            {"role": "user", "content": "Tell me about the Renaissance period."}
        ]

        # Example 1: Token count with chat template
        token_count_chat = get_messages_token_count(
            tokenizer,
            messages,
            add_special_tokens=False,
            add_generation_prompt=True
        )
        print(
            f"Token count for messages (with chat template): {token_count_chat}")

        # Example 2: Token count without chat template (concatenated content)
        tokenizer.chat_template = None  # Disable chat template for this example
        token_count_no_chat = get_messages_token_count(
            tokenizer,
            messages,
            add_special_tokens=False
        )
        print(
            f"Token count for messages (without chat template): {token_count_no_chat}")

        # Example 3: Token count with prefill response (continuing final message)
        messages_with_prefill = messages + \
            [{"role": "assistant", "content": "The Renaissance was..."}]
        token_count_prefill = get_messages_token_count(
            tokenizer,
            messages_with_prefill,
            add_special_tokens=False,
            continue_final_message=True,
            add_generation_prompt=False
        )
        print(f"Token count for messages with prefill: {token_count_prefill}")

        # Example 4: Token count with custom chat template config
        chat_template_config = {"bos_token": "<s>", "eos_token": "</s>"}
        token_count_custom = get_messages_token_count(
            tokenizer,
            messages,
            chat_template_config=chat_template_config,
            add_special_tokens=True
        )
        print(
            f"Token count for messages with custom config: {token_count_custom}")

    logger.info("\nRunning sample_get_max_context_length...")
    sample_get_max_context_length()
    logger.info("\nRunning sample_get_prompt_token_count...")
    sample_get_prompt_token_count()
    logger.info("\nRunning sample_get_messages_token_count...")
    sample_get_messages_token_count()


def main2():
    from mlx_lm import load
    from mlx_lm.sample_utils import make_sampler
    from jet.logger import logger

    # Load model and tokenizer
    model_path = "mlx-community/gemma-3-4b-it-qat-4bit"
    model, tokenizer = load(model_path)

    # Optional: Load a draft model for speculative decoding
    draft_model_path = "mlx-community/gemma-3-1b-it-qat-4bit"  # Example draft model
    draft_model, draft_tokenizer = load(draft_model_path)

    def sample1():
        # Define prompt
        prompt = "Write a short poem about the moon."

        # Get prompt token count
        prompt_tokens = get_prompt_token_count(
            tokenizer, prompt, add_special_tokens=True)
        print(f"Prompt token count: {prompt_tokens}")

        # Get response token count
        text, response_tokens = get_response_token_count(
            model,
            tokenizer,
            prompt,
            max_tokens=50
        )
        print(f"Generated text: {text}")
        print(f"Response token count: {response_tokens}")

        # Validate against max context
        max_context = get_max_context_length(model)
        total_tokens = prompt_tokens + response_tokens
        print(f"Total tokens used: {total_tokens}/{max_context}")

    def sample2():
        # Define messages
        messages = [
            {"role": "system", "content": "You are a helpful AI assistant specializing in history."},
            {"role": "user", "content": "Tell me about the Renaissance period."}
        ]

        # Get messages token count
        messages_tokens = get_messages_token_count(
            tokenizer,
            messages,
            add_special_tokens=False,
            add_generation_prompt=True
        )
        print(f"Messages token count: {messages_tokens}")

        # Apply chat template to create prompt
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)

        # Get response token count
        text, response_tokens = get_response_token_count(
            model,
            tokenizer,
            prompt,
            max_tokens=100
        )
        print(f"Generated text: {text}")
        print(f"Response token count: {response_tokens}")

        # Validate against max context
        max_context = get_max_context_length(model)
        total_tokens = messages_tokens + response_tokens
        print(f"Total tokens used: {total_tokens}/{max_context}")

    def sample3():
        # Define prompt
        prompt = "Explain the theory of relativity in simple terms."

        # Get prompt token count
        prompt_tokens = get_prompt_token_count(
            tokenizer, prompt, add_special_tokens=True)
        print(f"Prompt token count: {prompt_tokens}")

        # Create a sampler for generation
        sampler = make_sampler(temp=0.7, top_p=0.9,
                               min_p=0.0, min_tokens_to_keep=1)

        # Get response token count with speculative decoding
        text, response_tokens = get_response_token_count(
            model,
            tokenizer,
            prompt,
            max_tokens=80,
            draft_model=draft_model,
            num_draft_tokens=3,  # Speculative decoding with 3 draft tokens
            sampler=sampler
        )
        print(f"Generated text: {text}")
        print(f"Response token count: {response_tokens}")

        # Validate against max context
        max_context = get_max_context_length(model)
        total_tokens = prompt_tokens + response_tokens
        print(f"Total tokens used: {total_tokens}/{max_context}")

    logger.info("\nRunning sample1...")
    sample1()

    logger.info("\nRunning sample2...")
    sample2()

    logger.info("\nRunning sample3...")
    sample3()


if __name__ == "__main__":
    main1()
    main2()
