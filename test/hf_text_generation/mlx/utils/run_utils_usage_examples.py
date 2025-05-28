import mlx.core as mx
from mlx_lm import load
from mlx_lm.sample_utils import make_sampler
from jet.llm.mlx.utils.base import (
    get_model_max_tokens,
    get_hidden_size,
    get_prompt_token_count,
    get_messages_token_count,
    get_individual_message_token_counts,
    get_response_token_count,
)
from jet.logger import logger


def main1():
    # Load model and tokenizer
    model_path = "mlx-community/gemma-3-1b-it-qat-4bit"
    model, tokenizer = load(model_path)

    def sample_get_model_max_tokens():
        # Get max token length
        max_context = get_model_max_tokens(model_path)
        print(f"Maximum token length: {max_context} tokens")

        # Get max token length with a specified max_kv_size
        max_kv_size = 2048
        max_context_limited = get_model_max_tokens(
            model_path, max_kv_size=max_kv_size)
        print(
            f"Maximum token length (limited by max_kv_size): {max_context_limited} tokens")

        # Validate prompt
        prompt = "Write a detailed summary of the history of artificial intelligence."
        prompt_tokens = get_prompt_token_count(
            tokenizer, prompt, add_special_tokens=True)
        max_tokens_to_generate = 500
        total_tokens = prompt_tokens + max_tokens_to_generate

        if total_tokens > max_context:
            print(
                f"Warning: Total tokens ({total_tokens}) exceed max token length ({max_context}).")
        else:
            print(
                f"Total tokens ({total_tokens}) are within max token length ({max_context}).")

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

    logger.info("\nRunning sample_get_model_max_tokens...")
    sample_get_model_max_tokens()
    logger.info("\nRunning sample_get_prompt_token_count...")
    sample_get_prompt_token_count()
    logger.info("\nRunning sample_get_messages_token_count...")
    sample_get_messages_token_count()


def main2():
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

        # Validate against max tokens
        max_context = get_model_max_tokens(model_path)
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

        # Validate against max tokens
        max_context = get_model_max_tokens(model_path)
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

        # Validate against max tokens
        max_context = get_model_max_tokens(model_path)
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
