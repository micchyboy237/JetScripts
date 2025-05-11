from typing import List, Dict, Optional, TypedDict
import random


class YesNoQuestion(TypedDict):
    """Encapsulates a yes/no question."""
    prompt: str
    choices: List[str]
    answer: Optional[str]


class YesNoPromptGenerator:
    """Generates yes/no prompts for the chat template."""
    
    def __init__(self):
        self.choices = ["yes", "no"]
        
    def generate_prompt(self, prompt: str) -> YesNoQuestion:
        """Generates a yes/no question."""
        
        # Generate random choices
        choice1 = random.choice(self.choices)
        choice2 = random.choice([c for c in self.choices if c!= choice1])
        
        # Create the prompt
        return YesNoQuestion(
            prompt=f"Answer the following question by choosing one of the options provided without any additional text.\nOptions:\n{choice1}\n\n{'\n'.join(choice2)}",
            choices=[choice1, choice2],
        )
    
    def generate_choices(self) -> List[str]:
        """Generates a list of yes/no choices."""
        
        return [random.choice(self.choices) for _ in range(10)]


def yes_no_prompt_generator(prompt: str, choices: List[str]) -> YesNoQuestion:
    """Generates a yes/no prompt."""
    
    return YesNoPromptGenerator().generate_prompt(prompt)


def answer_yes_no(
    question: str,
    model_path: ModelType,
    method: Optional[str] = "stream_generate",
) -> AnswerResult:
    """
    Answers a yes/no question using the language model.

    Args:
        question: The yes/no question to be answered.
        model_path: Path to the language model.

    Returns:
        AnswerResult containing the answer and token ID.
    """
    
    # Generate choices
    yes_no_choices = YesNoPromptGenerator().generate_choices()
    
    # Validate the answer
    valid_answers = [answer for choice in yes_no_choices.choices if choice == question]
    
    # Generate the prompt
    system_prompt = "Answer the following question by choosing one of the options provided without any additional text.\nOptions:\n"
    for choice in yes_no_choices.choices:
        system_prompt += f"{choice}\n\n{'\n'.join([c for c in yes_no_choices.choices if c == choice])) + "\n\n"
    
    # Log the prompt
    logger.gray(system_prompt)
    
    # Generate the answer based on method
    if method == "stream_generate":
        answer, token_id = generate_answer_stream(
            model_components=YesNoPromptGenerator().generate_prompt(prompt),
            formatted_prompt="",
        )
    else:
        answer, token_id = generate_answer_step(
            model_components=YesNoPromptGenerator().generate_prompt(prompt),
        )
    
    # Validate the answer
    validate_answer(answer, yes_no_choices.choices)
    
    return AnswerResult(
        answer=answer,
        token_id=token_id,
        is_valid=True,
    )


# Example usage
model_path = "path/to/model"
question = yes_no_prompt_generator("yes/no", ["yes", "no"]).prompt
answer = answer_yes_no(question, model_path)
print(answer.answer)  # Output: "yes"