import asyncio
import os
from typing import Optional
from jet.audio.tts_engine import AdvancedTTSEngine
from jet.audio.transcribe_utils import transcribe_file_async, combine_audio_files_async
from jet.llm.mlx.base import MLX
from jet.models.model_registry.transformers.mlx_model_registry import MLXModelRegistry
from jet.models.model_types import LLMModelType
from jet.wordnet.sentence import split_sentences
from jet.logger import CustomLogger
from tqdm import tqdm

script_dir = os.path.dirname(os.path.abspath(__file__))
log_file = os.path.join(
    script_dir, f"{os.path.splitext(os.path.basename(__file__))[0]}.log")
logger = CustomLogger(log_file, overwrite=True)


class Agent:
    def __init__(self, name: Optional[str], system_prompt: str, model: LLMModelType = "qwen3-1.7b-4bit", overwrite_db: bool = False, session_id: Optional[str] = None, output_dir: Optional[str] = None) -> None:
        if name is None:
            raise ValueError("Agent name must be provided or set in subclass")
        self.name: str = name
        self.llm: MLX = MLXModelRegistry.load_model(
            model=model,
            session_id=session_id,
            with_history=True,
            overwrite_db=overwrite_db,
            dbname="job_interview_db1",
        )
        self.chat_history = self.llm.history
        self.tts = AdvancedTTSEngine(
            rate=200 if name == "Emma" else 180, output_dir=output_dir)
        self.output_dir = output_dir or script_dir

        self.system_prompt = system_prompt

    async def generate_response(self, external_message: str, **kwargs) -> tuple[str, Optional[asyncio.Task]]:
        content = ""
        buffer = ""
        audio_files = []
        generation_config = {
            "messages": external_message,
            "system_prompt": self.system_prompt,
            "verbose": True,
            **kwargs
        }
        for chunk in self.llm.stream_chat(**generation_config):
            content += chunk["content"]
            buffer += chunk["content"]
            sentences = split_sentences(buffer)
            if len(sentences) > 1:
                for sentence in sentences[:-1]:
                    clean_sentence = sentence.strip()
                    if clean_sentence:
                        file_path = await self.tts.speak_async(clean_sentence, speaker_name=self.name)
                        if file_path and os.path.exists(file_path):
                            audio_files.append(file_path)
                buffer = sentences[-1]
        final_sentence = buffer.strip()
        if final_sentence:
            file_path = await self.tts.speak_async(final_sentence, speaker_name=self.name)
            if file_path and os.path.exists(file_path):
                audio_files.append(file_path)

        combine_task = None
        if audio_files and len(audio_files) > 1:
            async def combine_and_transcribe():
                output_file = self.tts._get_audio_filename(
                    self.name, content, prefix="combined")
                combined_file = await combine_audio_files_async(audio_files, output_file)
                if combined_file:
                    logger.info("Scheduling background transcription")
                    await transcribe_file_async(combined_file, self.output_dir)
            combine_task = asyncio.create_task(combine_and_transcribe())
        elif audio_files:
            logger.info(
                f"Single audio file, skipping combine: {audio_files[0]}")
        return content, combine_task

    def clear_history(self) -> None:
        self.chat_history.clear()

    def cleanup(self):
        self.tts.cleanup()


class Interviewer(Agent):
    def __init__(self, model: LLMModelType = "qwen3-1.7b-4bit", name: Optional[str] = None, output_dir: str = None, **kwargs) -> None:
        name = name or "Emma"
        system_prompt = (
            f"You are {name}, a professional job interviewer for a software engineering position. "
            "Follow a structured agenda to assess the candidate's technical skills, experience, "
            "and problem-solving abilities. Ask one clear, relevant question at a time, following this agenda in order:\n"
            "1. Introduction: Ask the candidate to tell you about themselves and their background.\n"
            "2. Technical Skills: Ask about their experience with Python and how they've used it in a project.\n"
            "3. Problem-Solving: Ask about a challenging technical problem they faced and how they solved it.\n"
            "4. Teamwork: Ask about a time they worked in a team to complete a software project and their role.\n"
            "5. Weaknesses: Ask about a professional weakness they've identified and how they are working to improve it.\n"
            "6. Closing: Ask if they have any questions about the role or the company.\n"
            "Use the chat history to determine which agenda topic to address next, ensuring you progress through the topics in order.\n"
            "\n"
            f"Incorporate the candidate's previous response to maintain a natural conversation flow (e.g., 'Thank you for sharing that, Liam. [Next question]').\n"
            f"Be polite, professional, and concise. After asking the final agenda topic (Closing), wait for the candidate's response.\n"
            "If the candidate raises questions or concerns, address them appropriately and then ask again if they have any further questions or concerns, repeating this process until they have no more.\n"
            "If the candidate indicates they have no further questions or concerns (e.g., 'No questions' or 'I'm good'), end the interview politely and include '[TERMINATE]' in your final message."
        )
        super().__init__(name=name, system_prompt=system_prompt,
                         model=model, output_dir=output_dir, **kwargs)


class Applicant(Agent):
    def __init__(self, model: LLMModelType = "qwen3-1.7b-4bit", name: Optional[str] = None, output_dir: str = None, **kwargs) -> None:
        name = name or "Liam"
        system_prompt = (
            f"You are {name}, a job applicant applying for a software engineering position. "
            "You have a strong background in Python, Java, and web development, with 3 years of experience. "
            f"Respond to the interviewer's questions professionally, concisely, and with relevant details. "
            "If asked about weaknesses, be honest but frame them positively."
        )
        super().__init__(name=name, system_prompt=system_prompt,
                         model=model, output_dir=output_dir, **kwargs)


async def main(max_rounds: int = 10):
    overwrite_db = True
    output_dir = os.path.join(script_dir, "generated", "audio_output")
    interviewer = Interviewer(output_dir=output_dir, overwrite_db=overwrite_db)
    applicant = Applicant(output_dir=output_dir, overwrite_db=overwrite_db)
    playback_overlap = 0.5
    tasks = []
    question = "Start the interview."
    try:
        for current_round in tqdm(range(max_rounds), desc="Interview Rounds", unit="round"):
            logger.debug(f"\nStarting round {current_round + 1}/{max_rounds}")

            # Interviewer's turn
            logger.orange(f"\nTurn {current_round + 1} - Interviewer:")
            question_task = asyncio.create_task(
                interviewer.generate_response(question))
            question, combine_task = await question_task
            logger.success(question)
            if combine_task:
                tasks.append(combine_task)

            # Applicant's turn
            await asyncio.sleep(playback_overlap)
            logger.orange(f"\nTurn {current_round + 1} - Applicant:")
            applicant_task = asyncio.create_task(
                applicant.generate_response(question))
            response, combine_task = await applicant_task
            logger.success(response)
            if combine_task:
                tasks.append(combine_task)

            question = response  # Set the applicant's response as the next input for the interviewer

        # Await all background tasks
        if tasks:
            await asyncio.gather(*tasks)
    finally:
        interviewer.cleanup()
        applicant.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
