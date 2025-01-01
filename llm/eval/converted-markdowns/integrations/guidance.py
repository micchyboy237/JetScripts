from jet.logger import logger
from jet.llm.ollama import initialize_ollama_settings
initialize_ollama_settings()

class Song(BaseModel):
    title: str
    length_seconds: int


class Album(BaseModel):
    name: str
    artist: str
    songs: List[Song]

program = GuidancePydanticProgram(
    output_cls=Album,
    prompt_template_str="Generate an example album, with an artist and a list of songs. Using the movie {{movie_name}} as inspiration",
    guidance_llm=Ollama("text-davinci-003"),
    verbose=True,
)

output = program(movie_name="The Shining")

Album(
    name="The Shining",
    artist="Jack Torrance",
    songs=[
        Song(title="All Work and No Play", length_seconds=180),
        Song(title="The Overlook Hotel", length_seconds=240),
        Song(title="The Shining", length_seconds=210),
    ],
)

from llama_index.question_gen.guidance import GuidanceQuestionGenerator
from guidance.llms import Ollama as GuidanceOllama

question_gen = GuidanceQuestionGenerator.from_defaults(
    guidance_llm=GuidanceOllama("text-davinci-003"), verbose=False
)

query_engine_tools = ...

s_engine = SubQuestionQueryEngine.from_defaults(
    question_gen=question_gen,  # use guidance based question_gen defined above
    query_engine_tools=query_engine_tools,
)

logger.info("\n\n[DONE]", bright=True)