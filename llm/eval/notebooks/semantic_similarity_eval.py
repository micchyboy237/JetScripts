!pip install llama-indexfrom llama_index.core.evaluation import SemanticSimilarityEvaluator

evaluator = SemanticSimilarityEvaluator()

response = "The sky is typically blue"
reference = """The color of the sky can vary depending on several factors, including time of day, weather conditions, and location.

During the day, when the sun is in the sky, the sky often appears blue. 
This is because of a phenomenon called Rayleigh scattering, where molecules and particles in the Earth's atmosphere scatter sunlight in all directions, and blue light is scattered more than other colors because it travels as shorter, smaller waves. 
This is why we perceive the sky as blue on a clear day.
"""

result = await evaluator.aevaluate(
    response=response,
    reference=reference,
)
print("Score: ", result.score)
print("Passing: ", result.passing)  # default similarity threshold is 0.8
response = "Sorry, I do not have sufficient context to answer this question."
reference = """The color of the sky can vary depending on several factors, including time of day, weather conditions, and location.

During the day, when the sun is in the sky, the sky often appears blue. 
This is because of a phenomenon called Rayleigh scattering, where molecules and particles in the Earth's atmosphere scatter sunlight in all directions, and blue light is scattered more than other colors because it travels as shorter, smaller waves. 
This is why we perceive the sky as blue on a clear day.
"""

result = await evaluator.aevaluate(
    response=response,
    reference=reference,
)
print("Score: ", result.score)
print("Passing: ", result.passing)  # default similarity threshold is 0.8
from llama_index.core.evaluation import SemanticSimilarityEvaluator
from llama_index.core.embeddings import SimilarityMode, resolve_embed_model

embed_model = resolve_embed_model("local")
evaluator = SemanticSimilarityEvaluator(
    embed_model=embed_model,
    similarity_mode=SimilarityMode.DEFAULT,
    similarity_threshold=0.6,
)
response = "The sky is yellow."
reference = "The sky is blue."

result = await evaluator.aevaluate(
    response=response,
    reference=reference,
)
print("Score: ", result.score)
print("Passing: ", result.passing)
