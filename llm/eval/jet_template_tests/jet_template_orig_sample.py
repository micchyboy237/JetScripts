!pip install llama-index

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
print("Passing: ", result.passing)
