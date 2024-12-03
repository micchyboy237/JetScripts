import pprint

from langchain_community.utilities import SearxSearchWrapper

search = SearxSearchWrapper(searx_host="http://127.0.0.1:8080")

# answer = search.run("What is the capital of France")
# pprint.pprint(answer)

results = search.results(
    "How many seasons does”I’ll Become a Villainess Who Goes Down in History” have? Is it finished?",
    num_results=10,
    categories="science",
    time_range="year",
)
pprint.pp(results)
