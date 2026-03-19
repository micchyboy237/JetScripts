import argparse

from fast_bunkai import FastBunkai

DEFAULT_TEXT = "羽田から✈️出発して、友だちと🍣食べました。最高！また行きたいな😂でも、予算は大丈夫かな…?"

parser = argparse.ArgumentParser(
    description="Split Japanese text into sentences using FastBunkai."
)
parser.add_argument(
    "text",
    nargs="?",
    default=DEFAULT_TEXT,
    help=f"Text to split into sentences (default: '{DEFAULT_TEXT}')",
)
args = parser.parse_args()

splitter = FastBunkai()

sentences = list(splitter(args.text))

print(f"Sentences: {len(sentences)}")
for sent in sentences:
    print(sent)
