from codecs import ignore_errors
import json
import os

from jet.llm.summarizer import SummaryResultInfo, summarize_data
from jet.logger import logger


def main():
    input_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/scrapy/generated/passport/_main_preprocessed.json"
    generated_dir = os.path.join(
        "generated",
        os.path.basename(__file__).split('.')[0],
    )
    os.makedirs(generated_dir, exist_ok=True)

    summaries_output_file = f"{generated_dir}/_main_tree_summaries.json"
    final_info_file = f"{generated_dir}/_main_tree_summary_info.json"
    final_summary_file = f"{generated_dir}/_main_tree_final_summary.md"

    with open(input_file, "r") as f:
        text_data = json.load(f)
    content = "\n\n".join(item['content'] for item in text_data).strip()

    summaries: list[SummaryResultInfo] = []

    for result in summarize_data(content):
        if "final_summary" in result:
            with open(final_info_file, "w") as f:
                json.dump(result, f, indent=2)
            with open(final_summary_file, "w") as f:
                f.write(result['final_summary']['summary']['response'])

            logger.success(f"Saved final info to {final_info_file}")
            logger.success(f"Saved final summary to {final_summary_file}")
        else:
            summaries.append(result)

            with open(summaries_output_file, "w") as f:
                json.dump(summaries, f, indent=2)

            logger.success(
                f"Saved summaries ({len(summaries)}) to {summaries_output_file}")


if __name__ == "__main__":
    main()
