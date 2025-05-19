import re
from dataset import load_data, load_data_from_directories, save_data
from words import get_words, count_words, count_non_words
from tqdm import tqdm


def get_distributions(data, categories=[], filters=[]):
    def apply_filters_or(item, filters):
        if not filters:
            return True  # No filtering applied if no filters are provided

        passed = False
        for f in filters:
            parts = f.split(":")

            excludes = False
            if len(parts) == 4:
                # Only handle excludes filter
                allow, attribute, _, text = parts
                attributes = attribute.split("|")

                if allow == "excludes":
                    # Use re to match exact words
                    for attribute in attributes:
                        if excludes:
                            break
                        attribute_value = str(item.get(attribute, '')).lower()
                        text = text.lower()

                        excludes = text in attribute_value
                    if excludes:
                        break

                # Convert to 3 parts
                parts = parts[1:]

            if not excludes and len(parts) == 3:
                attribute, filter_type, text = parts
                attributes = attribute.split("|")

                if filter_type in ["contains", "starts", "ends"]:
                    # Use re to match exact words
                    for attribute in attributes:
                        # Check if attribute ends with a question mark, then continue
                        if attribute.endswith("?"):
                            continue
                        attribute_value = str(item.get(attribute, '')).lower()
                        text = text.lower()

                        if filter_type == "contains" and re.search(rf"\b{text}\b", attribute_value):
                            passed = True
                            break
                        elif filter_type == "starts" and attribute_value.startswith(text):
                            passed = True
                            break
                        elif filter_type == "ends" and attribute_value.endswith(text):
                            passed = True
                            break

                elif passed and filter_type == "min_words":
                    # Check if combined attributes word count is greater than or equal to the minimum word count
                    combined_text = " ".join(
                        [str(item.get(attribute, '')) for attribute in attributes])
                    passed = count_words(combined_text) >= int(text)

                elif passed and filter_type == "max_non_words":
                    # Check if combined attributes non-word count is less than or equal to the maximum non-word count
                    combined_text = " ".join(
                        [str(item.get(attribute, '')) for attribute in attributes])
                    passed = count_non_words(combined_text) <= int(text)
        return passed

    filtered_data = []

    for item in tqdm(data, desc="Filtering data", unit="item"):
        category = item.get('category', '')
        if ((not categories or category in categories) and
                (not filters or apply_filters_or(item, filters))):
            # Add new "keywords" field to the item that contains all the matching keywords based on the filters
            item['keywords'] = set()
            for f in filters:
                parts = f.split(":")
                if len(parts) == 3:
                    attribute, filter_type, text = parts
                    attributes = attribute.split("|")

                    # Use re to match exact words
                    for attribute in attributes:
                        # Check if attribute ends with a question mark, then remove it
                        if attribute.endswith("?"):
                            attribute = attribute[:-1]

                        attribute_value = str(item.get(attribute, '')).lower()
                        text = text.lower()

                        if filter_type == "contains" and re.search(rf"\b{text}\b", attribute_value):
                            item['keywords'].add(text)
                        elif filter_type == "starts" and attribute_value.startswith(text):
                            item['keywords'].add(text)
                        elif filter_type == "ends" and attribute_value.endswith(text):
                            item['keywords'].add(text)
            item['keywords'] = list(item['keywords'])
            filtered_data.append(item)

    # Calculate the category distribution in filtered data
    category_counts = {}
    for item in filtered_data:
        category = item.get('category', '')
        category_counts[category] = category_counts.get(category, 0) + 1

    # Log the results
    total_filtered_count = len(filtered_data)
    print(f"\nTotal Filtered Data: {total_filtered_count}")
    print("Category Distribution in Filtered Data:")
    for category, count in category_counts.items():
        percentage = count / total_filtered_count * \
            100 if total_filtered_count > 0 else 0
        print(f"{category}: {count} ({percentage:.2f}%)")

    return filtered_data


# Example usage
if __name__ == "__main__":
    directories = ['data/json']
    includes = ["alpaca_gpt4_data_en_classified.json"]
    excludes = []

    data = load_data_from_directories(directories, includes, excludes)

    # tl_data = []

    # for d in data:
    #     data_obj = {
    #         "category": d["category"],
    #         "instruction": d["tl"]["instruction"],
    #         "input": d["tl"]["input"],
    #         "output": d["tl"]["output"]
    #     }

    #     tl_data.append(data_obj)

    results = get_distributions(
        data=data,
        # categories=[
        #     "question and answering",
        #     "list, name and sequence"
        # ],
        filters=[
            "excludes:instruction|input|output:contains://",
            "excludes:instruction|input|output:contains:joke",
            "excludes:instruction|input|output:contains:jokes",
            # "instruction|input|output:min_words:10",
            # "instruction|input|output:max_non_words:90",
            "instruction:contains:conversation",
            "instruction:contains:conversations",
            "instruction:contains:dialogue",
            "instruction:contains:dialogue",
            "instruction:contains:dialog",
            "instruction:contains:dialogs",
            "instruction|input|output:contains:assistant",
            "instruction|input|output:contains:bot",
            "instruction|input|output?:contains:chatbot",
            "instruction|input|output?:contains:chatbots",
            "instruction|input|output?:contains:resume",
            "instruction|input|output?:contains:cv",
            "instruction|input|output?:contains:curriculum vitae",
            "instruction|input|output?:contains:app",
            "instruction|input|output?:contains:developer",
            "instruction|input|output?:contains:ai",
            "instruction|input|output?:contains:artificial intelligence",
            "instruction|input|output?:contains:software",
            "instruction|input|output?:contains:interview",
            "instruction|input|output?:contains:interviewer",
            "instruction|input|output?:contains:interviewee",
            "instruction|input|output?:contains:web application",
            "instruction|input|output?:contains:mobile application",
            "instruction|input|output?:contains:native application",
            "instruction|input|output?:contains:web applications",
            "instruction|input|output?:contains:mobile applications",
            "instruction|input|output?:contains:native applications",
        ]
    )

    # Replace all "input" attributes that contains "No input" with an empty string
    for item in results:
        # Make sure to check if these are the first 2 words in the input, remove all special characters and make it lower case before comparing
        if re.sub(r'\W+', '', item['input'].lower()).startswith("noinput"):
            item['input'] = ""

    # Include Jet Resume Data
    jet_resume_data = load_data(
        'server/static/models/dost-asti-gpt2/base_model/datasets/train/jet_resume_chatgpt.json')
    jet_resume_val_data = load_data(
        'server/static/models/dost-asti-gpt2/base_model/datasets/train/jet_resume_chatgpt.val.json')

    # jet_resume_data = jet_resume_data + jet_resume_val_data
    for item in jet_resume_data:
        item['tags'] = ['jet-resume']

    for item in jet_resume_val_data:
        item['tags'] = ['jet-resume']

    results = results + jet_resume_data

    output_file = 'server/static/models/dost-asti-gpt2/base_model/datasets/base/jet_resume_alpaca.json'
    save_data(output_file, results, write=True)

    # train_output_file = '/Users/jethroestrada/Desktop/External_Projects/GPT/lit-gpt/data/jet_resume/train.json'
    # save_data(train_output_file, jet_resume_data, write=True)
    # val_output_file = '/Users/jethroestrada/Desktop/External_Projects/GPT/lit-gpt/data/jet_resume/val.json'
    # save_data(val_output_file, jet_resume_val_data, write=True)

    # histogram_file_path = 'instruction_generator/datasets/histogram_alpaca_representative.json'
    # questions_data = []
    # for item in results:
    #     questions_data.append(item['instruction'])

    # from_start = True
    # top_n = 30
    # ngram_ranges = [(3, 6), (5, 8)]

    # (questions_data, top_n, ngram_ranges,
    #                    histogram_file_path, from_start)
