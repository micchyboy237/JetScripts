from jet.logger import logger
from jet.transformers.formatters import format_json
import nltk
from nltk import ne_chunk, pos_tag, word_tokenize
from nltk.tree import Tree

nltk.download('maxent_ne_chunker_tab')


def get_continuous_chunks(text):
    chunked = ne_chunk(pos_tag(word_tokenize(text)))
    current_chunk = []
    contiguous_chunk = []
    contiguous_chunks = []

    for i in chunked:
        print(f"{type(i)}: {i}")
        if type(i) == Tree:
            current_chunk = ' '.join([token for token, pos in i.leaves()])
            # Apparently, Tony and Morrison are two separate items,
            # but "Random House" and "New York City" are single items.
            contiguous_chunk.append(current_chunk)
        else:
            # discontiguous, append to known contiguous chunks.
            if len(contiguous_chunk) > 0:
                contiguous_chunks.append(' '.join(contiguous_chunk))
                contiguous_chunk = []
                current_chunk = []

    return contiguous_chunks


if __name__ == '__main__':
    my_sent = "Job Description\nAt least 4 years experience with backend Javascript and React (React Native, Graphql, SQL and Firebase experience also desirable)\nDesirable - Experience with backend programming for uploading flat files to mall servers, and experience with accounting, BIR and tax calculations such as VAT, NON-VAT, Service charge and discounts\nIn-depth knowledge of modern, JavaScript, Material-ui, Git, ES6, React hooks\nExperience with using Git and working in teams, making Pull Requests\nExperience of providing tech solutions to customer problems with emphasis on user experience\nExperience in optimizing components for maximum performance across a vast array of web-capable devices and browsers\nExperience with automated testing\nAbility to work independently and excellent problem-solving skills\nWe will be reaching out to you via email if you have been shortlisted, so we kindly ask that you check both your email inbox and spam folder.\nNote:\nSalary is negotiable.\n\n## Employer questions\nYour application will include the following questions:\n* What's your expected monthly basic salary?\n* How many years' experience do you have as a React Developer?\n* Which of the following programming languages are you experienced in?\n* Which of the following front end development libraries and frameworks are you proficient in?\n* Which of the following Relational Database Management Systems (RDBMS) are you experienced with?\n* How would you rate your English language skills?"

    print()
    contig_chunks = get_continuous_chunks(my_sent)
    logger.log("INPUT: My sentence:\n", my_sent, colors=["GRAY", "INFO"])
    logger.debug("ANSWER: My contiguous chunks:")
    logger.success(format_json(contig_chunks))
