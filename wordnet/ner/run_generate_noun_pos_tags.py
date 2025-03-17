from jet.wordnet.sentence import adaptive_split
from jet.wordnet.pos_ner_tagger import POSTaggerProperNouns


if __name__ == '__main__':
    model = POSTaggerProperNouns()
    text = "Job Description\nAt least 4 years experience with backend Javascript and React (React Native, Graphql, SQL and Firebase experience also desirable)\nDesirable - Experience with backend programming for uploading flat files to mall servers, and experience with accounting, BIR and tax calculations such as VAT, NON-VAT, Service charge and discounts\nIn-depth knowledge of modern, JavaScript, Material-ui, Git, ES6, React hooks\nExperience with using Git and working in teams, making Pull Requests\nExperience of providing tech solutions to customer problems with emphasis on user experience\nExperience in optimizing components for maximum performance across a vast array of web-capable devices and browsers\nExperience with automated testing\nAbility to work independently and excellent problem-solving skills\nWe will be reaching out to you via email if you have been shortlisted, so we kindly ask that you check both your email inbox and spam folder.\nNote:\nSalary is negotiable.\n\n## Employer questions\nYour application will include the following questions:\n* What's your expected monthly basic salary?\n* How many years' experience do you have as a React Developer?\n* Which of the following programming languages are you experienced in?\n* Which of the following front end development libraries and frameworks are you proficient in?\n* Which of the following Relational Database Management Systems (RDBMS) are you experienced with?\n* How would you rate your English language skills?"

    first_sentence = adaptive_split(text)[0]

    pos_ner_results = model.predict(first_sentence)

    print(f"Results:\n{pos_ner_results}")
