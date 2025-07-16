import os
from jet.file.utils import load_file, save_file
from jet.logger import logger
from jet.transformers.formatters import format_json
from jet.utils.commands import copy_to_clipboard
from jet.wordnet.phrase_detector import PhraseDetector
from shared.data_types.job import JobData


if __name__ == '__main__':
    data_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/my-jobs/saved/jobs.json"
    data: list[JobData] = load_file(data_file)

    sentences = [
        "\n".join([
            item["title"],
            item["details"],
            "\n".join([
                f"Tech: {tech}"
                for tech in sorted(
                    item["entities"]["technology_stack"],
                    key=str.lower
                )
            ]),
            "\n".join([
                f"Tag: {tech}"
                for tech in sorted(
                    item["tags"],
                    key=str.lower
                )
            ]),
        ])
        for item in data
    ]
    print(f"Number of sentences: {len(sentences)}")

    detector = PhraseDetector(sentences)

    sentences_for_analysis = [
        "Remote WFH Snr Backend Javascript/React Developer\nJob Description\nAt least 4 years experience with backend Javascript and React (React Native, Graphql, SQL and Firebase experience also desirable)\nDesirable - Experience with backend programming for uploading flat files to mall servers, and experience with accounting, BIR and tax calculations such as VAT, NON-VAT, Service charge and discounts\nIn-depth knowledge of modern, JavaScript, Material-ui, Git, ES6, React hooks\nExperience with using Git and working in teams, making Pull Requests\nExperience of providing tech solutions to customer problems with emphasis on user experience\nExperience in optimizing components for maximum performance across a vast array of web-capable devices and browsers\nExperience with automated testing\nAbility to work independently and excellent problem-solving skills\nWe will be reaching out to you via email if you have been shortlisted, so we kindly ask that you check both your email inbox and spam folder.\nNote:\nSalary is negotiable.\n\n## Employer questions\nYour application will include the following questions:\n* What's your expected monthly basic salary?\n* How many years' experience do you have as a React Developer?\n* Which of the following programming languages are you experienced in?\n* Which of the following front end development libraries and frameworks are you proficient in?\n* Which of the following Relational Database Management Systems (RDBMS) are you experienced with?\n* How would you rate your English language skills?",

        "Application Developer (Work from Home)\nApplication Developer (Work from Home)\nDrive innovation in cybersecurity, software development, and big data technology as part of our award-winning IT team. You can play a critical role as part of a global agile team for Bell Canada, Canada's largest telco, media, and tech company.\nWe know our success is fueled by our people, so we empower our Qmunity and provide a workplace where you can flourish and grow. We offer premium benefits, including:\nHome-based setup\nMiscellaneous allowances, performance-based bonus, and yearly increase\nHMO from day 1 for you + 2 free dependents\n6 months paid maternity; 15 days paternity leave\nCompany-sponsored training and upskilling, and career growth opportunities!\n\nYou will have an opportunity to...\nGain a deep understanding of the SmartPort platform from a development perspective.\nWork within a Linux-based environment in the Bell OpenStack infrastructure.\nNavigate and contribute to an enterprise application spanning multiple servers and environments.\nAddress and resolve bugs, workflow issues, and perform code updates.\nDevelop and maintain code primarily in JavaScript (native, React, web environment) with light Python code as needed.\nWhen the environment is learned and comfortable, we can move to heavier tasks like upgrades and new functionality.\nMinimum Required Skills & Experience:\n5+ years of experience with Javascript, React, NodeJS, and a solid understanding of HTML, CSS, and other web development components.\n3 to 5 years of experience with SQL, tracking down data problems, and building applications on an SQL DB while querying/interrogating databases through APIs.\n3 to 5 years of experience working in a Linux environment, with the ability to document work and processes effectively.\n3 to 5 years of years of experience in troubleshooting and debugging in Python, as well as general Python scripting or procedural coding.\nYou will thrive in this role if you have...\nExperience with Git for version control.\nKnowledge of containerization technologies (e.g., Docker, Kubernetes).\nFamiliarity with PostGIS for spatial database extensions.\nExperience with web mapping technologies and tools.\n\nIf this role sounds exciting to you, click\nAPPLY NOW!\n\n\n#LI-LD1",

        "Senior Web Developer (Payload CMS)\nAt De Novo Digital, we don't just build websites--we craft cutting-edge digital experiences that elevate brands and captivate audiences. We're seeking an experienced and visionary Senior Web Developer to lead web development projects using Payload CMS, creating exceptional, user-friendly solutions.\n\n\nKey Responsibilities :\n\nAs a Senior Web Developer (Payload CMS), you'll play a pivotal role at De Novo Digital, turning innovation into excellence. Your responsibilities will include:\n\n\nSetting up and configuring Payload CMS for new projects, ensuring seamless performance and robust security.\n\n\nDefining data schemas with Payload's collection and field configurations to tailor solutions to client needs.\n\n\nCustomizing the admin panel for intuitive content management systems.\n\n\nIntegrating Payload CMS with front-end frameworks like Next.js for visual storytelling and high-performance delivery.\n\n\nManaging user authentication and access control to guarantee security.\n\n\nDeveloping custom APIs, extensions, and features to enhance functionality and scalability.\n\n\nTroubleshooting and problem-solving Payload CMS-related issues with exceptional efficiency.\n\n\nRequired Qualifications\n\nTo excel in this role, you'll need to bring confidence, creativity, and deep technical expertise, along with the ability to make complex concepts approachable. Essential qualifications include:\n\n\nProficiency in Typescript to build scalable, type-safe applications.\n\n\nExpertise in Node.js, navigating server-side development with ease.\n\n\nSkilled in React, empowering seamless admin panel customization.\n\n\nExperience with MongoDB, understanding its NoSQL architecture or similar databases like PostgreSQL.\n\n\nAdvanced knowledge of headless CMS concepts and APIs (REST/GraphQL).\n\n\nFront-end expertise, preferably with Next.js, to merge design and functionality effortlessly.\n\n\nStrong problem-solving skills, independence, and top-tier communication in English, ensuring collaboration with teams and international clients.\n\n\nPreferred Experience:\n\nWhile not mandatory, these qualifications will set you apart as a leader in your field:\n\n\nDirect experience with Payload CMS or similar platforms like Strapi and Directus.\n\n\nKnowledge of UI/UX design principles to craft user-friendly digital environments.\n\n\nFamiliarity with testing frameworks like Jest or Cypress to ensure the highest code quality.\n\n\nCertifications in related technologies (e.g., AWS, React) are a plus.\n\n\nWhy Join De Novo Digital?\n\nAt De Novo Digital, we thrive on innovation and creativity. Our team collaborates to deliver bold solutions and personalized service to clients ranging from startups to established global brands. You'll have the opportunity to work on impactful projects, develop industry-changing websites, and be part of a dynamic culture that's as supportive as it is inspiring.\n\nTransform ideas into game-changing digital solutions. Come be a part of something extraordinary.\n\n\nExcited to build the future with us?\n\nApply now and take the next step in your web development career with De Novo Digital.",
    ]
    phrases_stream = detector.detect_phrases(sentences_for_analysis)

    for item in phrases_stream:
        sentence = item["sentence"]
        phrases = item["phrases"]
        results = item["results"]

        print(
            f"Sentence: {sentence}\nPhrases: {phrases}\nResults: {results}\n")

    phrase_grams = detector.get_phrase_grams()

    queries = [
        # "Mobile development",
        # "Web development",
        # "React Native",
        # "React.js",
        # "Node.js",
        "react native",
        "react developer",
        "react.js",
        "react",
        "mobile",
        "node",
        "web ",
    ]

    results_dict = {query: [] for query in queries}
    for phrase, score in phrase_grams.items():
        for query in queries:
            if query in phrase:
                results_dict[query].append({
                    "phrase": phrase,
                    "score": score,
                })

    logger.success(format_json(results_dict))
    logger.debug(f"Phrase grams: {len(phrase_grams)}")

    output_dir = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
    save_file(queries, f"{output_dir}/queries.json")
    save_file(results_dict, f"{output_dir}/results_dict.json")
    save_file(phrase_grams, f"{output_dir}/phrase_grams.json")
