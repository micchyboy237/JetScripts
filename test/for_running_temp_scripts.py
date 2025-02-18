from jet.logger import logger
from jet.scrapers.utils import clean_text, extract_paragraphs, extract_sections, extract_sentences, merge_texts, merge_texts_with_overlap
from jet.token.token_utils import get_ollama_tokenizer, token_counter
from llama_index.core.node_parser.text.sentence import SentenceSplitter
from llama_index.core.utils import set_global_tokenizer


SAMPLE_MARKDOWN_TEXT = """
Context information is below.
---------------------
Link: https://www.onlinejobs.ph/jobseekers/job/1198646
Title: Link Building Specialist
Company: Bronwyn Reynolds
Posted Date: 2025-02-18
Salary: $550/month
Job Type: Full Time
Tags: ['']
Domain: onlinejobs.ph
Details: Get Me Links (
getmelinks.com
) is looking to hire a Link Building Specialist.
Certain pre-qualifications are needed, please read through:
YOU ARE:
- Ambitious: Were growing FAST. A thirst for growth and quality of work to match will be compensated.
- Impeccable attention to detail: You read through, take notes, remember details like peoples names, etc. (btw my name is Bronwyn,  and if you send the application to any other name or Sir, Sir/Madam, Hiring Manager, etc., I wont read it ;) ).
-Competent: You are an experienced link builder and have knowledge about SEO.
- An A player: You dont just do your best, you do whats required.
- A team player: With a fast-growing team, everyone has to care about the other tea
m me
mbers success.
- A good communicator: Youll be capturing order details from clients, sometimes with plenty of little nuances / specific requirements. You need to understand clearly what was requested and ensure the message gets across to the fulfillment team clearly.
- Honest: You come forward when you mess up and communicate efficiently with your manager so you can fix the issue together. Everybody makes mistakes, its how we react to them that makes a difference.
- Effective: You get the most out of the hours you work.
YOU VALUE:
- Personal freedom: We dont have fixed schedules and the job is fully remote. You can set your own hours for as long as the job gets done.
- Recognition: We strive to have a company culture where competence and team play is greatly appreciated.
- Structure: While a young and flexible company, we have a defined structure of work and systems to complement them. Youll find an environment that supports your growth
WHAT YOULL FIND at Get Me Links:
- A friendly team: Our team is based in multiple locations, with a great % of the team based in the Philippines. Were all team players and care for our teammates success.
- A supportive environment: the managers and even the C-level team will be there for you (both CEO and COO) to support your growth and integration into the team.
- A relaxed atmosphere: Dont confuse growth and ambition with stiffness or stress. We value personal wellbeing above many other things. Youll find a relaxed workplace with few meetings and no silly time tracking or micromanaging bosses. There are KPIs in place that we use to know if works on track or not. We care about results, not minutes.
- Work/life balance: You wont be asked to go the extra mile with your hours.
SKILLS & EXPERIENCE NEEDED
- Fluent spoken and written English absolutely required.
- 2 years+ previous experience as a link builder or SEO specialist
- Proficiency in Excel/Google Sheets is desired
- Script (Google Sheets) development skills are highly valued although not necessarily required.
JOB CONDITIONS:
- $550 USD per month as a starting salary. One-month training period. six months probationary
- Full Social Benefits Package after trial period.period (PAGIBIG, SS, Philhealth)
- 13th month Payment (eligible after trial period) paid on December 15th
- Paid holidays (national holidays + Christmas to New Year week + 2 weeks per year)
NO SIDE-GIG POLICY
We have a strict no side-gig policy. Youll have to drop any other job, be it full-time or part-time and any freelancing gigs you are doing if you take a position with us.
HOW TO APPLY:
- Send the following documents to
bronwyn@getmelinks.com:
 a copy of your resume, at least two contacts of reference (name, work relationship with you, and contact method; we WILL conduct a background check), and the full breakdown of your DISC personality as per
www.123test.com
/disc-personality-test/ (do a new test and send a capture of the final result).
- Use the following subject line for your
 email
: Link Building Specialist for GetMeLinks - Im [Here goes your name] and Im an A player
- After you submit your test task (youll have 24h to do so), well review and well start conducting interviews with people who pass phase 1. There will be 1-2 more interviews if you give the first one.
**PLEASE NOTE: Only applications submitted exactly as per the instructions will be considered.
We're looking forward to your application!
Thank you very much.
---------------------

Given the context information and not prior knowledge, answer the query.

Query: Extract the data from the job posting.

Return only a single generated JSON value without any explanations surrounded by ```json that adheres to the model below:
```json
{'$defs': {'ApplicationProcess': {'properties': {'applicationLinks': {'anyOf': [{'items': {'format': 'uri', 'maxLength': 2083, 'minLength': 1, 'type': 'string'}, 'type': 'array'}, {'type': 'null'}], 'default': None, 'description': 'List of URLs for application submission', 'title': 'Applicationlinks'}, 'contactInfo': {'anyOf': [{'items': {'type': 'string'}, 'type': 'array'}, {'type': 'null'}], 'default': None, 'description': 'List of recruiter or HR contact details', 'title': 'Contactinfo'}, 'instructions': {'anyOf': [{'items': {'type': 'string'}, 'type': 'array'}, {'type': 'null'}], 'default': None, 'description': 'List of instructions on how to apply', 'title': 'Instructions'}}, 'title': 'ApplicationProcess', 'type': 'object'}, 'Compensation': {'properties': {'salaryRange': {'anyOf': [{'$ref': '#/$defs/SalaryRange'}, {'type': 'null'}], 'default': None, 'description': 'Salary range details'}, 'benefits': {'anyOf': [{'items': {'type': 'string'}, 'type': 'array'}, {'type': 'null'}], 'default': None, 'description': 'List of benefits (e.g., Health Insurance, Paid Time Off)', 'title': 'Benefits'}}, 'title': 'Compensation', 'type': 'object'}, 'Location': {'properties': {'city': {'anyOf': [{'type': 'string'}, {'type': 'null'}], 'default': None, 'description': 'City where the job is located', 'title': 'City'}, 'state': {'anyOf': [{'type': 'string'}, {'type': 'null'}], 'default': None, 'description': 'State where the job is located', 'title': 'State'}, 'country': {'anyOf': [{'type': 'string'}, {'type': 'null'}], 'default': None, 'description': 'Country where the job is located', 'title': 'Country'}, 'remote': {'anyOf': [{'type': 'boolean'}, {'type': 'null'}], 'default': None, 'description': 'Indicates if remote work is allowed', 'title': 'Remote'}}, 'title': 'Location', 'type': 'object'}, 'Qualifications': {'properties': {'mandatory': {'anyOf': [{'items': {'type': 'string'}, 'type': 'array'}, {'type': 'null'}], 'default': None, 'description': 'Required qualifications, skills, and experience', 'title': 'Mandatory'}, 'preferred': {'anyOf': [{'items': {'type': 'string'}, 'type': 'array'}, {'type': 'null'}], 'default': None, 'description': 'Preferred but not mandatory qualifications', 'title': 'Preferred'}}, 'title': 'Qualifications', 'type': 'object'}, 'SalaryRange': {'properties': {'min': {'anyOf': [{'type': 'integer'}, {'type': 'null'}], 'default': None, 'description': 'Minimum salary', 'title': 'Min'}, 'max': {'anyOf': [{'type': 'integer'}, {'type': 'null'}], 'default': None, 'description': 'Maximum salary', 'title': 'Max'}, 'currency': {'anyOf': [{'type': 'string'}, {'type': 'null'}], 'default': None, 'description': 'Currency of the salary (e.g., USD, EUR)', 'title': 'Currency'}}, 'title': 'SalaryRange', 'type': 'object'}, 'WorkArrangement': {'properties': {'schedule': {'anyOf': [{'type': 'string'}, {'type': 'null'}], 'default': None, 'description': 'Work schedule (e.g., Flexible, Fixed, Shift-based)', 'title': 'Schedule'}, 'hoursPerWeek': {'anyOf': [{'type': 'integer'}, {'type': 'null'}], 'default': None, 'description': 'Number of work hours per week', 'title': 'Hoursperweek'}, 'remote': {'anyOf': [{'type': 'boolean'}, {'type': 'null'}], 'default': None, 'description': 'Indicates if remote work is allowed', 'title': 'Remote'}}, 'title': 'WorkArrangement', 'type': 'object'}}, 'properties': {'jobTitle': {'anyOf': [{'type': 'string'}, {'type': 'null'}], 'default': '', 'description': 'Title of the job position', 'title': 'Jobtitle'}, 'jobType': {'anyOf': [{'type': 'string'}, {'type': 'null'}], 'default': '', 'description': 'Type of employment (e.g., Full-Time, Part-Time, Contract, Internship)', 'title': 'Jobtype'}, 'description': {'anyOf': [{'type': 'string'}, {'type': 'null'}], 'default': '', 'description': 'Brief job summary', 'title': 'Description'}, 'qualifications': {'anyOf': [{'$ref': '#/$defs/Qualifications'}, {'type': 'null'}], 'default': None, 'description': 'Job qualifications and requirements'}, 'responsibilities': {'anyOf': [{'items': {'type': 'string'}, 'type': 'array'}, {'type': 'null'}], 'default': None, 'description': 'List of job responsibilities', 'title': 'Responsibilities'}, 'company': {'anyOf': [{'type': 'string'}, {'type': 'null'}], 'default': None, 'description': 'Name of the hiring company or employer', 'title': 'Company'}, 'industry': {'anyOf': [{'type': 'string'}, {'type': 'null'}], 'default': None, 'description': 'Industry related to the job (e.g., Technology, Healthcare, Finance)', 'title': 'Industry'}, 'location': {'anyOf': [{'$ref': '#/$defs/Location'}, {'type': 'null'}], 'default': None, 'description': 'Job location details'}, 'skills': {'anyOf': [{'items': {'type': 'string'}, 'type': 'array'}, {'type': 'null'}], 'default': None, 'description': 'Required technical and soft skills', 'title': 'Skills'}, 'tools': {'anyOf': [{'items': {'type': 'string'}, 'type': 'array'}, {'type': 'null'}], 'default': None, 'description': 'List of required tools, software, or platforms', 'title': 'Tools'}, 'collaboration': {'anyOf': [{'items': {'type': 'string'}, 'type': 'array'}, {'type': 'null'}], 'default': None, 'description': 'Teams or individuals the candidate will work with', 'title': 'Collaboration'}, 'workArrangement': {'anyOf': [{'$ref': '#/$defs/WorkArrangement'}, {'type': 'null'}], 'default': None, 'description': 'Work arrangement details'}, 'compensation': {'anyOf': [{'$ref': '#/$defs/Compensation'}, {'type': 'null'}], 'default': None, 'description': 'Compensation details'}, 'applicationProcess': {'anyOf': [{'$ref': '#/$defs/ApplicationProcess'}, {'type': 'null'}], 'default': None, 'description': 'Details about how to apply'}, 'postedDate': {'description': 'Date when the job was posted', 'format': 'date', 'title': 'Posteddate', 'type': 'string'}}, 'title': 'JobPosting', 'type': 'object', '$schema': 'http://json-schema.org/draft-07/schema#'}
```

Response:
""".strip()

if __name__ == "__main__":
    # Example usage
    # model_max_chars = 32768
    # max_chars = get_max_prompt_char_length(model_max_chars)
    # print(f"Maximum characters for the prompt: {max_chars}")

    # context_file = "generated/drivers_license/_main.md"
    # with open(context_file, 'r') as f:
    #     context = f.read()
    model = "llama3.1"
    context = SAMPLE_MARKDOWN_TEXT
    cleaned_text_content = clean_text(context)

    tokenizer = get_ollama_tokenizer(model)
    set_global_tokenizer(tokenizer)

    token_count1 = len(tokenizer.encode(cleaned_text_content))
    token_count2: int = token_counter(cleaned_text_content, model)
    logger.newline()
    logger.info("Token counts:")
    logger.log("Token count 1:", token_count1, colors=["GRAY", "DEBUG"])
    logger.log("Token count 2:", token_count2, colors=["GRAY", "DEBUG"])

    chunk_size = 2048
    chunk_overlap = 100

    splitter = SentenceSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    text_chunks = splitter.split_text(cleaned_text_content)

    # Extract texts from the content
    sections = extract_sections(context)
    sentences = extract_sentences(context)
    paragraphs = extract_paragraphs(context)

    texts = paragraphs
    print(texts)
    # Print lengths of texts
    print([len(text) for text in texts])

    # Merge texts if it doesn't exceed the maximum number of characters
    # Order should be maintained
    max_chars_chunks = 2000
    max_chars_overlap = 200
    merged_texts = merge_texts(texts, max_chars_chunks)
    merged_texts_with_overlap = merge_texts_with_overlap(
        merged_texts, max_chars_overlap)
    print(merged_texts)
    print(merged_texts_with_overlap)
    # Print lengths of merged texts
    print([len(text) for text in merged_texts])

    # Get texts with the most and least number of characters
    sorted_texts = sorted(merged_texts, key=len)
    print(
        f"Least number of characters ({len(sorted_texts[0])} characters):\n{sorted_texts[0]}")
    print(
        f"Most number of characters ({len(sorted_texts[-1])} characters):\n{sorted_texts[-1]}")
