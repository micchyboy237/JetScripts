from jet.data.header_utils._prepare_for_rag import preprocess_text
from jet.logger import logger
from jet.utils.commands import copy_to_clipboard


text = """
# Job Title
React Native/ReactJS Developer - ASAP Only
Details:
Responsibilities:
Develop and maintain robust, scalable mobile applications using
React Native
.
Build and enhance web applications with
React JS
.
Collaborate with designers, backend developers, and QA to deliver high-quality user experiences.
Optimize applications for maximum speed and scalability.
Write clean, reusable, and testable code.
Troubleshoot and debug issues across platforms and devices.
Stay updated with the latest trends and best practices in React and mobile/web development.
Qualifications:
Hands-on experience in
React Native/React JS
development
Strong experience in
API integrations
(RESTful/JSON)
Solid understanding of
database management
(e.g., SQLite, Realm, or remote DBs)
Familiar with
Agile/Scrum methodologies
Knowledge of software architecture patterns such as
MVVM
and
CLEAN architecture
Proficient in handling
JSON data structures
Experience in writing and maintaining
unit tests
Keywords:
- React Native
- React.js
"""

text = preprocess_text(text)
logger.success(text)
copy_to_clipboard(text)
