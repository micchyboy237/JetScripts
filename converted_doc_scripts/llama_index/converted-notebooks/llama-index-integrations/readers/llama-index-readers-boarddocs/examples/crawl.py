from bs4 import BeautifulSoup
from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import html2text
import json
import os
import requests
import shutil
import xml.etree.ElementTree as ET


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

model_name = "sentence-transformers/all-MiniLM-L6-v2"
Settings.embed_model = HuggingFaceEmbedding(
    model_name=model_name,
    cache_folder=MODELS_CACHE_DIR,
)


"""
# BoardDocs Crawl

Let's figure out how to crawl BoardDocs!

We'll try the Redwood City School District site using BeautifulSoup.

https://go.boarddocs.com/ca/redwood/Board.nsf/Public
"""
logger.info("# BoardDocs Crawl")

site = "ca/redwood"
committeeID = "A4EP6J588C05"


baseURL = "https://go.boarddocs.com/" + site + "/Board.nsf"
publicURL = baseURL + "/Public"
meetingsListURL = baseURL + "/BD-GetMeetingsList?open"

headers = {
    "accept": "application/json, text/javascript, */*; q=0.01",
    "accept-language": "en-US,en;q=0.9",
    "content-type": "application/x-www-form-urlencoded; charset=UTF-8",
    "sec-ch-ua": '"Google Chrome";v="113", "Chromium";v="113", "Not-A.Brand";v="24"',
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": '"macOS"',
    "sec-fetch-dest": "empty",
    "sec-fetch-mode": "cors",
    "sec-fetch-site": "same-origin",
    "x-requested-with": "XMLHttpRequest",
}

data = "current_committee_id=" + committeeID

response = requests.post(meetingsListURL, headers=headers, data=data)

logger.debug("Status returned by meetings list request:", response.status_code)


meetingsData = json.loads(response.text)

meetings = [
    {
        "meetingID": meeting.get("unique", None),
        "date": meeting.get("numberdate", None),
        "unid": meeting.get("unid", None),
    }
    for meeting in meetingsData
]

logger.debug(str(len(meetings)) + " meetings found")


xmlMeetingListURL = baseURL + "/XML-ActiveMeetings"
xmlMeetingListData = requests.get(xmlMeetingListURL)
xmlMeetingList = ET.fromstring(xmlMeetingListData)

detailedMeetingAgendaURL = baseURL + "/PRINT-AgendaDetailed"

meetingID = "CPSNV9612DF1"

data = "id=" + meetingID + "&" + "current_committee_id=" + committeeID

response = requests.post(detailedMeetingAgendaURL, headers=headers, data=data)

logger.debug("Status returned by detailed agenda fetch request:", response.status_code)


soup = BeautifulSoup(response.content, "html.parser")
agendaDate = soup.find("div", {"class": "print-meeting-date"}).string
agendaTitle = soup.find("div", {"class": "print-meeting-name"}).string
agendaFiles = [
    fd.a.get("href") for fd in soup.find_all("div", {"class": "public-file"})
]
agendaData = html2text.html2text(response.text)
logger.debug("Agenda Title:", agendaTitle)
logger.debug("Agenda Date:", agendaDate)
logger.debug("Number of Files:", len(agendaFiles))

logger.debug(agendaFiles)

for meeting in meetings:
    logger.debug(meeting["meetingID"])

logger.info("\n\n[DONE]", bright=True)