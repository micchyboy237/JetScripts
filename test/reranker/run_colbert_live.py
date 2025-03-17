import colbert_live
from colbert_live.db.astra import AstraCQL
# or
from colbert_live.db.sqlite import Sqlite3DB


class MyDB(AstraCQL):
    ...


db = MyDB()

model = colbert_live.models.ColbertModel()
# or
model = colbert_live.models.ColpaliModel()

colbert = ColbertLive(db, model)
# Call search:
colbert.search(query_str, top_k)
