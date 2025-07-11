import os
from gliner import GLiNER

from jet.file.utils import save_file
from jet.logger import logger

model = GLiNER.from_pretrained("urchade/gliner_small-v2.1")

text = """
Cristiano Ronaldo dos Santos Aveiro (Portuguese pronunciation: [kɾiʃˈtjɐnu ʁɔˈnaldu]; born 5 February 1985) is a Portuguese professional footballer who plays as a forward for and captains both Saudi Pro League club Al Nassr and the Portugal national team. Widely regarded as one of the greatest players of all time, Ronaldo has won five Ballon d'Or awards,[note 3] a record three UEFA Men's Player of the Year Awards, and four European Golden Shoes, the most by a European player. He has won 33 trophies in his career, including seven league titles, five UEFA Champions Leagues, the UEFA European Championship and the UEFA Nations League. Ronaldo holds the records for most appearances (183), goals (140) and assists (42) in the Champions League, goals in the European Championship (14), international goals (128) and international appearances (205). He is one of the few players to have made over 1,200 professional career appearances, the most by an outfield player, and has scored over 850 official senior career goals for club and country, making him the top goalscorer of all time.
"""

labels = ["person", "award", "date", "competitions", "teams"]

entities = model.predict_entities(text, labels)


logger.newline()
logger.debug("Extracted Entities:")
for entity in entities:
    logger.newline()
    logger.log("Text:", entity['text'], colors=["WHITE", "INFO"])
    logger.log("Label:", entity['label'], colors=["WHITE", "SUCCESS"])
    logger.log("Score:", f"{entity['score']:.4f}", colors=[
               "WHITE", "SUCCESS"])
    # logger.log("Start:", f"{entity['char_start_index']}", colors=[
    #            "WHITE", "SUCCESS"])
    # logger.log("End:", f"{entity['char_end_index']}",
    #            colors=["WHITE", "SUCCESS"])
    logger.log("---")

output_dir = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
save_file(entities, f"{output_dir}/entities.json")
