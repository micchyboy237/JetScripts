from jet.logger import logger
from stanza.utils.visualization.ner_visualization import visualize_strings
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger.basicConfig(filename=log_file)
logger.info(f"Logs: {log_file}")


en_strings = ['''Samuel Jackson, a Christian man from Utah, went to the JFK Airport for a flight to New York.
                 He was thinking of attending the US Open, his favorite tennis tournament besides Wimbledon.
                 That would be a dream trip, certainly not possible since it is $5000 attendance and 5000 miles away.
                 On the way there, he watched the Super Bowl for 2 hours and read War and Piece by Tolstoy for 1 hour.
                 In New York, he crossed the Brooklyn Bridge and listened to the 5th symphony of Beethoven as well as
                 "All I want for Christmas is You" by Mariah Carey.''',
              "Barack Obama was born in Hawaii. He was elected President of the United States in 2008"]

visualize_strings(en_strings, "en")


zh_strings = ['''来自犹他州的基督徒塞缪尔杰克逊前往肯尼迪机场搭乘航班飞往纽约。
                 他正在考虑参加美国公开赛，这是除了温布尔登之外他最喜欢的网球赛事。
                 那将是一次梦想之旅，当然不可能，因为它的出勤费为 5000 美元，距离 5000 英里。
                 在去的路上，他看了 2 个小时的超级碗比赛，看了 1 个小时的托尔斯泰的《战争与碎片》。
                 在纽约，他穿过布鲁克林大桥，聆听了贝多芬的第五交响曲以及 玛丽亚凯莉的“圣诞节我想要的就是你”。''',
              "我觉得罗家费德勒住在加州, 在美国里面。"]
visualize_strings(zh_strings, "zh", colors={"PERSON": "yellow", "DATE": "red", "GPE": "blue"})
visualize_strings(zh_strings, "zh", select=['PERSON', 'DATE'])


ar_strings = [".أعيش في سان فرانسيسكو ، كاليفورنيا. اسمي أليكس وأنا ألتحق بجامعة ستانفورد. أنا أدرس علوم الكمبيوتر وأستاذي هو كريس مانينغ"
             , "اسمي أليكس ، أنا من الولايات المتحدة.",
               '''صامويل جاكسون ، رجل مسيحي من ولاية يوتا ، ذهب إلى مطار جون كنيدي في رحلة إلى نيويورك. كان يفكر في حضور بطولة الولايات المتحدة المفتوحة للتنس ، بطولة التنس المفضلة لديه إلى جانب بطولة ويمبلدون. ستكون هذه رحلة الأحلام ، وبالتأكيد ليست ممكنة لأنها تبلغ 5000 دولار للحضور و 5000 ميل. في الطريق إلى هناك ، شاهد Super Bowl لمدة ساعتين وقرأ War and Piece by Tolstoy لمدة ساعة واحدة. في نيويورك ، عبر جسر بروكلين واستمع إلى السيمفونية الخامسة لبيتهوفن وكذلك "كل ما أريده في عيد الميلاد هو أنت" لماريا كاري.''']

visualize_strings(ar_strings, "ar", colors={"PER": "pink", "LOC": "linear-gradient(90deg, #aa9cfc, #fc9ce7)", "ORG": "yellow"})

logger.info("\n\n[DONE]", bright=True)