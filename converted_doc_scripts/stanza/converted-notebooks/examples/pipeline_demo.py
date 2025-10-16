"""
A basic demo of the Stanza neural pipeline.
"""

import sys
import argparse
import os

from jet.file.utils import save_file
import stanza
from stanza.resources.common import DEFAULT_MODEL_DIR

from jet.logger import logger
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger.basicConfig(filename=log_file)
logger.info(f"Logs: {log_file}")


if __name__ == '__main__':
    # get arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--models_dir', help='location of models files | default: ~/stanza_resources',
                        default=DEFAULT_MODEL_DIR)
    parser.add_argument('-l', '--lang', help='Demo language',
                        default="en")
    parser.add_argument('-c', '--cpu', action='store_true', help='Use cpu as the device.')
    args = parser.parse_args()

    example_sentences = {"en": "Barack Obama was born in Hawaii.  He was elected president in 2008.",
            "zh": "中国文化经历上千年的历史演变，是各区域、各民族古代文化长期相互交流、借鉴、融合的结果。",
            "fr": "Van Gogh grandit au sein d'une famille de l'ancienne bourgeoisie. Il tente d'abord de faire carrière comme marchand d'art chez Goupil & C.",
            "vi": "Trận Trân Châu Cảng (hay Chiến dịch Hawaii theo cách gọi của Bộ Tổng tư lệnh Đế quốc Nhật Bản) là một đòn tấn công quân sự bất ngờ được Hải quân Nhật Bản thực hiện nhằm vào căn cứ hải quân của Hoa Kỳ tại Trân Châu Cảng thuộc tiểu bang Hawaii vào sáng Chủ Nhật, ngày 7 tháng 12 năm 1941, dẫn đến việc Hoa Kỳ sau đó quyết định tham gia vào hoạt động quân sự trong Chiến tranh thế giới thứ hai."}

    if args.lang not in example_sentences:
        logger.debug(f'Sorry, but we don\'t have a demo sentence for "{args.lang}" for the moment. Try one of these languages: {list(example_sentences.keys())}')
        sys.exit(1)

    # download the models
    # stanza.download(args.lang, dir=args.models_dir)
    # set up a pipeline
    logger.debug('---')
    logger.debug('Building pipeline...')
    pipeline = stanza.Pipeline(lang=args.lang, dir=args.models_dir, use_gpu=(not args.cpu))
    # process the document
    doc = pipeline(example_sentences[args.lang])
    # access nlp annotations
    logger.debug('')
    logger.debug('Input: {}'.format(example_sentences[args.lang]))
    logger.debug("The tokenizer split the input into {} sentences.".format(len(doc.sentences)))

    logger.debug('')
    logger.debug('---')
    logger.debug('tokens of first sentence: ')
    tokens_str = doc.sentences[0].tokens_string()
    save_file(tokens_str, f"{OUTPUT_DIR}/tokens.txt")

    logger.debug('')
    logger.debug('---')
    logger.debug('words of first sentence: ')
    words_str = doc.sentences[0].words_string()
    save_file(words_str, f"{OUTPUT_DIR}/words.txt")
    logger.debug('')

    logger.debug('')
    logger.debug('---')
    logger.debug('dependency parse of first sentence: ')
    deps_str = doc.sentences[0].dependencies_string()
    save_file(deps_str, f"{OUTPUT_DIR}/dependencies.txt")

    save_file(doc.sentences[0].to_dict(), f"{OUTPUT_DIR}/sentence_1.json")
