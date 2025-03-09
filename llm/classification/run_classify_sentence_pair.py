from jet.llm.classification import classify_sentence_pair
from jet.logger import logger
from jet.transformers.formatters import format_json
from jet.utils.sentence import split_sentences

if __name__ == '__main__':
    text = "Key Responsibilities:\nDevelop and maintain high-quality mobile applications using React Native.\nDeploy applications to the Apple App Store and Google Play Store.\nImplement efficient navigation and project structuring using Expo Router.\nDesign and build user interfaces with React Native Paper for consistent and accessible app design.\nStyle components using NativeWind and Tailwind CSS utilities.\nIntegrate and manage push notifications with services like Firebase Cloud Messaging and APNs.\nCollaborate with backend teams to integrate RESTful APIs into mobile apps.\nUtilize Zustand or similar libraries for effective state management.\nHandle form inputs and validations with React Hook Form and Zod.\nDebug and resolve application issues to ensure optimal performance.\nRequirements:\nExperience:\nMinimum of 3 years of professional experience in mobile application development using React Native.\nProven track record of building and maintaining mobile applications deployed on both iOS and Android platforms.\nTechnical Skills:\nStrong proficiency in React Native and its core principles.\nHands-on experience deploying apps to the Apple App Store and Google Play Store.\nProficiency with Expo Router for navigation and project structuring.\nExperience with React Native Paper for consistent app design.\nFamiliarity with NativeWind for styling components using Tailwind CSS utilities.\nKnowledge of push notification services such as Firebase Cloud Messaging and APNs.\nExperience working with native modules and integrating them with React Native.\nUnderstanding of RESTful APIs and integration with mobile applications.\nStrong debugging skills to troubleshoot and resolve issues effectively.\nProficiency in Zustand or similar state management libraries.\nExperience with React Hook Form for efficient form handling.\nKnowledge of Zod for schema validation.\nKnowledge of TypeScript.\n## Employer questions\nYour application will include the following questions:\n* Which of the following types of qualifications do you have?\n* What's your expected monthly basic salary?\n* How many years' experience do you have as a React Native Developer?\n### Company profile\n#### Cafisglobal Inc.\nInformation & Communication Technology11-50 employees\nCafisglobal Inc is a boutique ITBPO company servicing clients globally. We provide both voice support and software development services to our clients. We are small but growing organization expanding to new markets and is seeking experienced and dedicated workers to join our team.\nCafisglobal Inc is a boutique ITBPO company servicing clients globally. We provide both voice support and software development services to our clients. We are small but growing organization expanding to new markets and is seeking experienced and dedicated workers to join our team.\nMore about this company"

    sentences = split_sentences(text)

    # Get sentence pairs
    sentence_pairs = [
        {"sentence1": sentences[i],
            "sentence2": sentences[i + 1]}
        for i in range(len(sentences) - 1)
    ]

    acc_sentence = ""
    top_results = set()
    grouped_sentences = []
    for idx, sentence in enumerate(sentences):
        if idx == 0:
            acc_sentence += sentence + "\n"
            continue

        sentence_pair = {
            "sentence1": acc_sentence,
            "sentence2": sentence,
        }

        classification_results = classify_sentence_pair(sentence_pair)
        top_result = classification_results[0]

        if top_result['is_continuation']:
            acc_sentence += sentence + "\n"
        else:
            grouped_sentences.append(acc_sentence)
            acc_sentence = sentence + "\n"

    logger.debug(f"Grouped Results ({len(grouped_sentences)}):")
    logger.success(format_json(grouped_sentences))
