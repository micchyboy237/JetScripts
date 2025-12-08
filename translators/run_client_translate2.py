from jet.translators.client_translate import print_translations, translate_text


if __name__ == "__main__":
    # Single
    result = translate_text("今日はいい天気ですね。一緒に散歩しませんか？")
    print_translations(result)

    # Batch
    texts = [
        "昨日、友達と一緒に映画を見に行きました。",
        "日本は美しい国ですね！",
        "今日の天気はとても良いです。",
    ]
    results = translate_text(texts, device="cuda")
    print_translations(results)