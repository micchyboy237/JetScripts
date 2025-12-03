from jet.translators.google_translator import translate_text

ja_text = "朝が10時ぐらいに、10時過ぎに食べたのでまだ食べてません。"
en_text = translate_text(ja_text)
print(f"ja -> en translation: {en_text}")