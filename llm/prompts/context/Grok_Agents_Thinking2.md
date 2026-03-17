## Prompt

Which of these postprocess utils should I use to prepare the funasr transcribed ja_text for english translation

## Thinking • 37s

Thinking about your request

Choosing postprocess utils

Use the funasr/utils postprocess functions to clean and normalize the Japanese transcription before translating to English.
Understanding postprocessors

rich_transcription_postprocess is already applied to get ja_text from the transcription output.
Preparing for translation

Strip emotion tags and emojis from ja_text to ensure clean Japanese text for accurate English translation.
Refining for translation

rich_transcription_postprocess may insert emojis for emotions, so consider applying sentence_postprocess afterward for plainer Japanese text.
sentence_postprocess won't remove rich tokens like <|HAPPY|>, so it may not fully clean the text for translation.
rich_transcription_postprocess adds emojis for emotions, so skip it and use sentence_postprocess directly on raw output for cleaner Japanese text.
For translation, apply sentence_postprocess to the raw output to remove special tokens without adding emojis.
Searched web
FunAudioLLM/SenseVoiceSmall funasr transcription output example "rich_transcription_postprocess"
9 results

Searched web
SenseVoiceSmall japanese transcription example output site:github.com
10 results

Searched web
"rich_transcription_postprocess" funasr SenseVoice
5 results

Search for SenseVoiceSmall output examples to see if raw text can be cleaned without rich_postprocess.
Confirming recommended postprocess

Always use rich_transcription_postprocess for SenseVoiceSmall as it's the official method, but strip emojis manually before English translation.
sentence_postprocess expects a list of words and won't handle SenseVoice's rich tokens, so avoid it on raw output.
rich_transcription_postprocess is the official and recommended postprocessor for SenseVoiceSmall's rich output, including Japanese transcription.
After rich_transcription_postprocess, strip emojis and event indicators to get pure spoken Japanese text for translation.
rich_transcription_postprocess is the correct choice for preparing SenseVoiceSmall's Japanese output for English translation.
