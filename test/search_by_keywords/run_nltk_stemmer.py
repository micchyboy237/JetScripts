from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

lemmatizer = WordNetLemmatizer()
text = "React.js and JavaScript are used in web development."
tokens = word_tokenize(text)

print([lemmatizer.lemmatize(token) for token in tokens])
