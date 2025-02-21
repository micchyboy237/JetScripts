import Stemmer

# Create a stemmer for the English language
# stemmer = Stemmer.Stemmer('english')
stemmer = Stemmer.Stemmer('en')

# Example: Stemming a single word
word = "running"
stemmed_word = stemmer.stemWord(word)
print(stemmed_word)  # Output: run

# Example: Stemming a list of words
words = ["running", "jumps", "easily", "faster"]
stemmed_words = stemmer.stemWords(words)
print(stemmed_words)  # Output: ['run', 'jump', 'easili', 'faster']
