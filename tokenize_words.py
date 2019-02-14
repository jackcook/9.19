from nltk.corpus import stopwords
from nltk.tokenize.toktok import ToktokTokenizer
import string

tok = ToktokTokenizer()
# stop_words = stopwords.words("english")# + list(string.punctuation)
stop_words = []

negative_words = ["didn't", "didnt", "not", "no", "never"]
punctuation = list(string.punctuation)

def tokenize(text):
    return [w for w in tok.tokenize(text) if not w in stop_words]

    tokens = tok.tokenize(text)
    negative = False

    for i in range(len(tokens)):
        if tokens[i] in negative_words:
            negative = True
            continue
        elif tokens[i] in punctuation:
            negative = False
            continue

        if negative:
            tokens[i] = tokens[i] + "_NOT"

    return [w for w in tokens if not w in stop_words]
