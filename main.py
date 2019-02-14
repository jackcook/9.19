from nltk.corpus import stopwords
from nltk.tokenize.toktok import ToktokTokenizer
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import string
from tqdm import tqdm

tok = ToktokTokenizer()
stop_words = stopwords.words("english") + list(string.punctuation)

def train_naive_bayes(documents, classes):
    vocabulary = set()

    for i in range(len(documents)):
        d_vocab = set(tok.tokenize(documents.iloc[i, :]["text"]))
        vocabulary.update(d_vocab)

    for word in stop_words:
        try:
            vocabulary.remove(word)
        except:
            pass

    print("Loaded vocabulary, %d tokens" % len(vocabulary))

    log_prior = {}
    big_doc = {}
    log_likelihood = {}

    for c in classes:
        n_doc = len(documents)
        n_c = len(documents[documents["rating"] == c])

        log_prior[c] = np.log(n_c / n_doc)
        big_doc[c] = []

        counts = {}

        for i in tqdm(range(len(documents))):
            rating = documents.iloc[i, :]["rating"]

            if rating == c:
                text = tok.tokenize(documents.iloc[i, :]["text"])
                text = [w for w in text if not w in stop_words]

                big_doc[c].extend(text)

                for word in text:
                    if word in counts:
                        counts[word] += 1
                    else:
                        counts[word] = 1

        denom = 0

        for word in vocabulary:
            if word in counts:
                denom += counts[word] + 1
            else:
                counts[word] = 0

        log_likelihood[c] = {}

        for word in vocabulary:
            freq = counts[word]
            log_likelihood[c][word] = np.log((freq + 1) / denom)

        print("Done with class %f" % c)

    return log_prior, log_likelihood, vocabulary

# Load data from csv
train_df = pd.read_csv("./data/train.csv")

# Remove ten reviews with NaN text
train_df["text"] = train_df["review/text"].str.lower()
train_df = train_df[train_df["text"].apply(lambda x: isinstance(x, str))]

# Remove one review with 0 rating
train_df["rating"] = train_df["review/overall"]
train_df = train_df[train_df["rating"] > 0]

# Drop all other columns
train_df = train_df[["text", "rating"]]
X = train_df["text"]
y = train_df["rating"]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.1)
train_df = pd.concat([X_train, y_train], axis=1)
test_df = pd.concat([X_test, y_test], axis=1)

log_prior, log_likelihood, vocabulary = train_naive_bayes(train_df, [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5])

def test_naive_bayes(document, log_prior, log_likelihood, classes, vocabulary):
    sum = {}
    document = tok.tokenize(document)
    document = [w for w in document if not w in stop_words]

    for c in classes:
        sum[c] = log_prior[c]
        for word in document:
            if word in vocabulary:
                sum[c] = sum[c] + log_likelihood[c][word]

    best_class = None
    best_sum = -float("inf")

    for c in sum:
        if sum[c] > best_sum:
            best_class = c
            best_sum = sum[c]

    return best_class

# test_df = pd.read_csv("./data/test.csv")
# test_df["text"] = test_df["review/text"]
# test_df = test_df[["text"]]

def test(i):
    return test_naive_bayes(test_df.iloc[i]["text"], log_prior, log_likelihood, [1,1.5,2,2.5,3,3.5,4,4.5,5], vocabulary)

predicted = []
test_df = test_df.reset_index()
test_df = test_df.drop("index", axis=1)

for i in tqdm(range(len(test_df))):
    predicted.append(test(i))

test_df["predicted"] = predicted
test_df["squared_error"] = (test_df["rating"] - test_df["predicted"]) * (test_df["rating"] - test_df["predicted"])
print("Mean squared error: %f" % (test_df["squared_error"].mean()))
