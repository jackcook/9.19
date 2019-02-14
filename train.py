import numpy as np
from tokenize_words import tokenize
from tqdm import tqdm

def load_vocabulary(documents):
    vocabulary = set()

    for i in range(len(documents)):
        d_vocab = set(tokenize(documents.iloc[i, :]["text"]))
        vocabulary.update(d_vocab)

    return vocabulary

def train_naive_bayes(documents, classes):
    vocabulary = load_vocabulary(documents)
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
                text = tokenize(documents.iloc[i, :]["text"])
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
