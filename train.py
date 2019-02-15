from nltk.tokenize.toktok import ToktokTokenizer
import numpy as np
from tqdm import tqdm

alpha = 0.001
tok = ToktokTokenizer()

def load_vocabulary(documents):
    """
    Generates the vocabulary set from a dataframe of documents.

    Parameters:
        documents: Dataframe of 'documents' with a text column.

    Returns:
        A set of all words that occur at least once throughout the documents.
    """

    vocabulary = set()

    for i in range(len(documents)):
        d_vocab = set(tok.tokenize(documents.iloc[i, :]["text"]))
        vocabulary.update(d_vocab)

    return vocabulary

def train_naive_bayes(documents, classes):
    """
    Trains a Naive Bayes text classification model.

    Parameters:
        documents: Dataframe of 'documents' with a text and rating column.
        classes: List of all possible classes the documents can have.

    Returns:
        log_prior: Dictionary mapping each class 'c' to its prior probability.
        log_likelihood: Dictionary mapping each class 'c' to its likelihood.
        vocabulary: The vocabulary set of these training documents.
    """

    # Load the vocabulary from training data
    vocabulary = load_vocabulary(documents)
    print("Loaded vocabulary, %d tokens" % len(vocabulary))

    log_prior = {}
    big_doc = {c: [] for c in classes}
    log_likelihood = {c: {} for c in classes}

    # Total number of training documents
    n_doc = len(documents)

    for c in tqdm(classes):
        # All of the training documents belonging to this class
        docs_c = documents[documents["rating"] == c]
        n_c = len(docs_c)

        log_prior[c] = np.log(n_c / n_doc)

        # Find the frequencies of all tokens in the training dataset
        counts = {word: 0 for word in vocabulary}

        for i, row in docs_c.iterrows():
            text = tok.tokenize(docs_c.loc[i, :]["text"])
            big_doc[c].extend(text)

            for word in text:
                counts[word] += 1

        # Calculate normalizing constant
        sum = 0

        for word in vocabulary:
            sum += counts[word] + alpha

        # Calculate likelihood probability of each word
        for word in vocabulary:
            freq = counts[word]
            log_likelihood[c][word] = np.log((freq + alpha) / sum)

    return log_prior, log_likelihood, vocabulary
