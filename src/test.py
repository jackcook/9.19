from nltk.tokenize.toktok import ToktokTokenizer
from tqdm import tqdm

tok = ToktokTokenizer()

def test_batch(documents, log_prior, log_likelihood, classes, vocabulary):
    """
    Tests a batch of documents.

    Parameters:
        documents: Dataframe of 'documents' with a text column.
        log_prior: Dictionary mapping each class 'c' to its prior probability.
        log_likelihood: Dictionary mapping each class 'c' to its likelihood.
        classes: List of all possible classes the documents can have.
        vocabulary: The vocabulary set of the training documents.

    Returns:
        The list of predicted classes of each document.
    """

    predicted = []

    for i in tqdm(range(len(documents))):
        d = documents.iloc[i]["text"]
        c_hat = test_naive_bayes(d, log_prior, log_likelihood, classes, vocabulary)
        predicted.append(c_hat)

    return predicted

def test_naive_bayes(document, log_prior, log_likelihood, classes, vocabulary):
    """
    Tests a single document with a Naive Bayes model.

    Parameters:
        document: The sentence to classify.
        log_prior: Dictionary mapping each class 'c' to its prior probability.
        log_likelihood: Dictionary mapping each class 'c' to its likelihood.
        classes: List of all possible classes the documents can have.
        vocabulary: The vocabulary set of the training documents.

    Returns:
        The predicted class of the document.
    """

    # Calculate this document's probability of being each class
    document = tok.tokenize(document)
    sum = {}

    for c in classes:
        sum[c] = log_prior[c]

        for word in document:
            if word in vocabulary:
                sum[c] = sum[c] + log_likelihood[c][word]

    # Find the class with the highest probability
    best_class = None
    best_sum = -float("inf")

    for c in sum:
        if sum[c] > best_sum:
            best_class = c
            best_sum = sum[c]

    return best_class
