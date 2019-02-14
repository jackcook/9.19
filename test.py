from tokenize_words import tokenize

def test_naive_bayes(document, log_prior, log_likelihood, classes, vocabulary):
    sum = {}
    document = tokenize(document)

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
