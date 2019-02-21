from nltk import FreqDist
from nltk.tokenize.toktok import ToktokTokenizer

tok = ToktokTokenizer()

# Read in and tokenize WSJ dataset
train = open("wsj_1994_train.txt").read()
train_tokens = tok.tokenize(train)

test = open("wsj_1994_test.txt").read()
test_tokens = tok.tokenize(test)
test_vocab_size = len(set(test_tokens))

vocabulary = set(train_tokens + test_tokens)
print("Loaded vocabulary, %d tokens" % len(vocabulary))

# Create frequency distribution of each word in training set
train_fdist = FreqDist(train_tokens)

# Map words to list of words that come after that word
next_words = {token: [] for token in set(train_tokens)}

for i, token in enumerate(train_tokens):
    if i < len(train_tokens) - 1:
        next_words[token].append(train_tokens[i + 1])

# Map words to frequency distributions of words that come after that word
next_words_fdists = {token: FreqDist(next_words[token]) for token in next_words}

# Calculate model perplexity
print("Evaluating model...")

perplexity = 1
alpha = 0.0012

for i, token in enumerate(test_tokens):
    if i == 0: continue

    # Find the frequency of the last word in the training set
    last_word = test_tokens[i - 1]
    last_word_frequency = train_fdist[last_word]

    # Find the frequency of the (last_word, current_word) bigram from the
    # training set
    try:
        current_word_frequency = next_words_fdists[last_word][token]
    except:
        current_word_frequency = 0

    # Find the probability that the current word comes after the last one, with
    # add-alpha smoothing to account for zero probabilities
    current_word_probability = (current_word_frequency + alpha) / (last_word_frequency + len(vocabulary) * alpha)
    perplexity *= pow(current_word_probability, -1 / len(test_tokens))

print("Perplexity: " + str(perplexity))
