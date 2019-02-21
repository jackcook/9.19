import pandas as pd

from train import train_naive_bayes
from test import test_batch

# Load data from csv
df = pd.read_csv("./data/train.csv")

# Lowercase strings and remove ten reviews with NaN text
df["text"] = df["review/text"].str.lower()
df = df[df["text"].apply(lambda x: isinstance(x, str))]

# Remove one review with 0 rating
df = df[df["review/overall"] > 0]

# Prepare test dataframe
test_df = pd.read_csv("./data/test.csv")
test_df["text"] = test_df["review/text"]
test_df["idx"] = test_df["index"]
test_df = test_df[["idx", "text"]]
test_df = test_df.reset_index()

for category in ["appearance", "aroma", "overall", "palate", "taste"]:
    # Isolate the rating we're looking at for this category
    print("Training for %s" % category)
    df["rating"] = df["review/" + category]
    train_df = df[["text", "rating"]]

    # Train model
    classes = sorted(set(df.rating.values))
    log_prior, log_likelihood, vocabulary = train_naive_bayes(train_df, classes)

    # Test model
    print("Testing for %s" % category)
    test_df["review/" + category] = test_batch(test_df, log_prior, log_likelihood, classes, vocabulary)

# Save submission.csv
test_df["index"] = test_df["idx"]
test_df = test_df.drop(["text", "idx"], axis=1)
test_df.to_csv("submission.csv", index=False)
