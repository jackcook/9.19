import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from train import train_naive_bayes
from test import test_naive_bayes

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

# Randomly sample test dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
train_df = pd.concat([X_train, y_train], axis=1)
test_df = pd.concat([X_test, y_test], axis=1)

# Reset index values for test dataframe
test_df = test_df.reset_index()
test_df = test_df.drop("index", axis=1)

# Train model
classes = set(train_df.rating.values)
log_prior, log_likelihood, vocabulary = train_naive_bayes(train_df, classes)

# Test model
predicted = []

for i in tqdm(range(len(test_df))):
    c_hat = test_naive_bayes(test_df.iloc[i]["text"], log_prior, log_likelihood, classes, vocabulary)
    predicted.append(c_hat)

# Calculate model MSE
test_df["predicted"] = predicted
test_df["squared_error"] = (test_df["rating"] - test_df["predicted"]) * (test_df["rating"] - test_df["predicted"])
print("Mean squared error: %f" % (test_df["squared_error"].mean()))
