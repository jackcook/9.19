import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from train import train_naive_bayes
from test import test_batch

errors = []
k = 10

# Load data from csv
df = pd.read_csv("./data/train.csv")

# Lowercase strings and remove ten reviews with NaN text
df["text"] = df["review/text"].str.lower()
df = df[df["text"].apply(lambda x: isinstance(x, str))]

# Remove one review with 0 rating
df["rating"] = df["review/overall"]
df = df[df["rating"] > 0]

# Drop all other columns
df = df[["text", "rating"]]

# Create KFold model
kf = KFold(n_splits=k, shuffle=True, random_state=42)

for i, (train_idx, test_idx) in enumerate(kf.split(df)):
	# Use indices to create train and test dataframes
	print("Training fold %d" % i)
	train_df = df.iloc[train_idx]
	test_df = df.iloc[test_idx]

	# Reset index values for test dataframe
	test_df = test_df.reset_index()
	test_df = test_df.drop("index", axis=1)

	# Train model
	classes = sorted(set(df.rating.values))
	log_prior, log_likelihood, vocabulary = train_naive_bayes(train_df, classes)

	# Test model and calculate MSE
	print("Testing fold %d" % i)
	test_df["predicted"] = test_batch(test_df, log_prior, log_likelihood, classes, vocabulary)
	test_df["squared_error"] = (test_df["rating"] - test_df["predicted"]) * (test_df["rating"] - test_df["predicted"])
	mse = test_df["squared_error"].mean()

	errors.append(mse)
	print("Mean squared error: %f" % mse)

print("Final MSE: %f" % np.mean(errors))
