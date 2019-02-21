# naive-bayes

A simple naive Bayes text classifier. Currently set up to work with the [Kaggle beer ratings](http://kaggle.com/c/beer-ratings) dataset, but can be easily modified to work with other simple text-based datasets.

## Setup

### Download dataset

Start by downloading the dataset from the [competition page](https://www.kaggle.com/c/beer-ratings/data). Rename the downloaded folder to `data` and move it into the same directory as this `README.md` file.

### Install dependencies

Make sure you're using Python 3, and install all of the necessary dependencies for this project.

```bash
pip install -r requirements.txt
```

## Usage

### General model

By default, `main.py` will train a Naive Bayes classifier with 10-fold cross validation and print the mean squared error. In order to try this out, run the file in your terminal.

```bash
python src/main.py
```

### Kaggle submission

I created another file, `submission.py`, that creates a valid submission file for the Kaggle competition. This model should get you a score of 0.62784, which isn't amazing, but it's not too bad either.

```bash
python src/submission.py
```

Once the script finishes, it will create a `submission.csv` file that can be uploaded directly to Kaggle.

## License

`naive-bayes` is available under the MIT license. See the [LICENSE](https://github.com/jackcook/naive-bayes/blob/master/LICENSE) file for details.
