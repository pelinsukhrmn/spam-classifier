# Spam Classifier

A text classifier that flags SMS messages as spam or ham (not spam) using Naive Bayes.

## How it works

- Loads the SMS Spam Collection dataset (`data/sms.tsv`, 5,574 labeled real SMS messages)
- Splits the data into train/test sets (80/20)
- Vectorizes the messages with `CountVectorizer` (bag-of-words, English stop words removed)
- Trains a Multinomial Naive Bayes model
- Prints accuracy, a confusion matrix, and a classification report on the test set
- Runs two example messages through the model (an obvious spam prize message and a normal one) and prints the predictions

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
python spam_classifier.py
```
