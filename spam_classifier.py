import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

print("Loading SMS dataset...")
data = pd.read_csv("data/sms.tsv", sep="\t", header=None, names=["label", "message"])

print(f"Total messages: {len(data)}")
print(data["label"].value_counts())

X = data["message"]
y = data["label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

vectorizer = CountVectorizer(stop_words="english")
X_train_counts = vectorizer.fit_transform(X_train)
X_test_counts = vectorizer.transform(X_test)

print("\nTraining Naive Bayes model...")
model = MultinomialNB()
model.fit(X_train_counts, y_train)

predictions = model.predict(X_test_counts)
accuracy = accuracy_score(y_test, predictions)

print(f"\nModel Accuracy: {accuracy * 100:.2f}%")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))
print("\nClassification Report:")
print(classification_report(y_test, predictions))

sample_messages = [
    "Congratulations! You've won a $1000 gift card. Click here to claim now!",
    "Hey, are we still meeting for lunch tomorrow?",
]
sample_counts = vectorizer.transform(sample_messages)
sample_predictions = model.predict(sample_counts)

print("\nSample Predictions:")
for message, prediction in zip(sample_messages, sample_predictions):
    print(f"Message: {message}")
    print(f"Prediction: {prediction}\n")
