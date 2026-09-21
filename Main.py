import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

df = pd.read_csv("C:/Users/krish/Downloads/cyberbullying_uncleaned_dataset.csv")

print(df.head())
print(df.info())

df['likes'].fillna(df['likes'].median(), inplace=True)
df['language'].fillna('unknown', inplace=True)
df['label'].fillna('non-bully', inplace=True)

df['comment_text'] = df['comment_text'].str.lower()

df.drop_duplicates(inplace=True)

print("\nSummary Statistics:\n", df.describe())
print("\nLabel Distribution:\n", df['label'].value_counts())

plt.figure()
sns.countplot(x='label', data=df)
plt.title("Cyberbullying vs Non-Bullying")
plt.show()

plt.figure()
sns.countplot(x='platform', data=df)
plt.title("Platform Distribution")
plt.show()

plt.figure()
sns.histplot(df['likes'], bins=30)
plt.title("Likes Distribution")
plt.show()

plt.figure()
sns.histplot(df['replies'], bins=30)
plt.title("Replies Distribution")
plt.show()

plt.figure()
sns.boxplot(x='label', y='likes', data=df)
plt.title("Likes vs Label")
plt.show()

plt.figure()
sns.boxplot(x='label', y='replies', data=df)
plt.title("Replies vs Label")
plt.show()

plt.figure()
sns.histplot(df['report_count'], bins=20)
plt.title("Report Count Distribution")
plt.show()

plt.figure()
corr = df[['likes', 'replies', 'report_count']].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

sns.pairplot(df[['likes', 'replies', 'report_count']])
plt.show()

X = df['comment_text']
y = df['label']

vectorizer = TfidfVectorizer(stop_words='english')
X_vectorized = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized, y, test_size=0.2, random_state=42
)

model = MultinomialNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

plt.figure()
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

sample = ["you are an idiot"]
sample_vec = vectorizer.transform(sample)
prediction = model.predict(sample_vec)

print("\nCustom Input Prediction:", prediction[0])
