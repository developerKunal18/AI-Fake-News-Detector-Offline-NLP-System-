from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

news = [
    "Scientists discover new cure for cancer",
    "Government announces new education policy",
    "Click here to win free money now",
    "Aliens landed in New York last night",
    "Elections results announced officially",
    "You won lottery claim prize immediately"
]

labels = [0, 0, 1, 1, 0, 1]  # 1 = Fake, 0 = Real

vectorizer = TfidfVectorizer(stop_words="english")
X = vectorizer.fit_transform(news)

model = LogisticRegression()
model.fit(X, labels)

print("📰 AI Fake News Detector\n")

text = input("Paste news text: ")
vec = vectorizer.transform([text])
prediction = model.predict(vec)[0]

if prediction == 1:
    print("\n🚨 FAKE news detected")
else:
    print("\n✅ This looks like REAL news")
