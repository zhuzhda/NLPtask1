from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer


train = fetch_20newsgroups(
    subset="train",
    categories=["sci.crypt", "sci.electronics", "sci.med"],
    remove=("headers", "footers", "quotes")
)

vectorizer = CountVectorizer()
x_train = vectorizer.fit_transform(train.data)

word = input("Слово: ")
idx = vectorizer.vocabulary_[word]

for class_id, class_name in enumerate(train.target_names):
    mask = train.target == class_id

    df = x_train[mask, idx].getnnz()
    total = mask.sum()
    percentage = df / total * 100

    print(f"{class_name} DF = {df} / {total} ({percentage:.2f}%)")
