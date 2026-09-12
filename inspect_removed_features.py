from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer

train = fetch_20newsgroups(
    subset='train',
    categories=["sci.crypt", "sci.electronics", "sci.med"],
    remove=('headers', 'footers', 'quotes'))

x_train = train.data

full_vectorizer = TfidfVectorizer()
full_vectorizer.fit(x_train)
full_vocab = full_vectorizer.get_feature_names_out()

vectorizer_with_params = TfidfVectorizer(max_df=0.4) # min_df=2
vectorizer_with_params.fit(x_train)
filtered_vocab = vectorizer_with_params.get_feature_names_out()

removed_features = sorted(set(full_vocab) - set(filtered_vocab))
print(f"Удалено признаков: {len(removed_features)}")
print(f"Удалённые признаки: {removed_features}")
