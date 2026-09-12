import os
import time

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, ConfusionMatrixDisplay
from sklearn.linear_model import LogisticRegression


os.makedirs("logistic_regression_cf", exist_ok=True)

train = fetch_20newsgroups(
    subset='train',
    categories=["sci.crypt", "sci.electronics", "sci.med"],
    remove=('headers', 'footers', 'quotes'))

test = fetch_20newsgroups(
    subset='test',
    categories=["sci.crypt", "sci.electronics", "sci.med"],
    remove=('headers', 'footers', 'quotes'))

x_train, y_train = train.data, train.target
x_test, y_test = test.data, test.target

experiments = [
    {"name": "База", "params": {}},
    {"name": "Отсечение очень редких (min_df=2)", "params": {"min_df": 2}},
    {"name": "Отсечение редких (min_df=3)", "params": {"min_df": 3}},
    {"name": "Отсечение очень частых (max_df=0.5)", "params": {"max_df": 0.5}},
    {"name": "Отсечение частых (max_df=0.4)", "params": {"max_df": 0.4}},
    {"name": "Отсечение частых (max_df=0.3)", "params": {"max_df": 0.3}},
    {"name": "Отсечение частых и очень редких (min_df=2 max_df=0.4)", "params": {"min_df": 2, "max_df": 0.4}},
    {"name": "Отсечение очень частых и очень редких (min_df=2 max_df=0.5)", "params": {"min_df": 2, "max_df": 0.5}},
]


results = []

for experiment in experiments:
    vectorizer = TfidfVectorizer(**experiment["params"]) # type: ignore
    start_time = time.time()
    x_train_vec = vectorizer.fit_transform(x_train)
    x_test_vec = vectorizer.transform(x_test)

    classifier = LogisticRegression()
    classifier.fit(x_train_vec, y_train)
    fit_time = time.time() - start_time

    predictions = classifier.predict(x_test_vec)
    macro_f1 = f1_score(y_test, predictions, average="macro")

    ConfusionMatrixDisplay.from_predictions(
        y_test,
        predictions,
        display_labels=test.target_names,
        cmap=plt.cm.Blues
    )

    results.append({
        "Эксперимент": experiment["name"],
        "Количество признаков": x_train_vec.shape[1],
        "Время обучения": round(fit_time, 4),
        "Macro F1": round(macro_f1, 4),
    })
    print(
        f"Эксперимент: {experiment['name']}\n"
        f"Количество признаков: {x_train_vec.shape[1]}\n"
        f"Время обучения: {round(fit_time, 4)}\n"
        f"Macro F1: {round(macro_f1, 4)}\n"
    )

    plt.savefig(os.path.join("logistic_regression_cf", f"{experiment['name']}.png".replace(" ", "_")))

pd.DataFrame(results).to_csv("logistic_regression_results.csv", index=False)
