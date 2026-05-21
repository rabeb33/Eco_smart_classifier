"""
Module 4 — Pipeline NLP
Extrait du notebook 04_NLP.ipynb
"""

import argparse
import json
import os
import re
import warnings

import joblib
import mlflow
import mlflow.sklearn
import nltk
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

warnings.filterwarnings("ignore")

# Téléchargements NLTK silencieux
for pkg in ["stopwords", "punkt", "wordnet", "punkt_tab"]:
    nltk.download(pkg, quiet=True)

from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize

TARGET = "Categorie"

# Stopwords domaine métier (évite le data leakage)
STOP_DOMAINE = {
    "lot",
    "déchet",
    "collecté",
    "volume",
    "poids",
    "kg",
    "litre",
    "usine",
    "site",
    "matériau",
    "aspect",
    "organique",
    "aluminium",
}


# ══════════════════════════════════════════════════════════════════════
# PRÉTRAITEMENT TEXTUEL
# ══════════════════════════════════════════════════════════════════════


def get_stopwords() -> set:
    stop_fr = set(stopwords.words("french"))
    return stop_fr.union(STOP_DOMAINE)


def nettoyer_texte(texte: str) -> str:
    """Nettoyage de base : minuscules, suppression chiffres/ponctuation."""
    texte = texte.lower()
    texte = re.sub(r"[^a-zàâäéèêëîïôùûüç\s]", " ", texte)
    texte = re.sub(r"\s+", " ", texte).strip()
    return texte


def pretraiter_texte(texte: str, tous_stopwords: set, stemmer) -> str:
    """Pipeline complet : nettoyage → tokenisation → stopwords → stemming."""
    texte = nettoyer_texte(texte)
    tokens = word_tokenize(texte, language="french")
    tokens = [t for t in tokens if t not in tous_stopwords]
    tokens = [stemmer.stem(t) for t in tokens]
    return " ".join(tokens)


def preprocess_texts(df: pd.DataFrame) -> pd.DataFrame:
    tous_stopwords = get_stopwords()
    stemmer = SnowballStemmer("french")
    df = df.copy()
    df["texte_propre"] = df["Rapport_Collecte"].apply(
        lambda t: pretraiter_texte(str(t), tous_stopwords, stemmer)
    )
    print(f"[nlp] Prétraitement terminé sur {len(df)} lignes")
    return df


# ══════════════════════════════════════════════════════════════════════
# VECTORISATION + CLASSIFICATION
# ══════════════════════════════════════════════════════════════════════


def _log_run(experiment_name, vectorizer_name, classifier_name, acc, f1):
    """Helper MLflow pour logguer une expérience NLP."""
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(
        run_name=f"{vectorizer_name}_{classifier_name.replace(' ', '_')}"
    ):
        mlflow.log_param("vectorizer", vectorizer_name)
        mlflow.log_param("classifier", classifier_name)
        mlflow.log_metric("accuracy_test", acc)
        mlflow.log_metric("f1_test", f1)


def train_bow(X_train, X_test, y_train, y_test, scores: dict, experiment: str):
    """Bag of Words baseline."""
    bow = CountVectorizer()
    X_tr = bow.fit_transform(X_train)
    X_te = bow.transform(X_test)

    print("\n=== BoW ===")
    for nom, clf in {
        "Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "LinearSVC": LinearSVC(),
    }.items():
        clf.fit(X_tr, y_train)
        y_pred = clf.predict(X_te)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")
        scores[f"BoW+{nom}"] = acc
        _log_run(experiment, "BoW", nom, acc, f1)
        print(f"  {nom:25s} → Acc: {acc:.4f} | F1: {f1:.4f}")

    return bow


def train_tfidf(X_train, X_test, y_train, y_test, scores: dict, experiment: str):
    """TF-IDF avec unigrammes et bigrammes."""
    tfidf = TfidfVectorizer(ngram_range=(1, 2))
    X_tr = tfidf.fit_transform(X_train)
    X_te = tfidf.transform(X_test)

    best_clf = None
    best_acc = 0.0

    classifiers = {
        "Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "LinearSVC": LinearSVC(),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    }

    print("\n=== TF-IDF ===")
    for nom, clf in classifiers.items():
        clf.fit(X_tr, y_train)
        y_pred = clf.predict(X_te)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")
        scores[f"TF-IDF+{nom}"] = acc
        _log_run(experiment, "TF-IDF", nom, acc, f1)
        print(f"  {nom:25s} → Acc: {acc:.4f} | F1: {f1:.4f}")
        if acc > best_acc:
            best_acc = acc
            best_clf = clf

    return tfidf, best_clf


def train_lsa(X_train, X_test, y_train, y_test, scores: dict, experiment: str):
    """LSA = TF-IDF + TruncatedSVD (alternative sémantique à Word2Vec)."""
    lsa_pipe = Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
            ("svd", TruncatedSVD(n_components=100, random_state=42)),
        ]
    )
    X_tr = lsa_pipe.fit_transform(X_train)
    X_te = lsa_pipe.transform(X_test)

    print("\n=== LSA (alternative Word2Vec) ===")
    for nom, clf in {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "LinearSVC": LinearSVC(),
    }.items():
        clf.fit(X_tr, y_train)
        acc = accuracy_score(y_test, clf.predict(X_te))
        f1 = f1_score(y_test, clf.predict(X_te), average="weighted")
        scores[f"LSA+{nom}"] = acc
        _log_run(experiment, "LSA", nom, acc, f1)
        print(f"  {nom:25s} → Acc: {acc:.4f} | F1: {f1:.4f}")


def train_fasttext_like(
    X_train, X_test, y_train, y_test, scores: dict, experiment: str
):
    """FastText-like via char n-grammes (robuste aux fautes d'orthographe)."""
    ft_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2)
    clf_ft = LinearSVC()
    clf_ft.fit(ft_vec.fit_transform(X_train), y_train)
    acc = accuracy_score(y_test, clf_ft.predict(ft_vec.transform(X_test)))
    f1 = f1_score(y_test, clf_ft.predict(ft_vec.transform(X_test)), average="weighted")
    scores["FastText-like+LinearSVC"] = acc
    _log_run(experiment, "FastText-like", "LinearSVC", acc, f1)
    print(f"\n=== FastText-like ===")
    print(f"  LinearSVC                 → Acc: {acc:.4f} | F1: {f1:.4f}")
    print(f"  (robuste aux fautes d'orthographe via char n-grammes)")


# ══════════════════════════════════════════════════════════════════════
# POINT D'ENTRÉE PRINCIPAL
# ══════════════════════════════════════════════════════════════════════


def train_nlp(
    data_path: str, models_dir: str = "models", experiment: str = "EcoSmart_NLP"
):
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    # Chargement & prétraitement
    df = pd.read_csv(data_path)
    df_nlp = df[df[TARGET].notna()].copy()
    df_nlp = preprocess_texts(df_nlp)

    X_text = df_nlp["texte_propre"]
    y = df_nlp[TARGET]

    # Split 70/15/15
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_text, y, test_size=0.30, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )
    print(f"[split] Train:{len(X_train)} | Val:{len(X_val)} | Test:{len(X_test)}")

    scores = {}

    # Toutes les approches
    train_bow(X_train, X_test, y_train, y_test, scores, experiment)
    tfidf, best_nlp_clf = train_tfidf(
        X_train, X_test, y_train, y_test, scores, experiment
    )
    train_lsa(X_train, X_test, y_train, y_test, scores, experiment)
    train_fasttext_like(X_train, X_test, y_train, y_test, scores, experiment)

    # Résumé
    print("\n=== COMPARAISON FINALE NLP ===")
    for k, v in sorted(scores.items(), key=lambda x: -x[1]):
        print(f"  {k:45s} → {v:.4f}")

    best_combo = max(scores, key=scores.get)
    best_acc = scores[best_combo]
    print(f"\n[✓] Meilleure combinaison : {best_combo} ({best_acc:.4f})")

    # Sauvegarde modèles
    joblib.dump(tfidf, os.path.join(models_dir, "tfidf_vectorizer.pkl"))
    joblib.dump(best_nlp_clf, os.path.join(models_dir, "nlp_classifier.pkl"))
    print(f"[saved] tfidf_vectorizer.pkl + nlp_classifier.pkl → {models_dir}/")

    # Métriques JSON pour DVC
    with open("reports/metrics_nlp.json", "w") as f:
        json.dump(
            {
                "best_combination": best_combo,
                "best_accuracy": round(best_acc, 4),
                "all_scores": {k: round(v, 4) for k, v in scores.items()},
            },
            f,
            indent=2,
        )

    return tfidf, best_nlp_clf, scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/processed/dataset_clean.csv")
    parser.add_argument("--models", default="models")
    parser.add_argument("--experiment", default="EcoSmart_NLP")
    args = parser.parse_args()
    train_nlp(args.data, args.models, args.experiment)
