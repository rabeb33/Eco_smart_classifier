"""
Module 6 — API REST FastAPI
Endpoint /predict pour la classification de déchets
"""

import os
import re
from typing import Optional

import joblib
import nltk
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# NLTK downloads
for pkg in ["stopwords", "punkt", "wordnet", "punkt_tab"]:
    nltk.download(pkg, quiet=True)

from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize

# ── Chemins modèles ────────────────────────────────────────────────────────────
MODELS_DIR = os.getenv("MODELS_DIR", "models")

# Seuil de confiance NLP : en dessous, on fait confiance au numérique
NLP_CONFIDENCE_THRESHOLD = 0.70


def load_model(name: str):
    path = os.path.join(MODELS_DIR, name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Modèle introuvable : {path}")
    return joblib.load(path)


# ── App FastAPI ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Eco-Smart Classifier API",
    description="Classification de déchets et estimation de valeur",
    version="1.0.0",
)

# Chargement paresseux des modèles
_models = {}


def get_models():
    if not _models:
        _models["classifier"] = load_model("classifier_best.pkl")
        _models["regressor"] = load_model("regressor_best.pkl")
        _models["scaler"] = load_model("scaler.pkl")
        _models["le_source"] = load_model("le_source.pkl")
        _models["le_categorie"] = load_model("le_categorie.pkl")
        _models["tfidf"] = load_model("tfidf_vectorizer.pkl")
        _models["nlp_classifier"] = load_model("nlp_classifier.pkl")
    return _models


# ── Schémas Pydantic ──────────────────────────────────────────────────────────
class PredictionRequest(BaseModel):
    poids: float = Field(..., gt=0, description="Poids en kg")
    volume: float = Field(..., gt=0, description="Volume en litres")
    conductivite: float = Field(..., description="Conductivité")
    opacite: float = Field(..., ge=0, le=1, description="Opacité [0-1]")
    rigidite: float = Field(..., description="Rigidité")
    source: str = Field(..., description="Source du déchet")
    rapport: Optional[str] = Field(None, description="Rapport collecte (NLP)")


class PredictionResponse(BaseModel):
    categorie: str
    prix_estime: float
    confidence: Optional[float] = None
    nlp_categorie: Optional[str] = None
    nlp_confidence: Optional[float] = None
    fusion_source: Optional[str] = None  # "numerique" ou "nlp"


class HealthResponse(BaseModel):
    status: str
    models_loaded: bool


# ── Preprocessing NLP ─────────────────────────────────────────────────────────
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
    "papier",
    "plastique",
    "metal",
    "métal",
    "verre",
    "organique",
    "carton",
    "aluminium",
    "ferreux",
    "ferraille",
}

_stop_fr = set(stopwords.words("french")).union(STOP_DOMAINE)
_stemmer = SnowballStemmer("french")


def preprocess_text(texte: str) -> str:
    texte = texte.lower()
    texte = re.sub(r"[^a-zàâäéèêëîïôùûüç\s]", " ", texte)
    texte = re.sub(r"\s+", " ", texte).strip()
    tokens = word_tokenize(texte, language="french")
    tokens = [_stemmer.stem(t) for t in tokens if t not in _stop_fr]
    return " ".join(tokens)


# ── Endpoints ──────────────────────────────────────────────────────────────────
@app.get("/", response_model=dict)
def root():
    return {"message": "Eco-Smart Classifier API", "docs": "/docs"}


@app.get("/health", response_model=HealthResponse)
def health():
    try:
        get_models()
        loaded = True
    except Exception:
        loaded = False
    return {"status": "ok" if loaded else "degraded", "models_loaded": loaded}


@app.post("/predict", response_model=PredictionResponse)
def predict(req: PredictionRequest):
    try:
        m = get_models()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))

    # ── 1. Encoder Source ──────────────────────────────────────────────────────
    try:
        source_enc = m["le_source"].transform([req.source])[0]
    except ValueError:
        source_enc = 0  # classe inconnue → 0

    # ── 2. Features numériques ─────────────────────────────────────────────────
    import pandas as pd
    raw = pd.DataFrame([[req.poids, req.volume, req.conductivite, req.opacite, req.rigidite]],
                   columns=["Poids", "Volume", "Conductivite", "Opacite", "Rigidite"])
    scaled = m["scaler"].transform(raw)
    features = np.append(scaled[0], source_enc).reshape(1, -1)

    # ── 3. Classification numérique ────────────────────────────────────────────
    categorie_num = m["classifier"].predict(features)[0]

    confidence_num = None
    if hasattr(m["classifier"], "predict_proba"):
        proba = m["classifier"].predict_proba(features)[0]
        confidence_num = float(np.max(proba))

    # ── 4. Régression (prix estimé) ────────────────────────────────────────────
    prix = float(m["regressor"].predict(features)[0])

    # ── 5. NLP + logique de fusion ─────────────────────────────────────────────
    nlp_categorie = None
    nlp_confidence = None
    fusion_source = "numerique"
    categorie_finale = str(categorie_num)

    if req.rapport:
        texte_clean = preprocess_text(req.rapport)
        vec = m["tfidf"].transform([texte_clean])
        nlp_categorie = m["nlp_classifier"].predict(vec)[0]

        # Confiance NLP si le classifieur le supporte
        if hasattr(m["nlp_classifier"], "predict_proba"):
            proba_nlp = m["nlp_classifier"].predict_proba(vec)[0]
            nlp_confidence = float(np.max(proba_nlp))
        elif hasattr(m["nlp_classifier"], "decision_function"):
            # LinearSVC : score de décision → confiance approchée via softmax
            scores = m["nlp_classifier"].decision_function(vec)[0]
            exp_scores = np.exp(scores - np.max(scores))
            nlp_confidence = float(np.max(exp_scores / exp_scores.sum()))

        # Fusion : NLP prioritaire seulement si confiance suffisante
        if nlp_confidence is not None and nlp_confidence >= NLP_CONFIDENCE_THRESHOLD:
            categorie_finale = str(nlp_categorie)
            fusion_source = "nlp"
        else:
            # Confiance NLP trop faible → on garde le numérique
            nlp_categorie = str(nlp_categorie)  # on l'affiche quand même
            fusion_source = "numerique (nlp confiance insuffisante)"

    return PredictionResponse(
        categorie=categorie_finale,
        prix_estime=round(prix, 2),
        confidence=round(confidence_num, 4) if confidence_num is not None else None,
        nlp_categorie=str(nlp_categorie) if nlp_categorie else None,
        nlp_confidence=round(nlp_confidence, 4) if nlp_confidence is not None else None,
        fusion_source=fusion_source,
    )


@app.get("/categories")
def get_categories():
    try:
        m = get_models()
        return {"categories": list(m["le_categorie"].classes_)}
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))