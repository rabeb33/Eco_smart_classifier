"""
Module 2 — Modélisation supervisée + MLflow tracking
Extrait du notebook 02_ML_Supervise.ipynb
"""

import argparse
import json
import os
import warnings

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

FEATURES = ["Poids", "Volume", "Conductivite", "Opacite", "Rigidite", "Source_encoded"]
TARGET = "Categorie"
MIN_ACCURACY = 0.70
MODEL_REGISTRY_NAME = "EcoSmartClassifier"
mlflow.set_tracking_uri("http://localhost:5000")

print(f"[mlflow] Tracking URI: {mlflow.get_tracking_uri()}")

# ══════════════════════════════════════════════════════════════════════
# CHARGEMENT & SPLIT
# ══════════════════════════════════════════════════════════════════════


def load_data(path: str):
    df = pd.read_csv(path)
    df_ml = df[df[TARGET].notna()].copy()
    X = df_ml[FEATURES]
    y = df_ml[TARGET]
    print(f"[data] {len(df_ml)} lignes labellisées | {X.shape[1]} features")
    return X, y, df_ml


def split_data(X, y):
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )
    print(f"[split] Train:{len(X_train)} | Val:{len(X_val)} | Test:{len(X_test)}")
    return X_train, X_val, X_test, y_train, y_val, y_test


# ══════════════════════════════════════════════════════════════════════
# CLASSIFICATION — BASELINE
# ══════════════════════════════════════════════════════════════════════


def train_baseline_models(X_train, X_val, y_train, y_val, experiment_name: str):
    """Entraîne plusieurs modèles baseline et les logue dans MLflow."""
    mlflow.set_experiment(experiment_name)

    modeles = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "SVM": SVC(random_state=42),
        "KNN": KNeighborsClassifier(),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
    }

    resultats = {}
    print("\n=== Baseline Models ===")
    for nom, modele in modeles.items():
        with mlflow.start_run(run_name=f"baseline_{nom.replace(' ', '_')}"):
            modele.fit(X_train, y_train)
            y_pred = modele.predict(X_val)
            acc = accuracy_score(y_val, y_pred)
            f1 = f1_score(y_val, y_pred, average="weighted")

            mlflow.log_param("model", nom)
            mlflow.log_param("features", str(FEATURES))
            mlflow.log_metric("accuracy_val", acc)
            mlflow.log_metric("f1_val", f1)
            mlflow.sklearn.log_model(modele, artifact_path="model")

            resultats[nom] = {"Accuracy": acc, "F1": f1, "model": modele}
            print(f"  {nom:25s} → Acc: {acc:.4f} | F1: {f1:.4f}")

    return resultats


# ══════════════════════════════════════════════════════════════════════
# CLASSIFICATION — TUNING RANDOM FOREST
# ══════════════════════════════════════════════════════════════════════


def tune_random_forest(X_train, X_val, y_train, y_val, experiment_name: str):
    """GridSearchCV sur Random Forest avec MLflow + enregistrement au Registry."""
    mlflow.set_experiment(experiment_name)

    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
    }

    grid_search = GridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid,
        cv=5,
        scoring="accuracy",
        n_jobs=-1,
        verbose=0,
    )
    grid_search.fit(X_train, y_train)
    best = grid_search.best_estimator_

    y_pred_val = best.predict(X_val)
    acc_val = accuracy_score(y_val, y_pred_val)
    f1_val = f1_score(y_val, y_pred_val, average="weighted")
    cv_mean = grid_search.best_score_

    with mlflow.start_run(run_name="RF_GridSearchCV_tuned") as run:
        mlflow.log_params(grid_search.best_params_)
        mlflow.log_metric("accuracy_val", acc_val)
        mlflow.log_metric("f1_val", f1_val)
        mlflow.log_metric("cv_accuracy_mean", cv_mean)

        model_info = mlflow.sklearn.log_model(
            best,
            artifact_path="model",
            registered_model_name=MODEL_REGISTRY_NAME,
        )
        registered_run_id = run.info.run_id

    print(f"\n[tuning] Best params : {grid_search.best_params_}")
    print(f"[tuning] Val Acc: {acc_val:.4f} | CV: {cv_mean:.4f}")
    return best, acc_val, registered_run_id


# ══════════════════════════════════════════════════════════════════════
# MLFLOW MODEL REGISTRY — PROMOTION STAGING → PRODUCTION
# ══════════════════════════════════════════════════════════════════════


def promote_model_to_production(model_name: str, acc_test: float):
    """
    Récupère la dernière version du modèle enregistré et la promeut
    vers Staging puis Production si accuracy >= MIN_ACCURACY.
    """
    client = MlflowClient()

    # Récupérer toutes les versions du modèle
    try:
        versions = client.search_model_versions(f"name='{model_name}'")
    except Exception as e:
        print(f"[registry] Erreur lors de la récupération des versions : {e}")
        return

    if not versions:
        print(f"[registry] Aucune version trouvée pour '{model_name}'")
        return

    # Prendre la version la plus récente
    latest_version = max(versions, key=lambda v: int(v.version))
    version_number = latest_version.version

    print(f"\n=== MLflow Model Registry — Promotion ===")
    print(f"[registry] Modèle    : {model_name}")
    print(f"[registry] Version   : {version_number}")
    print(f"[registry] Accuracy  : {acc_test:.4f}")

    # Ajouter une description à la version
    client.update_model_version(
        name=model_name,
        version=version_number,
        description=(
            f"Random Forest tuné par GridSearchCV. "
            f"Accuracy test = {acc_test:.4f}. "
            f"Features : {FEATURES}."
        ),
    )

    if acc_test < MIN_ACCURACY:
        print(
            f"[registry] Accuracy {acc_test:.4f} < seuil {MIN_ACCURACY} "
            f"— promotion annulée."
        )
        return

    # Promouvoir vers Staging
    client.transition_model_version_stage(
        name=model_name,
        version=version_number,
        stage="Staging",
        archive_existing_versions=True,
    )
    print(f"[registry] {model_name} v{version_number} → Staging ✓")

    # Promouvoir vers Production
    client.transition_model_version_stage(
        name=model_name,
        version=version_number,
        stage="Production",
        archive_existing_versions=True,
    )
    print(f"[registry] {model_name} v{version_number} → Production ✓")

    # Ajouter un tag de production
    client.set_model_version_tag(
        name=model_name,
        version=version_number,
        key="stage",
        value="production",
    )
    client.set_model_version_tag(
        name=model_name,
        version=version_number,
        key="accuracy_test",
        value=str(round(acc_test, 4)),
    )
    print(f"[registry] Tags ajoutés : stage=production, accuracy_test={acc_test:.4f}")


# ══════════════════════════════════════════════════════════════════════
# CLASSIFICATION — ÉVALUATION FINALE (TEST SET)
# ══════════════════════════════════════════════════════════════════════


def evaluate_final_classifier(model, X_test, y_test, experiment_name: str):
    mlflow.set_experiment(experiment_name)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")

    with mlflow.start_run(run_name="final_test_evaluation"):
        mlflow.log_metric("accuracy_test", acc)
        mlflow.log_metric("f1_test", f1)
        mlflow.log_param("split", "test")

    print(f"\n=== Évaluation Finale Classification ===")
    print(f"\n[test] Accuracy : {acc:.4f} | F1 : {f1:.4f}")
    print(classification_report(y_test, y_pred))

    if acc >= MIN_ACCURACY:
        print(f"[✓] Seuil minimum {MIN_ACCURACY} atteint ({acc:.4f})")
    else:
        print(f"[✗] Seuil minimum {MIN_ACCURACY} NON atteint ({acc:.4f}) !")

    # Sauvegarde métriques JSON
    os.makedirs("reports", exist_ok=True)
    with open("reports/metrics_classification.json", "w") as f:
        json.dump(
            {"accuracy_test": round(acc, 4), "f1_test": round(f1, 4)}, f, indent=2
        )

    return acc, f1


# ══════════════════════════════════════════════════════════════════════
# RÉGRESSION — PRÉDICTION PRIX
# ══════════════════════════════════════════════════════════════════════


def train_regression(df_ml, experiment_name: str):
    """Entraîne plusieurs régresseurs pour prédire Prix_Revente."""
    mlflow.set_experiment(experiment_name)

    X_reg = df_ml[FEATURES]
    y_reg = df_ml["Prix_Revente"]

    X_train_r, X_temp_r, y_train_r, y_temp_r = train_test_split(
        X_reg, y_reg, test_size=0.30, random_state=42
    )
    X_val_r, X_test_r, y_val_r, y_test_r = train_test_split(
        X_temp_r, y_temp_r, test_size=0.50, random_state=42
    )

    modeles_reg = {
        "Linear Regression": LinearRegression(),
        "Ridge": Ridge(),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42),
        "SVR": SVR(),
    }

    resultats_reg = {}
    print("\n=== Modèles Régression ===")
    for nom, modele in modeles_reg.items():
        with mlflow.start_run(run_name=f"reg_{nom.replace(' ', '_')}"):
            modele.fit(X_train_r, y_train_r)
            y_pred = modele.predict(X_test_r)
            rmse = float(np.sqrt(mean_squared_error(y_test_r, y_pred)))
            r2 = float(r2_score(y_test_r, y_pred))

            mlflow.log_param("model", nom)
            mlflow.log_metric("rmse_test", rmse)
            mlflow.log_metric("r2_test", r2)

            resultats_reg[nom] = {"RMSE": rmse, "R2": r2, "model": modele}
            print(f"  {nom:25s} → RMSE: {rmse:.4f} | R²: {r2:.4f}")

    # Meilleur régresseur
    best_reg_nom = max(resultats_reg, key=lambda k: resultats_reg[k]["R2"])
    best_reg = resultats_reg[best_reg_nom]["model"]
    print(
        f"\n[regression] Meilleur modèle : {best_reg_nom} "
        f"(R²={resultats_reg[best_reg_nom]['R2']:.4f})"
    )
    return best_reg


# ══════════════════════════════════════════════════════════════════════
# POINT D'ENTRÉE PRINCIPAL
# ══════════════════════════════════════════════════════════════════════


def train(
    data_path: str,
    models_dir: str = "models",
    experiment: str = "EcoSmart_Classification",
):
    os.makedirs(models_dir, exist_ok=True)

    X, y, df_ml = load_data(data_path)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    # ── 1. Baseline ────────────────────────────────────────────────
    resultats = train_baseline_models(X_train, X_val, y_train, y_val, experiment)

    # ── 2. Tuning + enregistrement au Registry ─────────────────────
    print("\n=== Tuning Random Forest ===")
    best_clf, acc_val, run_id = tune_random_forest(
        X_train, X_val, y_train, y_val, experiment
    )

    # ── 3. Évaluation finale sur test set ──────────────────────────
    acc_test, f1_test = evaluate_final_classifier(best_clf, X_test, y_test, experiment)

    # ── 4. Promotion Staging → Production dans le Registry ─────────
    promote_model_to_production(MODEL_REGISTRY_NAME, acc_test)

    # ── 5. Régression ──────────────────────────────────────────────
    best_reg = train_regression(df_ml, experiment)

    # ── 6. Sauvegarde locale des modèles ───────────────────────────
    joblib.dump(best_clf, os.path.join(models_dir, "classifier_best.pkl"))
    joblib.dump(best_reg, os.path.join(models_dir, "regressor_best.pkl"))
    print(f"\n[saved] classifier_best.pkl + regressor_best.pkl → {models_dir}/")

    return best_clf, best_reg, acc_test


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/processed/dataset_clean.csv")
    parser.add_argument("--models", default="models")
    parser.add_argument("--experiment", default="EcoSmart_Classification")
    args = parser.parse_args()
    train(args.data, args.models, args.experiment)
