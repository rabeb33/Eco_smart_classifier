# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import requests
import json
import os

# Configuration de la page
st.set_page_config(
    page_title="Eco-Smart Classifier",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Titre principal
st.title("♻️ Eco-Smart Classifier")
st.markdown("---")

# -------------------------------
# 1. CHARGEMENT DES MODÈLES ET DONNÉES
# -------------------------------
@st.cache_resource
def load_models():
    """Charge les modèles ML et NLP depuis le dossier models/"""
    try:
        classifier = joblib.load("models/classifier_best.pkl")
        nlp_classifier = joblib.load("models/nlp_classifier.pkl")
        vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
        return classifier, nlp_classifier, vectorizer
    except FileNotFoundError as e:
        st.error(f"Modèle non trouvé : {e}. Vérifiez que les modèles sont dans le dossier 'models/'.")
        return None, None, None

@st.cache_data
def load_data():
    """Charge le dataset nettoyé et les clusters (si existants)"""
    try:
        df = pd.read_csv("projet_ML2026/data/processed/dataset_clean.csv")
    except:
        df = pd.read_csv("data/processed/dataset_clean.csv")
    
    # Si les clusters existent, on les charge, sinon on les calcule
    cluster_path = "projet_ML2026/data/processed/dataset_clusters.csv"
    if not os.path.exists(cluster_path):
        cluster_path = "data/processed/dataset_clusters.csv"
    
    if os.path.exists(cluster_path):
        clusters = pd.read_csv(cluster_path)
        if 'Cluster' in clusters.columns:
            df['Cluster'] = clusters['Cluster']
        else:
            df = compute_clusters(df)
    else:
        df = compute_clusters(df)
    return df

def compute_clusters(df):
    """Calcule les clusters PCA si le fichier n'existe pas"""
    from sklearn.cluster import KMeans
    features = ["Poids", "Volume", "Conductivite", "Opacite", "Rigidite", "Source_encoded"]
    X = df[features].dropna()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    kmeans = KMeans(n_clusters=4, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)
    # Rejoindre les clusters au DataFrame
    df_clustered = df.copy()
    df_clustered.loc[X.index, 'Cluster'] = clusters
    df_clustered['Cluster'] = df_clustered['Cluster'].fillna(-1).astype(int)
    return df_clustered

# -------------------------------
# 2. ONGLETS
# -------------------------------
tab1, tab2, tab3 = st.tabs(["📊 Dashboard Data", "🎚️ Prédiction Manuelle", "🤖 Assistant Intelligent"])

# ---------- TAB 1 : Dashboard Data ----------
with tab1:
    st.header("Visualisation des données et clusters")
    df = load_data()
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Aperçu des données")
        st.dataframe(df.head(100), use_container_width=True)
        st.caption(f"Total : {len(df)} lignes | {df['Categorie'].nunique()} catégories")
    
    with col2:
        st.subheader("Distribution des catégories")
        fig_cat = px.bar(df['Categorie'].value_counts(), 
                         title="Nombre d'échantillons par catégorie",
                         labels={'value':'Count', 'index':'Catégorie'})
        st.plotly_chart(fig_cat, use_container_width=True)
    
    st.subheader("Projection PCA des clusters (non-supervisé)")
    # Sélection des features numériques
    features_num = ["Poids", "Volume", "Conductivite", "Opacite", "Rigidite"]
    df_clean = df[features_num].dropna()
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df_clean)
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(scaled)
    df_pca = pd.DataFrame(pca_result, columns=['PCA1', 'PCA2'])
    df_pca['Cluster'] = df.loc[df_clean.index, 'Cluster'].values
    
    fig_pca = px.scatter(df_pca, x='PCA1', y='PCA2', color='Cluster',
                         title="Visualisation des clusters (PCA)",
                         color_continuous_scale='Viridis')
    st.plotly_chart(fig_pca, use_container_width=True)
    
    with st.expander("Informations sur les clusters"):
        st.write("Nombre de clusters déterminé par la méthode du coude (Elbow Method).")
        st.write("Les clusters représentent des sous-groupes de matériaux basés sur les propriétés physiques.")

# ---------- TAB 2 : Prédiction Manuelle ----------
with tab2:
    st.header("Prédiction en temps réel")
    st.markdown("Ajustez les curseurs pour voir la catégorie du déchet s'afficher instantanément.")
    
    col1, col2 = st.columns([1, 1])
    with col1:
        poids = st.slider("Poids (g)", min_value=0.0, max_value=2000.0, value=100.0, step=10.0)
        volume = st.slider("Volume (cm³)", min_value=0.0, max_value=5000.0, value=500.0, step=50.0)
        conductivite = st.slider("Conductivité (S/m)", min_value=0.0, max_value=100.0, value=10.0, step=1.0)
    with col2:
        opacite = st.slider("Opacité (%)", min_value=0.0, max_value=100.0, value=50.0, step=5.0)
        rigidite = st.slider("Rigidité (MPa)", min_value=0.0, max_value=500.0, value=150.0, step=10.0)
        source = st.selectbox("Source", options=[0, 1], format_func=lambda x: "Ménagère" if x==0 else "Industrielle")
    
    # Préparation des features
    features = np.array([[poids, volume, conductivite, opacite, rigidite, source]])
    
    # Chargement du modèle
    classifier, _, _ = load_models()
    if classifier is not None:
        prediction = classifier.predict(features)[0]
        # Afficher la prédiction en grand
        st.markdown("---")
        st.subheader("🔄 Résultat de la classification")
        col_a, col_b, col_c = st.columns([1,2,1])
        with col_b:
            st.markdown(f"<h1 style='text-align: center; color: #2ecc71;'>{prediction}</h1>", unsafe_allow_html=True)
            st.markdown("<p style='text-align: center;'>Catégorie prédite</p>", unsafe_allow_html=True)
        
        # Optionnel : afficher les probabilités si disponible
        if hasattr(classifier, "predict_proba"):
            proba = classifier.predict_proba(features)[0]
            proba_df = pd.DataFrame({"Catégorie": classifier.classes_, "Probabilité": proba})
            st.bar_chart(proba_df.set_index("Catégorie"))
    else:
        st.error("Modèle non disponible. Vérifiez le chargement.")

# ---------- TAB 3 : Assistant Intelligent (NLP) ----------
with tab3:
    st.header("🤖 Assistant Intelligent")
    st.markdown("Décrivez le déchet en français, l'assistant prédira sa catégorie.")
    
    user_text = st.text_area("Description du déchet", height=150, 
                              placeholder="Exemple : 'Bouteille en plastique transparente, légère, collectée dans un bac jaune.'")
    
    col_btn, _ = st.columns([1,3])
    with col_btn:
        if st.button("Analyser", type="primary", use_container_width=True):
            if user_text.strip():
                with st.spinner("Analyse en cours..."):
                    # Charger modèles NLP
                    _, nlp_clf, vec = load_models()
                    if nlp_clf and vec:
                        # Prétraitement simplifié (identique au script NLP)
                        import re
                        from nltk.corpus import stopwords
                        from nltk.stem import SnowballStemmer
                        from nltk.tokenize import word_tokenize
                        import nltk
                        nltk.download('stopwords', quiet=True)
                        nltk.download('punkt', quiet=True)
                        
                        stop_words = set(stopwords.words('french'))
                        stemmer = SnowballStemmer('french')
                        
                        def clean_text(text):
                            text = text.lower()
                            text = re.sub(r'[^a-zàâäéèêëîïôùûüç\s]', ' ', text)
                            text = re.sub(r'\s+', ' ', text).strip()
                            tokens = word_tokenize(text, language='french')
                            tokens = [t for t in tokens if t not in stop_words]
                            tokens = [stemmer.stem(t) for t in tokens]
                            return ' '.join(tokens)
                        
                        processed = clean_text(user_text)
                        X = vec.transform([processed])
                        pred = nlp_clf.predict(X)[0]
                        st.success(f"🔮 Catégorie prédite : **{pred}**")
                        
                        # Afficher la confiance si disponible
                        if hasattr(nlp_clf, "decision_function"):
                            confidence = nlp_clf.decision_function(X).max()
                            st.metric("Confiance", f"{confidence:.2f}")
                    else:
                        st.error("Modèle NLP non disponible.")
            else:
                st.warning("Veuillez entrer une description.")
    
    st.markdown("---")
    st.markdown("💡 **Exemples de descriptions :**")
    st.info("- 'Bocal en verre vert, très lourd' → Verre")
    st.info("- 'Feuille de papier froissée, légère' → Papier")
    st.info("- 'Canette en aluminium compressée' → Métal")
    st.info("- 'Emballage plastique souple, transparent' → Plastique")

# -------------------------------
# Pied de page
st.markdown("---")
st.markdown("**Eco-Smart Classifier** | Pipeline ML complet | Classification de déchets et estimation de valeur")