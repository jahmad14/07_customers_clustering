"""
Application Streamlit — Segmentation de clients (Clustering Doc 2 - Customer Segmentation)
Modèle retenu : DBSCAN du professeur avec normalisation L2 et projection sur les points cœur
Lancement en local : streamlit run app.py
"""

import os
import joblib
import numpy as np
import streamlit as st
from sklearn.preprocessing import normalize

# ----------------------------------------------------------------------
# Configuration de la page
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Customer Segmentation DBSCAN",
    page_icon="👥",
    layout="centered",
)

DESCRIPTION = (
    "Cette application permet de segmenter un client télécom ou de détecter une anomalie comportementale "
    "à partir du modèle **DBSCAN** ($eps=0.2$, $min\\_samples=9$) entraîné sur les données normalisées.\n\n"
    "**Règle d'affectation DBSCAN** : Le profil du client est normalisé ($L_2$) puis comparé aux 3 119 points cœur "
    "mémorisés lors de l'apprentissage. Si la distance minimale au point cœur le plus proche est inférieure ou égale "
    "au rayon de voisinage $\\varepsilon = 0.2$, le client hérite du cluster de ce point cœur (`0 : Client fidèle` "
    "ou `1 : Client non fidèle`). Au-delà de ce seuil ($> 0.2$), le client est classé comme **Anomalie (-1)** (profil atypique ou bruit)."
)

# ----------------------------------------------------------------------
# Chargement des artefacts (mis en cache)
# ----------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    base_dir = os.path.dirname(__file__)
    artefacts = joblib.load(os.path.join(base_dir, "modele_dbscan.joblib"))
    return artefacts


artefacts = load_artifacts()
points_coeur = artefacts["points_coeur"]
labels_coeur = artefacts["labels_coeur"]
eps = float(artefacts["eps"])
colonnes = artefacts["colonnes"]
valeurs_defaut = artefacts["valeurs_defaut"]
noms_classes = artefacts["noms_classes"]

# ----------------------------------------------------------------------
# Fonction d'affectation DBSCAN
# ----------------------------------------------------------------------
def predict_customer_segment(features_list):
    client = np.array(features_list, dtype=float).reshape(1, -1)
    client_norm = normalize(client)

    distances = np.linalg.norm(points_coeur - client_norm, axis=1)
    plus_proche = int(distances.argmin())
    min_dist = float(distances[plus_proche])

    if min_dist <= eps:
        classe_id = int(labels_coeur[plus_proche])
        nom_cluster = noms_classes.get(classe_id, f"Cluster {classe_id}")
        return classe_id, nom_cluster, min_dist
    else:
        return -1, "Anomalie / Bruit statistique (-1)", min_dist


# ----------------------------------------------------------------------
# Interface utilisateur
# ----------------------------------------------------------------------
st.title("👥 Customer Segmentation (DBSCAN)")
st.write(DESCRIPTION)

with st.form("form_customer_segmentation"):
    st.subheader("Caractéristiques comportementales du client")

    col1, col2 = st.columns(2)
    with col1:
        call_failure = st.number_input("Nombre d'appels échoués (Call Failure)", min_value=0.0, max_value=100.0, value=float(valeurs_defaut[0]), step=1.0)
        complains = st.number_input("Réclamations (Complains : 0 ou 1)", min_value=0.0, max_value=1.0, value=float(valeurs_defaut[1]), step=1.0)
        sub_length = st.number_input("Ancienneté d'abonnement (Subscription Length)", min_value=1.0, max_value=100.0, value=float(valeurs_defaut[2]), step=1.0)
        charge_amount = st.number_input("Montant des recharges (Charge Amount)", min_value=0.0, max_value=50.0, value=float(valeurs_defaut[3]), step=1.0)
        sec_of_use = st.number_input("Durée totale des appels en secondes (Seconds of Use)", min_value=0.0, max_value=50000.0, value=float(valeurs_defaut[4]), step=100.0)
        freq_use = st.number_input("Nombre total d'appels (Frequency of use)", min_value=0.0, max_value=500.0, value=float(valeurs_defaut[5]), step=5.0)

    with col2:
        freq_sms = st.number_input("Nombre total de SMS (Frequency of SMS)", min_value=0.0, max_value=1000.0, value=float(valeurs_defaut[6]), step=5.0)
        distinct_nums = st.number_input("Numéros distincts appelés (Distinct Called Numbers)", min_value=0.0, max_value=200.0, value=float(valeurs_defaut[7]), step=1.0)
        age_group = st.number_input("Tranche d'âge (Age Group : 1 à 5)", min_value=1.0, max_value=5.0, value=float(valeurs_defaut[8]), step=1.0)
        status = st.number_input("Statut du compte (Status : 1 ou 2)", min_value=1.0, max_value=2.0, value=float(valeurs_defaut[9]), step=1.0)
        age = st.number_input("Âge exact (Age)", min_value=15.0, max_value=100.0, value=float(valeurs_defaut[10]), step=1.0)
        cust_val = st.number_input("Valeur client (Customer Value)", min_value=0.0, max_value=5000.0, value=float(valeurs_defaut[11]), step=10.0)

    submit_btn = st.form_submit_button("Identifier le segment client", type="primary")

if submit_btn:
    try:
        inputs = [
            call_failure, complains, sub_length, charge_amount, sec_of_use,
            freq_use, freq_sms, distinct_nums, age_group, status, age, cust_val
        ]
        classe_id, nom_cluster, min_dist = predict_customer_segment(inputs)

        if classe_id == 0:
            st.success(f"✅ **Segment attribué : {nom_cluster} (Cluster 0)**")
        elif classe_id == 1:
            st.info(f"ℹ️ **Segment attribué : {nom_cluster} (Cluster 1)**")
        else:
            st.warning(f"⚠️ **Statut détecté : {nom_cluster}**")

        st.write("### Métriques de voisinage DBSCAN :")
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.metric(label="Distance au point cœur le plus proche", value=f"{min_dist:.4f}")
        with col_m2:
            st.metric(label="Rayon critique de voisinage (eps)", value=f"{eps:.2f}")

    except Exception as e:
        st.error(f"Erreur lors de la prédiction : {e}")
