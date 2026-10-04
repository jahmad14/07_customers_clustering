# Déploiement — Segmentation des clients (Customer Segmentation)

Application interactive développée avec **Streamlit** pour classifier un nouveau profil client ou détecter une anomalie comportementale via le modèle **DBSCAN** du professeur.

---

## 1. Modèle & Méthodologie

- **Modèle retenu :** `DBSCAN(eps=0.2, min_samples=9)` (`modele_dbscan.joblib`), appris sur les 3 150 clients après normalisation $L_2$ (`normalize`).
- **Affectation de nouveaux clients (Règle DBSCAN du professeur) :**
  - Le vecteur du client est normalisé unitairement (`normalize`).
  - La distance euclidienne minimale vers les 3 119 points cœur est calculée.
  - Si cette distance est inférieure ou égale à $\varepsilon = 0.2$, le client est affecté à la classe du point cœur le plus proche (`0 : Client fidèle` ou `1 : Client non fidèle`).
  - Si la distance dépasse $\varepsilon = 0.2$, le client est étiqueté **Anomalie (-1)** (profil atypique hors densité).
- **Variables explicatives (12 variables) :** `['Call Failure', 'Complains', 'Subscription Length', 'Charge Amount', 'Seconds of Use', 'Frequency of use', 'Frequency of SMS', 'Distinct Called Numbers', 'Age Group', 'Status', 'Age', 'Customer Value']`.

---

## 2. Structure du dossier

```text
Deployment/
├── app.py              # Interface utilisateur Streamlit
├── modele_dbscan.joblib# Points cœur, labels cœur, eps et métadonnées
├── requirements.txt    # Dépendances exactes (sans xgboost)
├── test_app.py         # Tests unitaires automatisés avec AppTest
└── README.md           # Documentation du projet
```

---

## 3. Installation et exécution

```bash
pip install -r requirements.txt
streamlit run app.py
```
