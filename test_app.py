import os
import sys
from streamlit.testing.v1 import AppTest

def test_customer_app():
    dep_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(dep_dir)
    print(f"Testing app.py in: {dep_dir}")

    # 1. Profil 1 : Client fidèle standard (Cluster 0 attendu)
    at1 = AppTest.from_file("app.py", default_timeout=30)
    at1.run()
    assert len(at1.exception) == 0, f"Exception initialisation 1 : {at1.exception}"

    # Valeurs par défaut du formulaire (client médian)
    at1.button[0].click().run()
    assert len(at1.exception) == 0, f"Exception prédiction 1 : {at1.exception}"
    assert len(at1.success) > 0, "Aucun message de succès trouvé pour test 1"
    res1 = at1.success[0].value
    print(f"Résultat Test 1 : {res1}")

    # 2. Profil 2 : Client avec comportement extrême / atypique (Anomalie -1 attendue)
    at2 = AppTest.from_file("app.py", default_timeout=30)
    at2.run()
    assert len(at2.exception) == 0, f"Exception initialisation 2 : {at2.exception}"

    at2.number_input[0].set_value(0.0)    # Call Failure
    at2.number_input[1].set_value(1.0)    # Complains
    at2.number_input[2].set_value(1.0)    # Subscription Length
    at2.number_input[3].set_value(50.0)   # Charge Amount
    at2.number_input[4].set_value(1.0)    # Seconds of Use
    at2.number_input[5].set_value(1.0)    # Frequency of use
    at2.number_input[6].set_value(1000.0) # Frequency of SMS
    at2.number_input[7].set_value(1.0)    # Distinct Called Numbers
    at2.number_input[8].set_value(1.0)    # Age Group
    at2.number_input[9].set_value(1.0)    # Status
    at2.number_input[10].set_value(99.0)  # Age
    at2.number_input[11].set_value(1.0)   # Customer Value

    at2.button[0].click().run()
    assert len(at2.exception) == 0, f"Exception prédiction 2 : {at2.exception}"
    assert len(at2.warning) > 0, "Aucun message d'anomalie trouvé pour test 2"
    res2 = at2.warning[0].value
    print(f"Résultat Test 2 : {res2}")

    assert res1 != res2, f"Les résultats sont identiques : {res1}"
    print("SUCCESS: Customer segmentation app test passed with 0 exception and two different results (Cluster 0 vs Anomalie)!")

if __name__ == "__main__":
    test_customer_app()
