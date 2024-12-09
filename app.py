import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import requests

# Charger les modèles sauvegardés
svm_model = joblib.load('svm_model.pkl')
scaler = joblib.load('scaler.pkl')
pca = joblib.load('pca.pkl')
kmeans = joblib.load('kmeans.pkl')

# Définir les colonnes attendues par le modèle
column_names = [
    "age", "sex", "cp", "trtbps", "chol", "fbs",
    "restecg", "thalachh", "exng", "oldpeak", "slp", 'caa', 'thall'
]

# Mapping des noms complets pour le formulaire
column_labels = {
    "age": "Age",
    "sex": "Gender",
    "cp": "Chest Pain",
    "trtbps": "Resting Blood Pressure",
    "chol": "Cholesterol",
    "fbs": "Fasting blood sugar level categorized as above 120 mg/dl",
    "restecg": "Resting ECG",
    "thalachh": "Maximum heart rate achieved during a stress test",
    "exng": "Exercise Induced Angina",
    "oldpeak": "ST depression induced by exercise relative to rest",
    "slp": "Slope of the peak exercise ST segment",
    "caa": "Number of Major Vessels colored by fluoroscopy",
    "thall": "Thalium Stress Test Result"
}

# Mapping des clusters pour les recommandations
cluster_names = {
    0: "Surveillance stricte, alimentation saine, activité physique accrue.",
    1: "Maintien des habitudes saines, surveillance préventive et gestion du stress."
}

# Seuils pour les variables critiques
chol_threshold = 200  # Seuil pour cholestérol total (mg/dL)
bp_threshold = 130    # Seuil pour pression artérielle (trtbps)
thalachh_low = 100    # Seuil bas pour fréquence cardiaque maximale (thalachh)

# Initialiser un DataFrame pour stocker les prédictions
if "data" not in st.session_state:
    st.session_state.data = pd.DataFrame(columns=column_names + ['Prediction', 'Cluster'])




# Function to interact with the Gemini API
def get_gemini_response(user_input):
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent"  # Correct API endpoint for Gemini
    headers = {
        "Content-Type": "application/json"
    }
    params = {
        "key": "AIzaSyBE1AXHM84mbW_jBrA-ArwqZodEDLuAbvU"  # Use your actual Gemini API key
    }
    data = {
        "contents": [{
            "parts": [{"text": user_input}]
        }]
    }
    response = requests.post(url, headers=headers, params=params, json=data)
    
    if response.status_code == 200:
        response_json = response.json()
        try:
            # Attempt to retrieve the generated content from Gemini (updated key)
            return response_json['candidates'][0]['content']['parts'][0]['text']
        except KeyError as e:
            # If the response format is incorrect or key is missing, return a generic error message
            return "Sorry, the response format is incorrect."
    else:
        # If the API call fails, return a generic error message
        return "Sorry, there was an error processing your request."



# Fonction de login
def login():
    st.title("Heart Attack Prediction & Recommendation App")
    st.header("Login")
    
    # Collecte du nom et de l'email
    name = st.text_input("Name :")
    email = st.text_input("Email :")
    
    if st.button("Login"):
        if name and email:
            # Sauvegarder dans session_state pour vérifier si l'utilisateur est connecté
            st.session_state.logged_in = True
            st.session_state.user_name = name
            st.session_state.user_email = email
            st.success(f"Welcome {name} ! You are connected.")
        else:
            st.error("Please complete all fields.")

# Si l'utilisateur est déjà connecté
if 'logged_in' in st.session_state and st.session_state.logged_in:
    # Afficher le formulaire de prédiction après la connexion
    st.title("Heart Attack Prediction & Recommendation")

            # Streamlit interface
    st.title("Chatbot")

    user_input = st.text_input("Question", "")

    if user_input:
        response = get_gemini_response(user_input)
        st.write(f"Bot: {response}")

    # Formulaire pour entrer les données utilisateur
    st.header("Entrer vos données")
    user_input = {}
    sex_mapping = {"Woman": 0, "Man": 1}
    user_input['sex'] = sex_mapping[st.selectbox("Sex", options=list(sex_mapping.keys()))]

    # Mapping des valeurs pour Chest Pain
    chest_pain_mapping = {
        "Typical angina": 0,
        "Atypical angina": 1,
        "Non-anginal pain": 2,
        "Asymptomatic": 3
    }
    user_input['cp'] = chest_pain_mapping[st.selectbox("Chest Pain", options=list(chest_pain_mapping.keys()))]

    # Mapping des valeurs pour Fasting Blood Sugar
    fbs_mapping = {"False (≤120 mg/dL)": 0, "True (>120 mg/dL)": 1}
    user_input['fbs'] = fbs_mapping[st.selectbox("Fasting blood sugar level categorized as above 120 mg/dL", options=list(fbs_mapping.keys()))]

    # Mapping des valeurs pour Resting ECG
    resting_ecg_mapping = {
        "Normal": 0,
        "Having ST-T wave abnormality": 1,
        "Showing probable or definite left ventricular hypertrophy": 2
    }
    user_input['restecg'] = resting_ecg_mapping[st.selectbox("Resting electrocardiographic results", options=list(resting_ecg_mapping.keys()))]

    # Mapping des valeurs pour Exercise Induced Angina
    exng_mapping = {
        "no": 0,
        "yes": 1
    }
    user_input['exng'] = exng_mapping[st.selectbox("Exercise Induced Angina", options=list(exng_mapping.keys()))]

    # Mapping des valeurs pour ST Slope
    slp_mapping = {
        "Upsloping": 0,
        "Flat": 1,
        "Downsloping": 2
    }
    user_input['slp'] = slp_mapping[st.selectbox("Slope of the peak exercise ST segment", options=list(slp_mapping.keys()))]

    # Mapping des valeurs pour Number of Major Vessels (caa)
    caa_mapping = {
        0: 0,
        1: 1,
        2: 2,
        3: 3,
        4: 4
    }
    user_input['caa'] = int(caa_mapping[st.selectbox("Number of Major Vessels", options=list(caa_mapping.values()))])

    # Mapping des valeurs pour Thalium Stress Test Result (thall)
    thall_mapping = {
        "Normal" : 0,
        "Fixed defect" : 1,
        "Reversible defect" : 2,
        "Not described" : 3
    }
    user_input['thall'] = thall_mapping[st.selectbox("Thalium Stress Test Result", options=list(thall_mapping.keys()))]
    
    # Créer le formulaire avec les noms complets
    for col in column_names:
        if col == "chol":  # On ajoute l'unité mg/dL pour ce champ
            user_input[col] = st.number_input(f"{column_labels[col]} (en mg/dL)", value=0.0)
        elif col == "trtbps":  # On ajoute l'unité mmHg pour ce champ
            user_input[col] = st.number_input(f"{column_labels[col]} (en mmHg)", value=0.0)
        elif col != "sex" and col != "cp" and col != "fbs" and col != "restecg" and col != "exng" and col != "slp" and col != "caa" and col != "thall":  # On saute les champs déjà ajoutés
            user_input[col] = st.number_input(f"{column_labels[col]}", value=0.0)

    # Bouton pour prédire et recommander
    if st.button("Predict and recommend"):
        # Convertir les entrées utilisateur en DataFrame
        input_df = pd.DataFrame([user_input])
        input_df = input_df[column_names]

        # Appliquer le scaling (StandardScaler)
        input_scaled = scaler.transform(input_df)

        # Appliquer PCA pour réduire la dimensionnalité
        input_pca = pca.transform(input_scaled)

        # Prédire le cluster avec K-means
        cluster_label = kmeans.predict(input_pca)[0]
        cluster_name = cluster_names[cluster_label]

        # Prédire avec le modèle SVM
        prediction = svm_model.predict(input_df)[0]

        # Ajouter les résultats au DataFrame global
        user_input['Prediction'] = prediction
        user_input['Cluster'] = cluster_name
        st.session_state.data = pd.concat(
            [st.session_state.data, pd.DataFrame([user_input])],
            ignore_index=True
        )

        if prediction == 0:
            st.success("Bonne nouvelle ! Aucune indication de risque de crise cardiaque détectée. 😊")
        else:
            st.error("Attention ! Il y a une chance d'avoir une crise cardiaque. 🚨")
        # Afficher les recommandations générales
        st.write(f"### Recommendations : **{cluster_name}**")
        # Analyse personnalisée
        st.write("### Analyse Personnalisée de Votre Bilan :")
        if user_input['chol'] > chol_threshold:
            st.warning(f"⚠️ Votre taux de cholestérol est **{user_input['chol']} mg/dL**, ce qui est supérieur au seuil normal ({chol_threshold} mg/dL).")
            st.write("💡 **Recommandation :** Adoptez une alimentation pauvre en graisses saturées et faites des exercices réguliers pour réduire votre cholestérol.")

        if user_input['trtbps'] > bp_threshold:
            st.warning(f"⚠️ Votre pression artérielle est **{user_input['trtbps']} mmHg**, ce qui est supérieure au seuil normal ({bp_threshold} mg/dL).")
            st.write("💡 **Recommandation :** Réduisez votre consommation de sel et consultez un médecin pour un suivi de votre tension artérielle.")

        if user_input['thalachh'] < thalachh_low:
            st.warning(f"⚠️ Votre fréquence cardiaque maximale est **{user_input['thalachh']} bpm**, ce qui est relativement faible.")
            st.write("💡 **Recommandation :** Consultez un professionnel de santé pour évaluer votre condition cardiaque.")

        # Graphiques comparatifs
        st.write("### Comparaison avec les Valeurs Normales")
        fig, ax = plt.subplots()
        ax.bar(['Utilisateur', 'Seuil Normal'], [user_input['chol'], chol_threshold], color=['orange', 'green'])
        ax.set_title("Taux de Cholestérol")
        ax.set_ylabel("Cholestérol (mg/dL)")
        st.pyplot(fig)

        fig, ax = plt.subplots()
        ax.bar(['Utilisateur', 'Seuil Normal'], [user_input['trtbps'], bp_threshold], color=['red', 'green'])
        ax.set_title("Pression Artérielle")
        ax.set_ylabel("Pression Artérielle (mmHg)")
        st.pyplot(fig)

        fig, ax = plt.subplots()
        ax.bar(['Utilisateur', 'Seuil Bas'], [user_input['thalachh'], thalachh_low], color=['blue', 'green'])
        ax.set_title("Fréquence Cardiaque Maximale")
        ax.set_ylabel("Fréquence Cardiaque (bpm)")
        st.pyplot(fig)

else:
    # Si l'utilisateur n'est pas encore connecté, afficher le formulaire de login
    login()
