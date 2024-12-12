import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

svm_model = joblib.load('svm_model.pkl')
scaler = joblib.load('scaler.pkl')
pca = joblib.load('pca.pkl')
kmeans = joblib.load('kmeans.pkl')

column_names = [
    "age", "sex", "cp", "trtbps", "chol", "fbs",
    "restecg", "thalachh", "exng", "oldpeak", "slp", 'caa', 'thall'
]

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

cluster_names = {
    0: "Patient appartient au Groupe à Risque cardiovasculaire accru.",
    1: "Patient appartient au Groupe à Risque cardiovasculaire faible."
}

recommendation_msg = {
    0: "Surveillance stricte, alimentation saine, activité physique accrue.",
    1: "Maintien des habitudes saines, surveillance préventive et gestion du stress."
}

chol_threshold = 200  
bp_threshold = 130   
thalachh_low = 100  

if "data" not in st.session_state:
    st.session_state.data = pd.DataFrame(columns=column_names + ['Prediction', 'Cluster'])

def get_gemini_response(user_input):
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent"  
    headers = {
        "Content-Type": "application/json"
    }
    params = {
        "key": "AIzaSyBE1AXHM84mbW_jBrA-ArwqZodEDLuAbvU" 
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
            return response_json['candidates'][0]['content']['parts'][0]['text']
        except KeyError as e:
            return "Sorry, the response format is incorrect."
    else:
        return "Sorry, there was an error processing your request."

def login():
    st.image("heart_logo.png", caption="", use_container_width=False, width=250)
    st.title("CardioAlert 🚨 Heart Attack Prediction & Recommendation App")    
    name = st.text_input("Name :")
    email = st.text_input("Email :")

    if st.button("Login"):
        if not name or not email:
            st.error("Please complete all fields.")
        elif not email.endswith("@gmail.com"):
            st.error("Invalid email address; it must end with @gmail.com.")
        else:
            st.session_state.logged_in = True
            st.session_state.user_name = name
            st.session_state.user_email = email
            st.success(f"Welcome {name}! You are connected.")

def send_email(recipient_email, subject, body):
    sender_email = "yassinetrabelsi110@gmail.com"  # Remplacez par votre adresse Gmail
    sender_password = "vntw llux heln btyr"      # Remplacez par votre mot de passe Gmail ou mot de passe d'application
    
    try:
        # Création de l'e-mail
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = recipient_email
        msg['Subject'] = subject
        
        # Corps de l'e-mail
        msg.attach(MIMEText(body, 'plain'))
        
        # Connexion au serveur SMTP
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender_email, sender_password)
        
        # Envoi de l'e-mail
        server.sendmail(sender_email, recipient_email, msg.as_string())
        server.quit()
        return True
    except Exception as e:
        st.error(f"Erreur lors de l'envoi de l'e-mail : {str(e)}")
        return False


def home_page():
    st.title("CardioAlert App 🚨")
    st.title("All You Need to Know About Heart Disease!")
    st.write("""
        **Heart disease**, also known as cardiovascular disease (CVD), is a group of conditions that affect the heart and blood vessels.
    """)
    st.markdown("### What are the symptoms of heart disease?")
    st.write("""
        Sometimes heart disease may be “silent” and not diagnosed until a person experiences signs or symptoms of:
        - A heart attack
        - Heart failure
        - An arrhythmia
    """)
    st.markdown("### What are the risk factors for heart disease?")
    st.write("""
        High blood pressure, high blood cholesterol, and smoking are key risk factors for heart disease. Several other medical conditions and lifestyle choices can also put people at a higher risk for heart disease, including:
    """)
    st.write("""
        - **Diabetes**
        - **Overweight and obesity**
        - **Unhealthy diet**
        - **Physical inactivity**
        - **Excessive alcohol use**
    """)
    st.image("CVD_Infographics.png", caption="Stay Heart-Healthy ❤️", use_container_width=False, width=750)
    

def about_us():
    st.title("Discover Our Team")
    st.write(
        """
        As a team of passionate individuals, we embarked on a journey to create a 
        user-friendly and efficient application to predict **Heart Disease**.
        """
    )

    st.subheader("Team Members:")
    st.markdown("""
    - **Trabelsi Yassine**
    - **Medyouni Saber**  
    - **Bensalah Yassine**  
    - **Ouertani Fatma**
    - **Haouari Khalil**
    - **Mannai Molka**
    - **Chniti Yasser**  
    """)

    st.subheader("Guide:")
    st.markdown("""
    - **M.Jihene Hlel**  
    - Our mentor and guide, whose invaluable support and expertise have been instrumental in shaping this project.  
    """)

    st.write(
        """
        Throughout the development process, we have combined our diverse skills and knowledge 
        to deliver a robust and accurate heart disease prediction system.  
        We are committed to promoting health awareness and providing a valuable tool for 
        individuals to assess their health risks.
        """
    )

    st.success(
        "Thank you for choosing our CardioAlert App. "
        "We hope it proves to be a valuable resource for you and others."
    )


def chat_page():
    st.title("CardioAlert App 🚨")
    st.title("Chat Bot")
    user_input = {}
    user_input = st.text_input("Question", "")

    if user_input:
        response = get_gemini_response(user_input)
        st.write(f"Bot: {response}")

def prediction_page():
    st.title("CardioAlert App 🚨")
    st.header("Enter your informations")
    user_input = {}
    sex_mapping = {"Woman": 0, "Man": 1}
    user_input['sex'] = sex_mapping[st.selectbox("Sex", options=list(sex_mapping.keys()))]

    chest_pain_mapping = {
        "Typical angina": 0,
        "Atypical angina": 1,
        "Non-anginal pain": 2,
        "Asymptomatic": 3
    }
    user_input['cp'] = chest_pain_mapping[st.selectbox("Chest Pain", options=list(chest_pain_mapping.keys()))]

    fbs_mapping = {"False (≤120 mg/dL)": 0, "True (>120 mg/dL)": 1}
    user_input['fbs'] = fbs_mapping[st.selectbox("Fasting blood sugar level categorized as above 120 mg/dL", options=list(fbs_mapping.keys()))]

    resting_ecg_mapping = {
        "Normal": 0,
        "Having ST-T wave abnormality": 1,
        "Showing probable or definite left ventricular hypertrophy": 2
    }
    user_input['restecg'] = resting_ecg_mapping[st.selectbox("Resting electrocardiographic results", options=list(resting_ecg_mapping.keys()))]

    exng_mapping = {
        "no": 0,
        "yes": 1
    }
    user_input['exng'] = exng_mapping[st.selectbox("Exercise Induced Angina", options=list(exng_mapping.keys()))]

    slp_mapping = {
        "Upsloping": 0,
        "Flat": 1,
        "Downsloping": 2
    }
    user_input['slp'] = slp_mapping[st.selectbox("Slope of the peak exercise ST segment", options=list(slp_mapping.keys()))]

    caa_mapping = {
        0: 0,
        1: 1,
        2: 2,
        3: 3,
        4: 4
    }
    user_input['caa'] = int(caa_mapping[st.selectbox("Number of Major Vessels", options=list(caa_mapping.values()))])

    thall_mapping = {
        "Normal" : 0,
        "Fixed defect" : 1,
        "Reversible defect" : 2,
        "Not described" : 3
    }
    user_input['thall'] = thall_mapping[st.selectbox("Thalium Stress Test Result", options=list(thall_mapping.keys()))]
    
    for col in column_names:
        if col == "chol":  
            user_input[col] = st.number_input(f"{column_labels[col]} (en mg/dL)", value=0.0)
        elif col == "trtbps":  
            user_input[col] = st.number_input(f"{column_labels[col]} (en mmHg)", value=0.0)
        elif col != "sex" and col != "cp" and col != "fbs" and col != "restecg" and col != "exng" and col != "slp" and col != "caa" and col != "thall":  # On saute les champs déjà ajoutés
            user_input[col] = st.number_input(f"{column_labels[col]}", value=0.0)

    if st.button("Predict and recommend"):
        input_df = pd.DataFrame([user_input])
        input_df = input_df[column_names]

        input_scaled = scaler.transform(input_df)

        input_pca = pca.transform(input_scaled)

        cluster_label = kmeans.predict(input_pca)[0]
        cluster_name = cluster_names[cluster_label]

        prediction = svm_model.predict(input_df)[0]

        user_input['Prediction'] = prediction
        user_input['Cluster'] = cluster_name
        st.session_state.data = pd.concat(
            [st.session_state.data, pd.DataFrame([user_input])],
            ignore_index=True
        )
        st.write("### Résultat :")
        if prediction == 0:
            st.success(f"**Bonne nouvelle ! Aucune indication de risque de crise cardiaque détectée.** 😊\n\n{cluster_names[1]}\n\n**Recommendations :** {recommendation_msg[1]}")
        else:
            st.error(f"**Attention ! Il y a un risque d'avoir une crise cardiaque.** 🚨\n\n{cluster_names[0]}\n\n**Recommendations :** {recommendation_msg[0]}")

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


                # Mapping inversé pour récupérer les libellés à partir des valeurs numériques
        inverse_mappings = {
            "sex": {0: "Woman", 1: "Man"},
            "cp": {0: "Typical angina", 1: "Atypical angina", 2: "Non-anginal pain", 3: "Asymptomatic"},
            "fbs": {0: "False (≤120 mg/dL)", 1: "True (>120 mg/dL)"},
            "restecg": {0: "Normal", 1: "Having ST-T wave abnormality", 2: "Showing probable or definite left ventricular hypertrophy"},
            "exng": {0: "no", 1: "yes"},
            "slp": {0: "Upsloping", 1: "Flat", 2: "Downsloping"},
            "caa": {0: 0, 1: 1, 2: 2, 3: 3, 4: 4},
            "thall": {0: "Normal", 1: "Fixed defect", 2: "Reversible defect", 3: "Not described"},
        }

        # Créer le bilan des valeurs saisies par l'utilisateur avec correction pour les valeurs non numériques
        bilan = "Bilan de vos Informations Entrées :\n\n"
        for key, value in user_input.items():
            if key in column_labels:
                # Vérifier si une valeur existe dans le mapping inverse pour utiliser le libellé au lieu du chiffre
                if key in inverse_mappings:
                    display_value = inverse_mappings[key].get(value, value)  # Récupérer le libellé ou laisser tel quel
                else:
                    display_value = value  # Valeurs numériques directement
                bilan += f"{column_labels[key]} : {display_value}\n"

        # Ajouter les résultats de la prédiction
        resultat_prediction = "Aucun risque de crise cardiaque détecté 😊" if prediction == 0 else "Risque potentiel de crise cardiaque détecté 🚨"
        bilan += "\nRésultat de la Prédiction :\n"
        bilan += f"{resultat_prediction}\n"

        # Ajouter les recommandations
        bilan += "\nRecommandations :\n"
        bilan += f"{cluster_name}\n"

        # Envoi de l'e-mail après prédiction
        if send_email(
            st.session_state.user_email,
            "Votre Bilan CardioAlert 🚨",
            f"Bonjour {st.session_state.user_name},\n\n{bilan}\n\nPrenez soin de votre santé !\n\nCordialement,\nL'équipe CardioAlert."
        ):
            st.success("📧 Votre bilan, incluant les résultats et recommandations, a été envoyé avec succès à votre adresse e-mail !")
        else:
            st.error("Une erreur est survenue lors de l'envoi de l'e-mail.")

        st.write("### Comparaison avec les Valeurs Normales")

        # Plot pour le taux de cholestérol
        fig, ax = plt.subplots()
        ax.bar(['Utilisateur', 'Seuil Normal'], [user_input['chol'], chol_threshold], color=['orange', 'green'])
        ax.set_title("Taux de Cholestérol")
        ax.set_ylabel("Cholestérol (mg/dL)")
        st.pyplot(fig)

        # Interprétation pour le taux de cholestérol
        if user_input['chol'] > chol_threshold:
            st.write("⚠️ **Interprétation :** Votre taux de cholestérol est supérieur au seuil normal, "
                     "ce qui peut indiquer un risque accru de maladies cardiovasculaires. "
                     "Il est conseillé de réduire les aliments riches en graisses saturées et de pratiquer des exercices physiques.")
        else:
            st.write("✅ **Interprétation :** Votre taux de cholestérol est dans la norme. Continuez à maintenir un mode de vie sain.")

        # Plot pour la pression artérielle
        fig, ax = plt.subplots()
        ax.bar(['Utilisateur', 'Seuil Normal'], [user_input['trtbps'], bp_threshold], color=['red', 'green'])
        ax.set_title("Pression Artérielle")
        ax.set_ylabel("Pression Artérielle (mmHg)")
        st.pyplot(fig)

        # Interprétation pour la pression artérielle
        if user_input['trtbps'] > bp_threshold:
            st.write("⚠️ **Interprétation :** Votre pression artérielle est élevée. "
                     "Cela peut entraîner des complications comme l'hypertension. "
                     "Il est recommandé de limiter la consommation de sel et de consulter un professionnel de santé.")
        else:
            st.write("✅ **Interprétation :** Votre pression artérielle est dans la norme. Gardez de bonnes habitudes alimentaires et physiques.")

        # Plot pour la fréquence cardiaque maximale
        fig, ax = plt.subplots()
        ax.bar(['Utilisateur', 'Seuil Bas'], [user_input['thalachh'], thalachh_low], color=['blue', 'green'])
        ax.set_title("Fréquence Cardiaque Maximale")
        ax.set_ylabel("Fréquence Cardiaque (bpm)")
        st.pyplot(fig)

        # Interprétation pour la fréquence cardiaque maximale
        if user_input['thalachh'] < thalachh_low:
            st.write("⚠️ **Interprétation :** Votre fréquence cardiaque maximale est relativement faible. "
                     "Cela peut indiquer une capacité physique réduite ou un problème cardiaque potentiel. "
                     "Il est conseillé de consulter un professionnel de santé pour une évaluation approfondie.")
        else:
            st.write("✅ **Interprétation :** Votre fréquence cardiaque maximale est normale. Cela montre une bonne capacité physique.")


if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if st.session_state.logged_in:
    st.sidebar.image("heart_logo.png", caption="", use_container_width=False, width=200)
    st.sidebar.title("Main Menu :")
    if st.sidebar.button("Home"):
        st.session_state.page = "Home"
    if st.sidebar.button("Prediction & Recommandations"):
        st.session_state.page = "Prediction & Recommandations"    
    if st.sidebar.button("Chat Bot"):
        st.session_state.page = "Chat Bot"  
    if st.sidebar.button("About Us"):
        st.session_state.page = "About Us"        

    if "page" not in st.session_state:
        st.session_state.page = "Home"

    if st.session_state.page == "Home":
        home_page()
    elif st.session_state.page == "Prediction & Recommandations":
        prediction_page()
    elif st.session_state.page == "Chat Bot":
        chat_page()
    elif st.session_state.page == "About Us":
        about_us()  
else:
    login()
