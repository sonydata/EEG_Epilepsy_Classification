import streamlit as st

# Barre latérale pour la navigation
st.sidebar.title("Navigation")
options = st.sidebar.radio("Choisissez une section :", ["Accueil", "EDA", "Modèle"])

# Section Accueil
if options == "Accueil":
    st.title("Bienvenue sur le Tableau de Bord")
    st.write("Ce tableau de bord est conçu pour intégrer des modèles et des analyses exploratoires.")

# Section EDA
elif options == "EDA":
    st.title("Analyse Exploratoire des Données")
    uploaded_file = st.file_uploader("Téléchargez un fichier CSV", type="csv")
    if uploaded_file:
        st.write("Données téléchargées. Placeholders pour graphiques à venir.")

# Section Modèle
elif options == "Modèle":
    st.title("Tester un Modèle ML")
    model_name = st.text_input("Nom du modèle Hugging Face", "bert-base-uncased")
    st.write("Ajout des fonctionnalités pour les prédictions à venir.")
