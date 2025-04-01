from transformers import pipeline

def load_model(task="text-classification", model_name="bert-base-uncased"):
    return pipeline(task, model=model_name)

def predict(text, model):
    return model(text)

from models.inference import load_model, predict

if options == "Modèle":
    st.title("Tester un Modèle ML")
    model_name = st.text_input("Nom du modèle Hugging Face :", "bert-base-uncased")
    if st.button("Charger le Modèle"):
        model = load_model(model_name=model_name)
        st.success(f"Modèle {model_name} chargé avec succès.")
        
    # Placeholder pour faire des prédictions
    text = st.text_area("Entrez un texte pour prédiction :", "")
    if st.button("Prédire"):
        result = predict(text, model)
        st.write("Résultat :", result)
