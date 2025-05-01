import streamlit as st
import requests
import tempfile
import os
import pandas as pd

st.set_page_config(page_title="OCR Assurance", page_icon="📄", layout="wide")

# Backend config
BACKEND_URL = "http://127.0.0.1:5090"
ENDPOINTS = {
    "image": f"{BACKEND_URL}/ocr/image",
    "pdf": f"{BACKEND_URL}/ocr/pdf"
}

st.title("📄 OCR - Documents d'Assurance")

# Type de document
doc_type = st.radio("Type de document :", ("Image", "PDF"))
file_type = "image" if doc_type == "Image" else "pdf"
file = st.file_uploader("Charger un document", type=["jpg", "jpeg", "png"] if file_type == "image" else ["pdf"])

if file:
    if file_type == "image":
        st.image(file, caption="Aperçu du document", use_column_width=True)
    else:
        st.info("📄 Document PDF chargé")

    if st.button("📤 Lancer l'analyse OCR"):
        with st.spinner("Traitement en cours..."):
            # Sauvegarde temporaire
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.name)[1]) as tmp:
                tmp.write(file.read())
                temp_path = tmp.name

            try:
                with open(temp_path, "rb") as f:
                    files = {"file": f}
                    response = requests.post(ENDPOINTS[file_type], files=files, timeout=600)

                if response.status_code == 200:
                    result = response.json()
                    st.success("✅ Traitement terminé")

                    # Affichage brut
                    st.subheader("🧾 Résultat JSON")
                    st.json(result)

                    # Tentative d'affichage tabulaire
                    st.subheader("📊 Données tabulaires")
                    for key, value in result.items():
                        if isinstance(value, list) and value and isinstance(value[0], dict):
                            st.write(f"**{key.replace('_', ' ').title()}**")
                            df = pd.DataFrame(value)
                            st.dataframe(df, use_container_width=True)
                        elif isinstance(value, dict):
                            df = pd.DataFrame([value]).T.reset_index()
                            df.columns = ["Champ", "Valeur"]
                            st.write(f"**{key.replace('_', ' ').title()}**")
                            st.dataframe(df, use_container_width=True)

                else:
                    st.error(f"Erreur {response.status_code} : {response.text}")

            except Exception as e:
                st.error(f"Erreur de connexion : {e}")
            finally:
                os.unlink(temp_path)
else:
    st.info("📂 Veuillez charger un document pour commencer.")
