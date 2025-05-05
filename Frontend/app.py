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

# Token d'API pour l'authentification
API_TOKEN = "7eFtEwvyrWwychOMQPeIVvQbrN3xFbJn_Pq1bEaRWTA"

st.title("📄 OCR - Documents d'Assurance")

# Type de document
doc_type = st.radio("Type de document :", ("Image", "PDF"))
file_type = "image" if doc_type == "Image" else "pdf"
file = st.file_uploader("Charger un document", type=["jpg", "jpeg", "png"] if file_type == "image" else ["pdf"])

# Affichage immédiat après upload
if file:
    file_bytes = file.read() # Lire une seule fois
    if file_type == "image":
        st.info("📄 Document Assurance chargé")
    else:
        st.info("📄 Document Constat chargé")

    # Lancer l'analyse
    if st.button("📤 Lancer l'analyse OCR"):
        with st.spinner("Traitement en cours..."):
            # Sauvegarde temporaire
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.name)[1]) as tmp:
                tmp.write(file_bytes)
                temp_path = tmp.name
            
            try:
                with open(temp_path, "rb") as f:
                    # Préparation des fichiers à envoyer
                    files = {"file": f}
                    
                    # Préparation des headers avec le token d'authentification
                    headers = {"Authorization": f"Bearer {API_TOKEN}"}
                    
                    # Envoi de la requête avec le token dans les headers
                    response = requests.post(
                        ENDPOINTS[file_type],
                        files=files,
                        headers=headers,
                        timeout=900
                    )
                
                if response.status_code == 200:
                    result = response.json()
                    st.success("✅ Traitement terminé")
                    
                    # Vérification du type de document pour l'affichage
                    if file_type == "image":
                        # Pour les images (format simple)
                        st.subheader("📋 Informations extraites")
                        
                        # Extraction des données
                        extracted_info = result.get("extracted_info", {})
                        date_debut = extracted_info.get("date_debut", "Non disponible")
                        date_fin = extracted_info.get("date_fin", "Non disponible")
                        numero_police = extracted_info.get("numero_police", "Non disponible")
                        
                        # Affichage formaté avec des puces
                        st.markdown(f"* **Date de début:** {date_debut}")
                        st.markdown(f"* **Date de fin:** {date_fin}")
                        st.markdown(f"* **Numéro de police:** {numero_police}")
                    
                    else:
                        # Pour les PDF (format constat)
                        
                        # Obtenir les données
                        en_tete = result.get("en_tete", {})
                        partie_A = result.get("partie_A", {})
                        partie_B = result.get("partie_B", {})
                        
                        # SECTION 1: Vue d'ensemble avec en-tête, parties A et B
                        
                        # En-tête
                        st.subheader("🗓️ Informations de l'accident")
                        st.markdown(f"* **Date de l'accident:** {en_tete.get('date_accident', 'Non disponible')}")
                        st.markdown(f"* **Heure de l'accident:** {en_tete.get('heure_accident', 'Non disponible')}")
                        st.markdown(f"* **Lieu de l'accident:** {en_tete.get('lieu_accident', 'Non disponible')}")
                        st.markdown(f"* **Précision sur le lieu:** {en_tete.get('lieu_accident_precis', 'Non disponible')}")
                            
                        # Partie A
                        st.subheader("🚗 DÉTAILS PARTIE A")

                        a_assurance = partie_A.get('assurance', {})
                        st.subheader("📝 Assurance")
                        st.markdown(f"* **Nom:** {a_assurance.get('nom', 'Non disponible')}")
                        st.markdown(f"* **Prénom:** {a_assurance.get('prenom', 'Non disponible')}")
                        st.markdown(f"* **Adresse:** {a_assurance.get('adresse', 'Non disponible')}")
                        st.markdown(f"* **N° Police:** {a_assurance.get('n_police', 'Non disponible')}")
                        st.markdown(f"* **N° Attestation:** {a_assurance.get('n_attestation', 'Non disponible')}")
                        st.markdown(f"* **Date de validité:** {a_assurance.get('date_validite', 'Non disponible')}")

                        a_conducteur = partie_A.get('conducteur', {})
                        st.subheader("👤 Conducteur")
                        st.markdown(f"* **Nom:** {a_conducteur.get('nom', 'Non disponible')}")
                        st.markdown(f"* **Prénom:** {a_conducteur.get('prenom', 'Non disponible')}")
                        st.markdown(f"* **Adresse:** {a_conducteur.get('adresse', 'Non disponible')}")
                        st.markdown(f"* **N° Permis:** {a_conducteur.get('n_permis', 'Non disponible')}")
                        st.markdown(f"* **Date de validité du permis:** {a_conducteur.get('date_validite_permis', 'Non disponible')}")

                        a_vehicule = partie_A.get('vehicule', {})
                        st.subheader("🚗 Véhicule")
                        st.markdown(f"* **Type de véhicule:** {a_vehicule.get('type_vehicule', 'Non disponible')}")
                        st.markdown(f"* **N° Immatriculation:** {a_vehicule.get('n_immatriculation', 'Non disponible')}")
                        st.markdown(f"* **Téléphone:** {a_vehicule.get('telephone', 'Non disponible')}")
                        st.markdown(f"* **Venant de:** {a_vehicule.get('venant_de', 'Non disponible')}")
                        st.markdown(f"* **Allant à:** {a_vehicule.get('allant_a', 'Non disponible')}")

                        st.subheader("💥 Dégâts apparents")
                        st.markdown(f"* **Description:** {partie_A.get('degat_apparence', {}).get('description', 'Non disponible')}")
                        
                        # partie B
                        st.subheader("🚙 DÉTAILS PARTIE B")

                        b_assurance = partie_B.get('assurance', {})
                        st.subheader("📝 Assurance")
                        st.markdown(f"* **Nom:** {b_assurance.get('nom', 'Non disponible')}")
                        st.markdown(f"* **Prénom:** {b_assurance.get('prenom', 'Non disponible')}")
                        st.markdown(f"* **Adresse:** {b_assurance.get('adresse', 'Non disponible')}")
                        st.markdown(f"* **N° Police:** {b_assurance.get('n_police', 'Non disponible')}")
                        st.markdown(f"* **N° Attestation:** {b_assurance.get('n_attestation', 'Non disponible')}")
                        st.markdown(f"* **Date de validité:** {b_assurance.get('date_validite', 'Non disponible')}")

                        b_conducteur = partie_B.get('conducteur', {})
                        st.subheader("👤 Conducteur")
                        st.markdown(f"* **Nom:** {b_conducteur.get('nom', 'Non disponible')}")
                        st.markdown(f"* **Prénom:** {b_conducteur.get('prenom', 'Non disponible')}")
                        st.markdown(f"* **Adresse:** {b_conducteur.get('adresse', 'Non disponible')}")
                        st.markdown(f"* **N° Permis:** {b_conducteur.get('n_permis', 'Non disponible')}")
                        st.markdown(f"* **Date de validité du permis:** {b_conducteur.get('date_validite_permis', 'Non disponible')}")

                        b_vehicule = partie_B.get('vehicule', {})
                        st.subheader("🚙 Véhicule")
                        st.markdown(f"* **Type de véhicule:** {b_vehicule.get('type_vehicule', 'Non disponible')}")
                        st.markdown(f"* **N° Immatriculation:** {b_vehicule.get('n_immatriculation', 'Non disponible')}")
                        st.markdown(f"* **Téléphone:** {b_vehicule.get('telephone', 'Non disponible')}")
                        st.markdown(f"* **Venant de:** {b_vehicule.get('venant_de', 'Non disponible')}")
                        st.markdown(f"* **Allant à:** {b_vehicule.get('allant_a', 'Non disponible')}")
                            
                        st.subheader("💥 Dégâts apparents")
                        st.markdown(f"* **Description:** {partie_B.get('degat_apparence', {}).get('description', 'Non disponible')}")
                else:
                    st.error(f"Erreur {response.status_code} : {response.text}")
            except Exception as e:
                st.error(f"Erreur de connexion : {e}")
            finally:
                # Nettoyage du fichier temporaire
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
else:
    st.info("📂 Veuillez charger un document pour commencer.")