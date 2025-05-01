import streamlit as st
import requests
import json
import os
import time
from PIL import Image
import io

# Set page configuration
st.set_page_config(
    page_title="OCR for Insurance Documents",
    page_icon="📄",
    layout="wide"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-title {
        font-size: 3rem !important;
        font-weight: 700;
        text-align: center;
        margin-bottom: 2rem;
        color: #1E3A8A;
    }
    .card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    .info-header {
        background-color: #1E3A8A;
        color: white;
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 15px;
    }
    .button-style {
        background-color: #1E3A8A;
        color: white;
        border-radius: 5px;
        padding: 10px 20px;
    }
    .success-message {
        background-color: #d1e7dd;
        color: #0f5132;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .error-message {
        background-color: #f8d7da;
        color: #842029;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# App title
st.markdown("<h1 class='main-title'>OCR for Insurance Documents</h1>", unsafe_allow_html=True)

# Initialize session states if they don't exist
if 'file_uploaded' not in st.session_state:
    st.session_state.file_uploaded = False
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False
if 'document_type' not in st.session_state:
    st.session_state.document_type = None
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'processing' not in st.session_state:
    st.session_state.processing = False
if 'upload_time' not in st.session_state:
    st.session_state.upload_time = None

# Function to reset the app state
def reset_app():
    st.session_state.file_uploaded = False
    st.session_state.analysis_complete = False
    st.session_state.document_type = None
    st.session_state.analysis_results = None
    st.session_state.processing = False
    st.session_state.upload_time = None

# Display document type selection
col1, col2 = st.columns(2)

with col1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    if st.button("Constat Document (PDF)", use_container_width=True):
        st.session_state.document_type = "constat"
        reset_app()
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    if st.button("Insurance Document (Image)", use_container_width=True):
        st.session_state.document_type = "insurance"
        reset_app()
    st.markdown("</div>", unsafe_allow_html=True)

# Display selected document type
if st.session_state.document_type:
    st.markdown(f"<div class='success-message'>Selected document type: {st.session_state.document_type.upper()}</div>", unsafe_allow_html=True)

    # File uploader section
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='info-header'>Upload Document</div>", unsafe_allow_html=True)
    
    if st.session_state.document_type == "constat":
        uploaded_file = st.file_uploader("Upload Constat PDF", type=["pdf"])
    else:
        uploaded_file = st.file_uploader("Upload Insurance Document Image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None and not st.session_state.file_uploaded:
        st.session_state.file_uploaded = True
        st.session_state.upload_time = time.time()
        
        # Display preview of the uploaded file
        if st.session_state.document_type == "insurance":
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Insurance Document", width=400)
        else:
            st.markdown(f"Uploaded PDF: {uploaded_file.name}")
    
    st.markdown("</div>", unsafe_allow_html=True)

    # Analyze button
    if st.session_state.file_uploaded and not st.session_state.processing and not st.session_state.analysis_complete:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        if st.button("Analyze Document", key="analyze_btn"):
            st.session_state.processing = True
            
            # Send the document to the backend for processing
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # Save the uploaded file temporarily
                file_bytes = uploaded_file.getvalue()
                file_type = uploaded_file.type
                file_name = uploaded_file.name
                
                # Configure API endpoint based on document type
                if st.session_state.document_type == "insurance":
                    api_endpoint = "http://localhost:5000/api/process_insurance"
                else:
                    api_endpoint = "http://localhost:5000/api/process_constat"
                
                # For demo purposes, we'll still simulate but show the API connection code (commented)
                """
                # Create a multipart form-data request
                files = {'file': (file_name, file_bytes, file_type)}
                
                # Send request to backend
                response = requests.post(api_endpoint, files=files)
                
                if response.status_code == 200:
                    st.session_state.analysis_results = response.json()
                else:
                    st.error(f"Error processing document: {response.text}")
                    st.session_state.processing = False
                    return
                """
                
                # Simulate processing with progress bar
                for i in range(101):
                    progress_bar.progress(i)
                    status_text.text(f"Processing... {i}% complete")
                    time.sleep(0.05)
                
                # Simulate API response based on document type
                if st.session_state.document_type == "insurance":
                    # Sample insurance document response
                    st.session_state.analysis_results = {
                        "extracted_info": {
                            "numero_police": "123456789",
                            "date_debut": "30 octobre 2024",
                            "date_fin": "29 janvier 2025"
                        },
                        "execution_time": {
                            "seconds": 53.37154936790466,
                            "device": "CPU"
                        }
                    }
                else:
                    st.session_state.analysis_results = {
                        "en_tete": {
                            "date_accident": "15/04/2025",
                            "heure_accident": "14:30",
                            "lieu_accident": "Paris",
                            "lieu_accident_precis": "Intersection Avenue des Champs-Élysées et Rue de Rivoli"
                        },
                        "partie_A": {
                            "vehicule": {
                                "telephone": "+33612345678",
                                "type_vehicule": "Citroën C3",
                                "n_immatriculation": "AB-123-CD",
                                "venant_de": "Neuilly-sur-Seine",
                                "allant_a": "Paris Centre"
                            },
                            "assurance": {
                                "nom": "Dubois",
                                "prenom": "Jean",
                                "adresse": "123 Rue de Paris, 75001 Paris",
                                "n_attestation": "ATT12345",
                                "n_police": "POL987654",
                                "date_validite": "31/12/2025"
                            },
                            "conducteur": {
                                "nom": "Dubois",
                                "prenom": "Jean",
                                "adresse": "123 Rue de Paris, 75001 Paris",
                                "n_permis": "P12345678",
                                "date_validite_permis": "15/06/2030"
                            },
                            "degat_apparence": {
                                "description": "Pare-choc avant enfoncé, phare droit cassé"
                            }
                        },
                        "partie_B": {
                            "vehicule": {
                                "telephone": "+33698765432",
                                "type_vehicule": "Renault Clio",
                                "n_immatriculation": "EF-456-GH",
                                "venant_de": "Paris Centre",
                                "allant_a": "Neuilly-sur-Seine"
                            },
                            "assurance": {
                                "nom": "Martin",
                                "prenom": "Sophie",
                                "adresse": "456 Avenue Victor Hugo, 75016 Paris",
                                "n_attestation": "ATT67890",
                                "n_police": "POL123456",
                                "date_validite": "15/05/2026"
                            },
                            "conducteur": {
                                "nom": "Martin",
                                "prenom": "Sophie",
                                "adresse": "456 Avenue Victor Hugo, 75016 Paris",
                                "n_permis": "P87654321",
                                "date_validite_permis": "20/08/2028"
                            },
                            "degat_apparence": {
                                "description": "Porte conducteur enfoncée, rétroviseur gauche cassé"
                            }
                        }
                    }
                st.session_state.processing = False
                st.session_state.analysis_complete = True
                st.experimental_rerun()

            except Exception as e:
                st.session_state.processing = False
                st.error(f"Une erreur est survenue pendant l'analyse : {str(e)}")

    # Display analysis results
    if st.session_state.analysis_complete:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='info-header'>Analysis Results</div>", unsafe_allow_html=True)
        
        # Calculate processing time
        processing_time = time.time() - st.session_state.upload_time
        st.markdown(f"<p>Document processed in {processing_time:.2f} seconds</p>", unsafe_allow_html=True)
        
        if st.session_state.document_type == "insurance":
            # Display insurance document results
            extracted_info = st.session_state.analysis_results["extracted_info"]
            execution_time = st.session_state.analysis_results["execution_time"]
            
            st.markdown("<h3>Extracted Information</h3>", unsafe_allow_html=True)
            st.markdown(f"<p><strong>Numéro de Police:</strong> {extracted_info['numero_police']}</p>", unsafe_allow_html=True)
            st.markdown(f"<p><strong>Date de Début:</strong> {extracted_info['date_debut']}</p>", unsafe_allow_html=True)
            st.markdown(f"<p><strong>Date de Fin:</strong> {extracted_info['date_fin']}</p>", unsafe_allow_html=True)
            
            st.markdown("<h3>Execution Details</h3>", unsafe_allow_html=True)
            st.markdown(f"<p><strong>Processing Time:</strong> {execution_time['seconds']:.2f} seconds</p>", unsafe_allow_html=True)
            st.markdown(f"<p><strong>Processing Device:</strong> {execution_time['device']}</p>", unsafe_allow_html=True)
        
        else:
            # Display constat document results
            constat_data = st.session_state.analysis_results
            
            # En-tête (Header) Information
            st.markdown("<h3>Informations Générales</h3>", unsafe_allow_html=True)
            en_tete = constat_data["en_tete"]
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"<p><strong>Date d'accident:</strong> {en_tete['date_accident']}</p>", unsafe_allow_html=True)
                st.markdown(f"<p><strong>Heure d'accident:</strong> {en_tete['heure_accident']}</p>", unsafe_allow_html=True)
            with col2:
                st.markdown(f"<p><strong>Lieu d'accident:</strong> {en_tete['lieu_accident']}</p>", unsafe_allow_html=True)
                st.markdown(f"<p><strong>Précision du lieu:</strong> {en_tete['lieu_accident_precis']}</p>", unsafe_allow_html=True)
            
            # Create tabs for Vehicle A and Vehicle B
            tab1, tab2 = st.tabs(["Véhicule A", "Véhicule B"])
            
            # Vehicle A Information
            with tab1:
                partie_a = constat_data["partie_A"]
                
                st.markdown("<h4>Informations du Véhicule</h4>", unsafe_allow_html=True)
                vehicule_a = partie_a["vehicule"]
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"<p><strong>Téléphone:</strong> {vehicule_a['telephone']}</p>", unsafe_allow_html=True)
                    st.markdown(f"<p><strong>Type de véhicule:</strong> {vehicule_a['type_vehicule']}</p>", unsafe_allow_html=True)
                    st.markdown(f"<p><strong>N° Immatriculation:</strong> {vehicule_a['n_immatriculation']}</p>", unsafe_allow_html=True)
                with col2:
                    st.markdown(f"<p><strong>Venant de:</strong> {vehicule_a['venant_de']}</p>", unsafe_allow_html=True)
                    st.markdown(f"<p><strong>Allant à:</strong> {vehicule_a['allant_a']}</p>", unsafe_allow_html=True)
                
                st.markdown("<h4>Informations d'Assurance</h4>", unsafe_allow_html=True)
                assurance_a = partie_a["assurance"]
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"<p><strong>Nom:</strong> {assurance_a['nom']}</p>", unsafe_allow_html=True)
                    st.markdown(f"<p><strong>Prénom:</strong> {assurance_a['prenom']}</p>", unsafe_allow_html=True)
                    st.markdown(f"<p><strong>Adresse:</strong> {assurance_a['adresse']}</p>", unsafe_allow_html=True)
                with col2:
                    st.markdown(f"<p><strong>N° Attestation:</strong> {assurance_a['n_attestation']}</p>", unsafe_allow_html=True)
                    st.markdown(f"<p><strong>N° Police:</strong> {assurance_a['n_police']}</p>", unsafe_allow_html=True)
                    st.markdown(f"<p><strong>Date de validité:</strong> {assurance_a['date_validite']}</p>", unsafe_allow_html=True)
                
                st.markdown("<h4>Informations du Conducteur</h4>", unsafe_allow_html=True)
                conducteur_a = partie_a["conducteur"]
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"<p><strong>Nom:</strong> {conducteur_a['nom']}</p>", unsafe_allow_html=True)
                    st.markdown(f"<p><strong>Prénom:</strong> {conducteur_a['prenom']}</p>", unsafe_allow_html=True)
                    st.markdown(f"<p><strong>Adresse:</strong> {conducteur_a['adresse']}</p>", unsafe_allow_html=True)
                with col2:
                    st.markdown(f"<p><strong>N° Permis:</strong> {conducteur_a['n_permis']}</p>", unsafe_allow_html=True)
                    st.markdown(f"<p><strong>Date de validité du permis:</strong> {conducteur_a['date_validite_permis']}</p>", unsafe_allow_html=True)
                
                st.markdown("<h4>Dégâts Apparents</h4>", unsafe_allow_html=True)
                st.markdown(f"<p><strong>Description:</strong> {partie_a['degat_apparence']['description']}</p>", unsafe_allow_html=True)
            
            # Vehicle B Information
            with tab2:
                partie_b = constat_data["partie_B"]
                
                st.markdown("<h4>Informations du Véhicule</h4>", unsafe_allow_html=True)
                vehicule_b = partie_b["vehicule"]
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"<p><strong>Téléphone:</strong> {vehicule_b['telephone']}</p>", unsafe_allow_html=True)
                    st.markdown(f"<p><strong>Type de véhicule:</strong> {vehicule_b['type_vehicule']}</p>", unsafe_allow_html=True)
                    st.markdown(f"<p><strong>N° Immatriculation:</strong> {vehicule_b['n_immatriculation']}</p>", unsafe_allow_html=True)
                with col2:
                    st.markdown(f"<p><strong>Venant de:</strong> {vehicule_b['venant_de']}</p>", unsafe_allow_html=True)
                    st.markdown(f"<p><strong>Allant à:</strong> {vehicule_b['allant_a']}</p>", unsafe_allow_html=True)
                
                st.markdown("<h4>Informations d'Assurance</h4>", unsafe_allow_html=True)
                assurance_b = partie_b["assurance"]
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"<p><strong>Nom:</strong> {assurance_b['nom']}</p>", unsafe_allow_html=True)
                    st.markdown(f"<p><strong>Prénom:</strong> {assurance_b['prenom']}</p>", unsafe_allow_html=True)
                    st.markdown(f"<p><strong>Adresse:</strong> {assurance_b['adresse']}</p>", unsafe_allow_html=True)
                with col2:
                    st.markdown(f"<p><strong>N° Attestation:</strong> {assurance_b['n_attestation']}</p>", unsafe_allow_html=True)
                    st.markdown(f"<p><strong>N° Police:</strong> {assurance_b['n_police']}</p>", unsafe_allow_html=True)
                    st.markdown(f"<p><strong>Date de validité:</strong> {assurance_b['date_validite']}</p>", unsafe_allow_html=True)
                
                st.markdown("<h4>Informations du Conducteur</h4>", unsafe_allow_html=True)
                conducteur_b = partie_b["conducteur"]
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"<p><strong>Nom:</strong> {conducteur_b['nom']}</p>", unsafe_allow_html=True)
                    st.markdown(f"<p><strong>Prénom:</strong> {conducteur_b['prenom']}</p>", unsafe_allow_html=True)
                    st.markdown(f"<p><strong>Adresse:</strong> {conducteur_b['adresse']}</p>", unsafe_allow_html=True)
                with col2:
                    st.markdown(f"<p><strong>N° Permis:</strong> {conducteur_b['n_permis']}</p>", unsafe_allow_html=True)
                    st.markdown(f"<p><strong>Date de validité du permis:</strong> {conducteur_b['date_validite_permis']}</p>", unsafe_allow_html=True)
                
                st.markdown("<h4>Dégâts Apparents</h4>", unsafe_allow_html=True)
                st.markdown(f"<p><strong>Description:</strong> {partie_b['degat_apparence']['description']}</p>", unsafe_allow_html=True)

        # Reset button
        if st.button("Process Another Document", key="reset_btn"):
            reset_app()
            st.experimental_rerun()
            
        st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown("""
<div style="text-align: center; margin-top: 50px; padding: 20px; background-color: #f8f9fa;">
    <p>© 2025 OCR Insurance Document Processor</p>
</div>
""", unsafe_allow_html=True)