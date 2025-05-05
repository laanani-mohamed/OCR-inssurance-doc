from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import csv
import torch
import os
import json
from datetime import datetime
import numpy as np
from PIL import Image, ImageDraw
 
class ConstatProcessor:
    """
    Classe pour traiter les PDFs de constats amiables
    """
    
    def __init__(self, use_gpu=True):
        """
        Initialise le processeur de constats
        
        Args:
            use_gpu: Utiliser le GPU si disponible
        """
        # Vérifier la mémoire GPU disponible avant de charger les modèles
        if use_gpu:
            self.use_gpu = self.usage_gpu()  # Correction: Appeler la méthode statique correctement
        else:
            self.use_gpu = False
        
        self.device = torch.device("cuda" if self.use_gpu else "cpu")
        print(f"Utilisation de l'appareil: {self.device}")
        
        # Charger le modèle TrOCR
        print("Chargement du modèle TrOCR...")
        self.processor = TrOCRProcessor.from_pretrained("microsoft/trocr-large-handwritten")
        self.trocr_model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-large-handwritten").to(self.device)
        print(f"Modèle chargé avec succès sur {self.device}!")
        
        # Définir toutes les parties à analyser
        self.parties = [
            # En-tête
            {
                "name": "Date_Accident",
                "x": 200, "y": 160, "w": 400, "h": 80,
                "line_height": 80,
                "mapping": [
                    {"structure_path": ["en_tete", "date_accident"], "ligne_num": 1, "description": "Date de l'accident"}
                ]
            },
            {
                "name": "Heure_Accident",
                "x": 590, "y": 160, "w": 330, "h": 80,
                "line_height": 80,
                "mapping": [
                    {"structure_path": ["en_tete", "heure_accident"], "ligne_num": 1, "description": "Heure de l'accident"}
                ]
            },
            {
                "name": "Lieu_Accident",
                "x": 930, "y": 160, "w": 400, "h": 80,
                "line_height": 80,
                "mapping": [
                    {"structure_path": ["en_tete", "lieu_accident"], "ligne_num": 1, "description": "Lieu de l'accident"}
                ]
            },
            {
                "name": "Lieu_Precis",
                "x": 140, "y": 220, "w": 1200, "h": 60,
                "line_height": 60,
                "mapping": [
                    {"structure_path": ["en_tete", "lieu_accident_precis"], "ligne_num": 1, "description": "Lieu précis de l'accident"}
                ]
            },
 
            # Partie A
            {
                "name": "Telephone_A",
                "x": 0, "y": 330, "w": 480, "h": 110,
                "line_height": 110,
                "mapping": [
                    {"structure_path": ["partie_A", "vehicule", "telephone"], "ligne_num": 1, "description": "Téléphone"}
                ]
            },
            {
                "name": "Vehicule_A",
                "x": 0, "y": 460, "w": 480, "h": 230,
                "line_height": 45,
                "mapping": [
                    {"structure_path": ["partie_A", "vehicule", "type_vehicule"], "ligne_num": 1, "description": "Type de véhicule"},
                    {"structure_path": ["partie_A", "vehicule", "n_immatriculation"], "ligne_num": 2, "description": "N° d'immatriculation"},
                    {"structure_path": ["partie_A", "vehicule", "venant_de"], "ligne_num": 3, "description": "Venant de"},
                    {"structure_path": ["partie_A", "vehicule", "allant_a"], "ligne_num": 4, "description": "Allant à"}
                ]
            },
            {
                "name": "Assure_A",
                "x": 0, "y": 690, "w": 480, "h": 335,
                "line_height": 50,
                "mapping": [
                    {"structure_path": ["partie_A", "assurance", "nom"], "ligne_num": 1, "description": "Nom"},
                    {"structure_path": ["partie_A", "assurance", "prenom"], "ligne_num": 2, "description": "Prénom"},
                    {"structure_path": ["partie_A", "assurance", "adresse"], "ligne_num": 3, "description": "Adresse"},
                    {"structure_path": ["partie_A", "assurance", "n_attestation"], "ligne_num": 4, "description": "N° d'attestation"},
                    {"structure_path": ["partie_A", "assurance", "n_police"], "ligne_num": 5, "description": "N° de police"},
                    {"structure_path": ["partie_A", "assurance", "date_validite"], "ligne_num": 6, "description": "Date de validité"}
                ]
            },
            {
                "name": "Conducteur_A",
                "x": 0, "y": 1050, "w": 480, "h": 300,
                "line_height": 45,
                "mapping": [
                    {"structure_path": ["partie_A", "conducteur", "nom"], "ligne_num": 1, "description": "Nom"},
                    {"structure_path": ["partie_A", "conducteur", "prenom"], "ligne_num": 2, "description": "Prénom"},
                    {"structure_path": ["partie_A", "conducteur", "adresse"], "ligne_num": 3, "description": "Adresse"},
                    {"structure_path": ["partie_A", "conducteur", "n_permis"], "ligne_num": 4, "description": "N° du permis"},
                    {"structure_path": ["partie_A", "conducteur", "date_validite_permis"], "ligne_num": 5, "description": "Date de validité du permis"}
                ]
            },
            {
                "name": "Degat_Observation_A",
                "x": 0, "y": 1550, "w": 480, "h": 300,
                "line_height": 300,
                "mapping": [
                    {"structure_path": ["partie_A", "degat_apparence", "description"], "ligne_num": 1, "description": "Description des dégâts apparents"}
                ]
            },
 
            # Partie B
            {
                "name": "Telephone_B",
                "x": 1020, "y": 330, "w": 480, "h": 110,
                "line_height": 110,
                "mapping": [
                    {"structure_path": ["partie_B", "vehicule", "telephone"], "ligne_num": 1, "description": "Téléphone"}
                ]
            },
            {
                "name": "Vehicule_B",
                "x": 1020, "y": 460, "w": 480, "h": 230,
                "line_height": 45,
                "mapping": [
                    {"structure_path": ["partie_B", "vehicule", "type_vehicule"], "ligne_num": 1, "description": "Type de véhicule"},
                    {"structure_path": ["partie_B", "vehicule", "n_immatriculation"], "ligne_num": 2, "description": "N° d'immatriculation"},
                    {"structure_path": ["partie_B", "vehicule", "venant_de"], "ligne_num": 3, "description": "Venant de"},
                    {"structure_path": ["partie_B", "vehicule", "allant_a"], "ligne_num": 4, "description": "Allant à"}
                ]
            },
            {
                "name": "Assure_B",
                "x": 1020, "y": 690, "w": 480, "h": 335,
                "line_height": 50,
                "mapping": [
                    {"structure_path": ["partie_B", "assurance", "nom"], "ligne_num": 1, "description": "Nom"},
                    {"structure_path": ["partie_B", "assurance", "prenom"], "ligne_num": 2, "description": "Prénom"},
                    {"structure_path": ["partie_B", "assurance", "adresse"], "ligne_num": 3, "description": "Adresse"},
                    {"structure_path": ["partie_B", "assurance", "n_attestation"], "ligne_num": 4, "description": "N° d'attestation"},
                    {"structure_path": ["partie_B", "assurance", "n_police"], "ligne_num": 5, "description": "N° de police"},
                    {"structure_path": ["partie_B", "assurance", "date_validite"], "ligne_num": 6, "description": "Date de validité"}
                ]
            },
            {
                "name": "Conducteur_B",
                "x": 1020, "y": 1000, "w": 480, "h": 340,
                "line_height": 50,
                "mapping": [
                    {"structure_path": ["partie_B", "conducteur", "nom"], "ligne_num": 1, "description": "Nom"},
                    {"structure_path": ["partie_B", "conducteur", "prenom"], "ligne_num": 2, "description": "Prénom"},
                    {"structure_path": ["partie_B", "conducteur", "adresse"], "ligne_num": 3, "description": "Adresse"},
                    {"structure_path": ["partie_B", "conducteur", "n_permis"], "ligne_num": 4, "description": "N° du permis"},
                    {"structure_path": ["partie_B", "conducteur", "date_validite_permis"], "ligne_num": 5, "description": "Date de validité du permis"}
                ]
            },
            {
                "name": "Degat_Observation_B",
                "x": 1020, "y": 1550, "w": 480, "h": 300,
                "line_height": 300,
                "mapping": [
                    {"structure_path": ["partie_B", "degat_apparence", "description"], "ligne_num": 1, "description": "Description des dégâts apparents"}
                ]
            }
        ]
 
    def check_gpu_memory(self, required_memory_mb=3000):
        """Vérifie s'il y a assez de mémoire GPU disponible pour charger les modèles.
        
        Args:
            required_memory_mb: Mémoire requise en Mo (par défaut 3000 Mo)
        
        Returns:
            bool: True si assez de mémoire GPU disponible, False sinon
        """
        if not torch.cuda.is_available():
            print("GPU non disponible, utilisation du CPU.")
            return False
        
        try:
            # Récupérer les informations de mémoire GPU
            free_memory = torch.cuda.mem_get_info()[0] / (1024 * 1024)  # En Mo
            print(f"Mémoire GPU libre: {free_memory:.2f} Mo")
            
            if free_memory < required_memory_mb:
                print(f"Mémoire GPU insuffisante ({free_memory:.2f} Mo libre, {required_memory_mb} Mo requis)")
                return False
            
            return True
        except Exception as e:
            print(f"Erreur lors de la vérification de la mémoire GPU: {str(e)}")
            return False
 

    @staticmethod  # Ajout du décorateur pour indiquer clairement que c'est une méthode statique
    def usage_gpu():
        """
        Vérifie si un GPU est disponible pour le traitement.
        Returns:
            bool: True si le GPU est disponible, False sinon.
        """
        import torch
        # Vérifier si CUDA (GPU) est disponible
        gpu_available = torch.cuda.is_available()
        if gpu_available:
            # Obtenir le nom du périphérique GPU pour le logging
            device_name = torch.cuda.get_device_name(0)
            print(f"GPU disponible: {device_name}")
            # Définir l'appareil par défaut comme le GPU
            device = torch.device("cuda")
            return True
        else:
            print("GPU non disponible. Utilisation du CPU à la place.")
            return False
        
 
    def init_structure_constat(self):
        """Structure de données pour le constat amiable"""
        return {
            "en_tete": {
                "date_accident": None,
                "heure_accident": None,
                "lieu_accident": None,
                "lieu_accident_precis": None
            },
            "partie_A": {
                "vehicule": {
                    "telephone": None,
                    "type_vehicule": None,
                    "n_immatriculation": None,
                    "venant_de": None,
                    "allant_a": None
                },
                "assurance": {
                    "nom": None,
                    "prenom": None,
                    "adresse": None,
                    "n_attestation": None,
                    "n_police": None,
                    "date_validite": None
                },
                "conducteur": {
                    "nom": None,
                    "prenom": None,
                    "adresse": None,
                    "n_permis": None,
                    "date_validite_permis": None
                },
                "degat_apparence": {
                    "description": None
                }
            },
            "partie_B": {
                "vehicule": {
                    "telephone": None,
                    "type_vehicule": None,
                    "n_immatriculation": None,
                    "venant_de": None,
                    "allant_a": None
                },
                "assurance": {
                    "nom": None,
                    "prenom": None,
                    "adresse": None,
                    "n_attestation": None,
                    "n_police": None,
                    "date_validite": None
                },
                "conducteur": {
                    "nom": None,
                    "prenom": None,
                    "adresse": None,
                    "n_permis": None,
                    "date_validite_permis": None
                },
                "degat_apparence": {
                    "description": None
                }
            }
        }
 
    def set_nested_dict_value(self, d, path, value):
        """
        Met à jour un dictionnaire imbriqué en suivant un chemin spécifié
        """
        current = d
        for key in path[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        current[path[-1]] = value
 
    def visualiser_roi(self, image, partie):
        """
        Visualise une ROI en dessinant un rectangle sur l'image
        """
        draw = ImageDraw.Draw(image)
        x, y, w, h = partie["x"], partie["y"], partie["w"], partie["h"]
        # Rectangle rouge avec une épaisseur de 2 pixels
        draw.rectangle([(x, y), (x + w, y + h)], outline="red", width=2)
        return image
 
    def extraire_et_analyser_partie(self, image, partie):
        """
        Extrait une partie de l'image et fait la reconnaissance de texte par ligne
        """
        try:
            # Extraire les paramètres
            x, y, w, h = partie["x"], partie["y"], partie["w"], partie["h"]
            line_height = partie["line_height"]
            name = partie["name"]
 
            # Vérifier si les coordonnées sont dans les limites
            img_width, img_height = image.size
            if x + w > img_width or y + h > img_height:
                print(f"Attention: La partie {name} dépasse les limites de l'image!")
                # Ajuster les coordonnées
                x = min(x, img_width - 1)
                y = min(y, img_height - 1)
                w = min(w, img_width - x)
                h = min(h, img_height - y)
 
            # Extraire la partie - création temporaire sans sauvegarde
            partie_image = image.crop((x, y, x + w, y + h))
 
            # S'assurer que l'image est en mode RGB
            if partie_image.mode != 'RGB':
                partie_image = partie_image.convert('RGB')
 
            # Découper la partie en lignes selon line_height
            num_lines = max(1, int(h / line_height))
            lignes_resultats = []
 
            print(f"Analyse de {num_lines} lignes pour la partie {name}:")
 
            # Parcourir chaque ligne
            for i in range(num_lines):
                # Calculer les coordonnées de la ligne
                ligne_y = i * line_height
                ligne_h = min(line_height, h - ligne_y)
 
                if ligne_h <= 0:
                    break
 
                # Extraire la ligne - temporaire
                ligne_image = partie_image.crop((0, ligne_y, w, ligne_y + ligne_h))
 
                # S'assurer que la ligne est en mode RGB
                if ligne_image.mode != 'RGB':
                    ligne_image = ligne_image.convert('RGB')
 
                # Reconnaissance de texte avec TrOCR
                pixel_values = self.processor(images=ligne_image, return_tensors="pt").pixel_values.to(self.device)
                generated_ids = self.trocr_model.generate(pixel_values)
                generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
 
                # Nettoyer un peu le texte
                generated_text = generated_text.strip()
 
                # Ajouter le résultat (uniquement le texte, pas l'image)
                ligne_info = {
                    "partie": name,
                    "ligne_num": i+1,
                    "texte": generated_text
                }
                lignes_resultats.append(ligne_info)
 
                print(f"  Ligne {i+1}: {generated_text}")
 
            # Ne retourne pas l'image, seulement les données textuelles
            return {
                "partie_name": name,
                "lignes": lignes_resultats
            }
 
        except Exception as e:
            print(f"Erreur lors de l'analyse de la partie {partie['name']}: {str(e)}")
            return None
 
    def traiter_image_complete(self, image_path=None, image_redimensionnee=None):
        """
        Traite une image complète avec toutes les ROI définies.
        Peut accepter soit un chemin d'image, soit directement une image déjà redimensionnée
        """
        try:
            # Charger l'image - soit à partir du chemin, soit utiliser celle fournie
            if image_path and not image_redimensionnee:
                print(f"Chargement de l'image: {image_path}")
                image = Image.open(image_path)
            elif image_redimensionnee is not None:
                print("Utilisation de l'image redimensionnée fournie")
                image = image_redimensionnee
            else:
                raise ValueError("Vous devez fournir soit un chemin d'image, soit une image redimensionnée")
 
            # Initialiser la structure de données pour les résultats
            structure_constat = self.init_structure_constat()
            resultats_parties = []
 
            # Créer uniquement un dossier pour le JSON final
            output_dir = "resultats_ocr"
            os.makedirs(output_dir, exist_ok=True)
 
            print(f"\n=== TRAITEMENT DE L'IMAGE ===\n")
 
            # Traiter chaque partie
            for partie in self.parties:
                print(f"\nTraitement de la partie: {partie['name']}...")
 
                # Analyser la partie
                resultat = self.extraire_et_analyser_partie(image, partie)
 
                if resultat:
                    resultats_parties.append(resultat)
 
                    # Mettre à jour la structure constat
                    for ligne in resultat["lignes"]:
                        ligne_num = ligne["ligne_num"]
                        texte = ligne["texte"]
 
                        # Trouver le mapping correspondant
                        for mapping_item in partie.get("mapping", []):
                            if mapping_item["ligne_num"] == ligne_num:
                                structure_path = mapping_item["structure_path"]
                                self.set_nested_dict_value(structure_constat, structure_path, texte)
 
            # Uniquement pour le retour de la fonction, ne pas sauvegarder de fichiers intermédiaires
            print(f"\nTraitement terminé!")
 
            # Afficher directement le JSON dans la console
            print("\n=== RÉSULTATS AU FORMAT JSON ===")
            print(json.dumps(structure_constat, ensure_ascii=False, indent=2))
 
            # Retourner seulement la structure du constat
            return structure_constat
 
        except Exception as e:
            print(f"Erreur lors du traitement de l'image: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
 
    def process_pdf(self, pdf_path):
        """
        Méthode principale pour traiter un PDF de constat amiable
        """
        try:
            # Vérifier que le PDF existe
            if not os.path.exists(pdf_path):
                print(f"ERREUR: Le PDF {pdf_path} n'existe pas!")
                return None
 
            # Chargement du document avec PyMuPDF pour préserver la qualité
            print("Chargement du PDF...")
            import fitz  # PyMuPDF
            doc = fitz.open(pdf_path)
            page = doc.load_page(0)  # Première page
 
            # Obtenir un rendu haute qualité de la page
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # Facteur de zoom 2 pour plus de qualité
            img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
 
            # Si l'image est en CMYK (4 canaux) ou a un canal alpha (4 canaux)
            if pix.n == 4:
                # Convertir en RGB si nécessaire
                if pix.colorspace == fitz.csRGB:  # RGB+alpha
                    # Ignorer le canal alpha
                    img_array = img_array[:, :, :3]
                else:  # CMYK
                    img_rgb = Image.frombytes("CMYK", [pix.w, pix.h], pix.samples)
                    img_rgb = img_rgb.convert("RGB")
                    img_array = np.array(img_rgb)
 
            # Appliquer la détection de texte avec doctr
            from doctr.io import DocumentFile
            from doctr.models import ocr_predictor, fast_tiny
 
            doc_for_ocr = DocumentFile.from_pdf(pdf_path)
            first_page_for_ocr = doc_for_ocr[0]
            first_page_only = [first_page_for_ocr]
 
            # Initialiser le modèle OCR pré-entraîné
            det_model = fast_tiny(pretrained=True).to(self.device)
            predictor = ocr_predictor(det_arch=det_model).to(self.device)
 
            # Exécuter l'OCR sur la première page
            result = predictor(first_page_only)
 
            # Extraire les boîtes
            def extract_boxes_only(ocr_result):
                boxes = []
 
                for page in ocr_result.pages:
                    for block in page.blocks:
                        for line in block.lines:
                            for word in line.words:
                                # Obtenir les coordonnées
                                box = word.geometry
 
                                # Format attendu: ((x1, y1), (x2, y2))
                                if isinstance(box, tuple) and len(box) == 2:
                                    (x1, y1), (x2, y2) = box
                                    boxes.append([x1, y1, x2, y2])
 
                # Créer un DataFrame avec les coordonnées
                import pandas as pd
                df = pd.DataFrame({
                    'x1': [box[0] for box in boxes],
                    'y1': [box[1] for box in boxes],
                    'x2': [box[2] for box in boxes],
                    'y2': [box[3] for box in boxes]
                })
 
                return df
 
            # Extraire seulement les boîtes
            boxes_df = extract_boxes_only(result)
 
            # Calculer la boîte englobante
            def calculer_boite_finale(boxes_df):
                if len(boxes_df) == 0:
                    return None
 
                # Trouver les coordonnées min/max pour englober toutes les boîtes
                x1_final = boxes_df['x1'].min()
                y1_final = boxes_df['y1'].min()
                x2_final = boxes_df['x2'].max()
                y2_final = boxes_df['y2'].max()
 
                return {
                    'x1': x1_final,
                    'y1': y1_final,
                    'x2': x2_final,
                    'y2': y2_final
                }
 
            # Définir les coordonnées de la boîte englobante
            coords_boite_englobante = calculer_boite_finale(boxes_df)
 
            # Extraire la zone englobante
            def extraire_zone_normalisee(image_array, coords, margin=0):
                # Vérifier si l'image est déjà normalisée (0-255) ou non (0-1)
                if np.issubdtype(image_array.dtype, np.floating) and image_array.max() <= 1.0:
                    # Si l'image a des valeurs entre 0 et 1, la normaliser correctement
                    image_array = (image_array * 255).astype(np.uint8)
                elif not np.issubdtype(image_array.dtype, np.uint8):
                    # Si ce n'est pas déjà uint8, convertir
                    image_array = image_array.astype(np.uint8)
 
                # Convertir l'image NumPy en image PIL
                image_pil = Image.fromarray(image_array)
 
                # S'assurer que l'image est en mode RGB
                if image_pil.mode != 'RGB':
                    image_pil = image_pil.convert('RGB')
 
                img_width, img_height = image_pil.size
 
                # Convertir les coordonnées normalisées en pixels avec marge
                x1 = max(0, int(coords['x1'] * img_width) - margin)
                y1 = max(0, int(coords['y1'] * img_height) - margin)
                x2 = min(img_width, int(coords['x2'] * img_width) + margin)
                y2 = min(img_height, int(coords['y2'] * img_height) + margin)
 
                # Extraire la région
                region = image_pil.crop((x1, y1, x2, y2))
 
                return region, (x1, y1, x2, y2)
 
            # Extraire la zone englobante
            first_page = img_array  # Utiliser l'image obtenue de PyMuPDF
            zone_englobante, _ = extraire_zone_normalisee(
                first_page, coords_boite_englobante, margin=0)
 
            # Définir une taille cible uniforme pour la standardisation
            TARGET_WIDTH = 1500  # Largeur cible en pixels
            TARGET_HEIGHT = 2000  # Hauteur cible en pixels
 
            # Redimensionner la zone englobante à la taille cible
            zone_redimensionnee = zone_englobante.resize((TARGET_WIDTH, TARGET_HEIGHT), Image.LANCZOS)
 
            # Traiter l'image redimensionnée
            structure_constat = self.traiter_image_complete(
                image_redimensionnee=zone_redimensionnee
            )
 
            # Retourner uniquement le JSON
            if structure_constat:
                # Afficher le JSON proprement formaté
                json_data = json.dumps(structure_constat, ensure_ascii=False, indent=4)
                print("\n=== RÉSULTAT FINAL ===")
                print(json_data)
                
                # Uniquement afficher le JSON, pas de sauvegarde locale
                print("Traitement terminé. JSON prêt à être utilisé.")
                
                # Retourner le dictionnaire
                return structure_constat
            else:
                print("Le traitement n'a pas pu être complété en raison d'erreurs.")
                return None
                
        except Exception as e:
            print(f"Erreur lors du traitement du PDF: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
 
 
if __name__ == "__main__":
    # Test simple de la classe
    processor = ConstatProcessor(use_gpu=True)