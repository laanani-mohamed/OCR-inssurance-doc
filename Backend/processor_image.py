from PIL import Image
import io
import re
from Levenshtein import distance as levenshtein_distance
from paddleocr import PaddleOCR
import paddle
import os
 
def check_gpu_memory(required_memory_mb=1500):
    """Vérifie s'il y a assez de mémoire GPU disponible pour PaddleOCR."""
    if not paddle.device.is_compiled_with_cuda():
        return False
    
    try:
        import torch  # Utilisé uniquement pour vérifier la mémoire
        free_memory = torch.cuda.mem_get_info()[0] / (1024 * 1024)  # En Mo
        print(f"Mémoire GPU libre: {free_memory:.2f} Mo")
        
        if free_memory < required_memory_mb:
            print(f"Mémoire GPU insuffisante pour PaddleOCR ({free_memory:.2f} Mo libre, {required_memory_mb} Mo requis)")
            return False
        
        return True
    except Exception as e:
        print(f"Erreur lors de la vérification de la mémoire GPU: {str(e)}")
        return False

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


class ImageProcessor:
    def __init__(self, use_gpu=True):  # Ajout du paramètre use_gpu
        # Vérifier s'il y a assez de mémoire GPU uniquement si use_gpu est True
        if use_gpu:
            self.use_gpu = usage_gpu()
        else:
            self.use_gpu = False
            
        if not self.use_gpu:
            print("Utilisation du CPU pour PaddleOCR")
            
        # Initialiser PaddleOCR avec le bon périphérique
        self.ocr = PaddleOCR(use_angle_cls=True, lang='fr', use_gpu=self.use_gpu)
        
        # Dictionnaire de mots corrects
        self.correct_words = [
            "de", "la", "janvier", "février", "mars", "avril", "mai", "juin", "juillet", "août", "septembre",
            "octobre", "novembre", "décembre", "Numéro"
        ]
        
        # Motifs regex pour les dates
        self.day_pattern = r"(lun\.|mar\.|mer\.|jeu\.|ven\.|sam\.|dim\.|[0-9]{1,2})"
        self.month_pattern = r"(janvier|février|mars|avril|mai|juin|juillet|août|septembre|octobre|novembre|décembre)"
        self.year_pattern = r"(20[2-3][0-9])"
        self.date_pattern = re.compile(rf"{self.day_pattern}\s+{self.month_pattern}\s+{self.year_pattern}", re.IGNORECASE)
 
    def correct_word(self, word):
        """Corriger un mot en utilisant la distance de Levenshtein"""
        min_distance = float('inf')
        best_match = word
        for correct_word in self.correct_words:
            d = levenshtein_distance(word.lower(), correct_word.lower())
            if d < min_distance and d <= 2:
                min_distance = d
                best_match = correct_word
        return best_match
 
    def correct_text(self, text):
        """Corriger tout le texte d'une boîte"""
        words = text.split()
        corrected_words = [self.correct_word(word) for word in words]
        return " ".join(corrected_words)
 
    def extract_info(self, ocr_data):
        """Extraire les informations importantes du texte OCR"""
        numero_police = None
        date_debut = None
        date_fin = None
 
        # Extraire le numéro de police
        for i, box in enumerate(ocr_data):
            # Utiliser une regex pour ignorer les espaces et les erreurs de reconnaissance
            if re.search(r"Numéro\s*de\s*la\s*police", box["text"].lower(), re.IGNORECASE):
                # Vérifier que i + 1 est dans les limites de la liste
                if i + 1 < len(ocr_data):
                    numero_police = ocr_data[i + 1]["text"]
                break
 
        # Extraire les dates
        text_combined = " ".join([box["text"] for box in ocr_data])
        dates = self.date_pattern.findall(text_combined)
 
        if len(dates) >= 2:
            date_debut = " ".join(dates[0])
            date_fin = " ".join(dates[1])
 
        return numero_police, date_debut, date_fin
 
    def process_image(self, image_path):
        """Traiter une image avec PaddleOCR et extraire les informations"""
        # Traiter l'image avec PaddleOCR
        results = self.ocr.ocr(image_path, cls=True)
        
        # Extraire les boîtes, les textes et les scores de confiance
        ocr_results = []
        if results[0]:
            for res in results[0]:
                text = res[1][0]
                score = res[1][1]
                corrected_text = self.correct_text(text)
                ocr_results.append({"text": corrected_text, "score": score})
        
        # Extraire les informations importantes
        numero_police, date_debut, date_fin = self.extract_info(ocr_results)
        
        return {
            #"ocr_results": ocr_results,
            "extracted_info": {
                "numero_police": numero_police,
                "date_debut": date_debut,
                "date_fin": date_fin
            }
        }