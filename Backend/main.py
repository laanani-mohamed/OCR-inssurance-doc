from flask import Flask, request, jsonify
import os
from werkzeug.utils import secure_filename
import tempfile
import logging
import time
from processor_image import ImageProcessor
from processor_pdf import ConstatProcessor
import torch

# Configuration du logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Fonction pour vérifier la mémoire GPU disponible
def check_gpu_memory(required_memory_mb):
    """Vérifie s'il y a assez de mémoire GPU disponible.
    
    Args:
        required_memory_mb: Mémoire requise en Mo
        
    Returns:
        bool: True si assez de mémoire GPU disponible, False sinon
    """
    if not torch.cuda.is_available():
        logger.info("GPU non disponible, utilisation du CPU.")
        return False
    
    try:
        # Récupérer les informations de mémoire GPU
        free_memory = torch.cuda.mem_get_info()[0] / (1024 * 1024)  # En Mo
        logger.info(f"Mémoire GPU libre: {free_memory:.2f} Mo")
        
        if free_memory < required_memory_mb:
            logger.warning(f"Mémoire GPU insuffisante ({free_memory:.2f} Mo libre, {required_memory_mb} Mo requis)")
            return False
        
        return True
    except Exception as e:
        logger.error(f"Erreur lors de la vérification de la mémoire GPU: {str(e)}")
        return False

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Limite à 16 Mo
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()  # Dossier temporaire pour les fichiers

# Extensions permises
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'pdf'}

def allowed_file(filename):
    """Vérifier si le fichier a une extension autorisée"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/health', methods=['GET'])
def health_check():
    """Endpoint pour vérifier que l'API fonctionne"""
    return jsonify({"status": "ok"})

@app.route('/models/check', methods=['GET'])
def check_models():
    """Endpoint pour vérifier que les modèles sont chargés"""
    try:
        # Vérifier la mémoire GPU disponible
        gpu_available = torch.cuda.is_available()
        if gpu_available:
            free_memory = torch.cuda.mem_get_info()[0] / (1024 * 1024)  # En Mo
            memory_status = f"{free_memory:.2f} Mo disponible"
        else:
            memory_status = "non disponible"
        
        # Vérifier PaddleOCR
        from paddleocr import PaddleOCR
        import paddle
        use_gpu_paddle = check_gpu_memory(1500)  # 1.5 Go minimum pour PaddleOCR
        paddle_ocr = PaddleOCR(use_angle_cls=True, lang='fr', use_gpu=use_gpu_paddle)
        paddle_device = "GPU" if use_gpu_paddle else "CPU"
        
        # Vérifier TrOCR
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        use_gpu_trocr = check_gpu_memory(3000)  # 3 Go minimum pour TrOCR large
        device_trocr = torch.device("cuda" if use_gpu_trocr else "cpu")
        trocr_processor = TrOCRProcessor.from_pretrained("microsoft/trocr-large-handwritten")
        trocr_model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-large-handwritten").to(device_trocr)
        trocr_device = "GPU" if use_gpu_trocr else "CPU"
        
        # Vérifier doctr
        from doctr.models import ocr_predictor, fast_tiny
        use_gpu_doctr = check_gpu_memory(1000)  # 1 Go minimum pour doctr
        device_doctr = torch.device("cuda" if use_gpu_doctr else "cpu")
        doctr_model = fast_tiny(pretrained=True).to(device_doctr)
        doctr_device = "GPU" if use_gpu_doctr else "CPU"
        
        return jsonify({
            "status": "ok",
            "gpu": {
                "available": gpu_available,
                "memory": memory_status
            },
            "models": {
                "paddleocr": f"loaded on {paddle_device}",
                "trocr": f"loaded on {trocr_device}",
                "doctr": f"loaded on {doctr_device}"
            }
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/ocr/image', methods=['POST'])
def process_image():
    """Endpoint pour traiter une image avec OCR"""
    # Démarrer le chronomètre
    start_time = time.time()
    
    # Vérifier qu'un fichier a été envoyé
    if 'file' not in request.files:
        return jsonify({"error": "Aucun fichier n'a été envoyé"}), 400
    
    file = request.files['file']
    
    # Vérifier que le fichier a un nom
    if file.filename == '':
        return jsonify({"error": "Aucun fichier sélectionné"}), 400
    
    # Vérifier que c'est un fichier image autorisé
    if file and allowed_file(file.filename) and file.filename.rsplit('.', 1)[1].lower() in ['png', 'jpg', 'jpeg']:
        try:
            # Sauvegarder le fichier temporairement
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Vérifier la mémoire GPU disponible avant d'initialiser le processeur
            use_gpu = check_gpu_memory(1500)  # 1.5 Go pour PaddleOCR
            device_type = "GPU" if use_gpu else "CPU"
            logger.info(f"Traitement de l'image sur {device_type}")
            
            # Initialiser le processeur d'images et traiter l'image - sans passer use_gpu
            image_processor = ImageProcessor()
            result = image_processor.process_image(filepath)
            
            # Nettoyer le fichier temporaire
            os.remove(filepath)
            
            # Calculer le temps d'exécution
            execution_time = time.time() - start_time
            
            # Ajouter le temps d'exécution et l'appareil utilisé au résultat
            result['execution_time'] = {
                'seconds': execution_time,
                'device': device_type
            }
            
            # Journaliser le temps d'exécution
            logger.info(f"Image traitée en {execution_time:.2f} secondes sur {device_type}: {file.filename}")
            
            return jsonify(result)
        except Exception as e:
            # Calculer le temps d'exécution même en cas d'erreur
            execution_time = time.time() - start_time
            logger.error(f"Erreur lors du traitement de l'image ({execution_time:.2f}s): {str(e)}")
            
            return jsonify({
                "error": f"Erreur lors du traitement: {str(e)}",
                "execution_time": {
                    'seconds': execution_time,
                }
            }), 500
    
    return jsonify({"error": "Type de fichier non pris en charge"}), 400

@app.route('/ocr/pdf', methods=['POST'])
def process_pdf():
    """Endpoint pour traiter un PDF de constat amiable"""
    # Démarrer le chronomètre
    start_time = time.time()
    
    # Vérifier qu'un fichier a été envoyé
    if 'file' not in request.files:
        return jsonify({"error": "Aucun fichier n'a été envoyé"}), 400
    
    file = request.files['file']
    
    # Vérifier que le fichier a un nom
    if file.filename == '':
        return jsonify({"error": "Aucun fichier sélectionné"}), 400
    
    # Vérifier que c'est un fichier PDF
    if file and allowed_file(file.filename) and file.filename.rsplit('.', 1)[1].lower() == 'pdf':
        try:
            # Sauvegarder le fichier temporairement
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Vérifier la mémoire GPU disponible avant d'initialiser le processeur
            use_gpu = check_gpu_memory(3000)  # 3 Go pour TrOCR large
            device_type = "GPU" if use_gpu else "CPU"
            logger.info(f"Traitement du PDF sur {device_type}")
            
            # Forcer l'utilisation du CPU si la mémoire GPU est insuffisante
            if not use_gpu:
                os.environ['CUDA_VISIBLE_DEVICES'] = ''
                logger.warning("Mémoire GPU insuffisante, forçage du traitement sur CPU")
            
            # Initialiser le processeur de PDF et traiter le fichier
            pdf_processor = ConstatProcessor(use_gpu=use_gpu)
            result = pdf_processor.process_pdf(filepath)
            
            # Nettoyer le fichier temporaire
            os.remove(filepath)
            
            # Calculer le temps d'exécution
            execution_time = time.time() - start_time
            
            # Si le résultat est un dictionnaire, ajouter le temps d'exécution
            if isinstance(result, dict):
                result['execution_time'] = {
                    'seconds': execution_time,
                    'device': device_type
                }
            else:
                # Sinon, encapsuler le résultat
                result = {
                    'data': result,
                    'execution_time': {
                        'seconds': execution_time,
                        'device': device_type
                    }
                }
            
            # Journaliser le temps d'exécution
            logger.info(f"PDF traité en {execution_time:.2f} secondes sur {device_type}: {file.filename}")
            
            return jsonify(result)
        except Exception as e:
            # Calculer le temps d'exécution même en cas d'erreur
            execution_time = time.time() - start_time
            logger.error(f"Erreur lors du traitement du PDF ({execution_time:.2f}s): {str(e)}")
            
            return jsonify({
                "error": f"Erreur lors du traitement: {str(e)}",
                "execution_time": {
                    'seconds': execution_time,
                }
            }), 500
    
    return jsonify({"error": "Type de fichier non pris en charge"}), 400

def process_image_file(image_path):
    """Fonction pour traiter une image spécifiée en ligne de commande"""
    import json
    
    # Vérifier la mémoire GPU disponible
    use_gpu = check_gpu_memory(1500)  # 1.5 Go pour PaddleOCR
    device_type = "GPU" if use_gpu else "CPU"
    logger.info(f"Traitement de l'image sur {device_type}")
    
    start_time = time.time()
    # Initialiser le processeur d'image sans passer le paramètre use_gpu
    image_processor = ImageProcessor()
    result = image_processor.process_image(image_path)
    execution_time = time.time() - start_time
    
    if isinstance(result, dict):
        result['execution_time'] = {
            'seconds': execution_time,
            'device': device_type
        }
    else:
        result = {
            'data': result,
            'execution_time': {
                'seconds': execution_time,
                'device': device_type
            }
        }
    
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"Image traitée en {execution_time:.2f} secondes sur {device_type}")
    return result

def process_pdf_file(pdf_path):
    """Fonction pour traiter un PDF spécifié en ligne de commande"""
    import json
    
    # Vérifier la mémoire GPU disponible
    use_gpu = check_gpu_memory(3000)  # 3 Go pour TrOCR large
    device_type = "GPU" if use_gpu else "CPU"
    logger.info(f"Traitement du PDF sur {device_type}")
    
    # Forcer l'utilisation du CPU si la mémoire GPU est insuffisante
    if not use_gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        logger.warning("Mémoire GPU insuffisante, forçage du traitement sur CPU")
    
    start_time = time.time()
    pdf_processor = ConstatProcessor(use_gpu=use_gpu)
    result = pdf_processor.process_pdf(pdf_path)
    execution_time = time.time() - start_time
    
    if isinstance(result, dict):
        result['execution_time'] = {
            'seconds': execution_time,
            'device': device_type
        }
    else:
        result = {
            'data': result,
            'execution_time': {
                'seconds': execution_time,
                'device': device_type
            }
        }
    
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"PDF traité en {execution_time:.2f} secondes sur {device_type}")
    return result

if __name__ == '__main__':
    import sys
    
    # Si un argument est fourni, on détermine si c'est un PDF ou une image
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        # Vérifier l'extension du fichier
        if file_path.lower().endswith(('.pdf')):
            print(f"Traitement du PDF: {file_path}")
            process_pdf_file(file_path)
        elif file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            print(f"Traitement de l'image: {file_path}")
            process_image_file(file_path)
        else:
            print(f"Format de fichier non pris en charge: {file_path}")
            print("Extensions supportées: .pdf, .png, .jpg, .jpeg")
    else:
        # Démarrage normal de l'API
        debug_mode = False  # Simplifié ici
        port = int(os.environ.get('PORT', 5090))
        logger.info(f"Démarrage de l'API OCR sur le port {port}, debug={debug_mode}")
        app.run(host='0.0.0.0', port=port, debug=debug_mode)