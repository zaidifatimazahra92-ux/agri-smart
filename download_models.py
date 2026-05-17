"""
Télécharge les modèles depuis Google Drive au démarrage.
"""
import os
import gdown

# ── Remplace ces IDs par les vrais IDs de tes fichiers Google Drive ──
# Format lien : https://drive.google.com/file/d/ID_ICI/view
# L'ID est la partie entre /d/ et /view

MODELES = {
    'models/features.pkl'             : 'https://drive.google.com/file/d/1VrLmc7bAnznAogU2YWkDtZ0-Gxlt9XBv/view?usp=sharing',
    'models/label_encoder.pkl'        : 'https://drive.google.com/file/d/1iYe8dpXKQYMz66BVcDIARmaI9mZoDSV5/view?usp=sharing',
    'models/model_classification.pkl' : 'https://drive.google.com/file/d/1kxLvONSjZmDh346i8TzwtkeOR3SjJ8jE/view?usp=sharing',
    'models/modeles_regression.pkl'   : 'https://drive.google.com/file/d/1cdwNnC4ehelE54EF5LMT9ZEL2Ubsqr1y/view?usp=sharing',
    'models/scaler.pkl'               : 'https://drive.google.com/file/d/14c5EG7xpeZSdiDpwH59mDMr3WWX6myeB/view?usp=sharing',
}

def telecharger_modeles():
    os.makedirs('models', exist_ok=True)

    for chemin, file_id in MODELES.items():
        if os.path.exists(chemin):
            print(f"✅ Déjà présent : {chemin}")
            continue

        print(f"⬇️  Téléchargement : {chemin}")
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, chemin, quiet=False)
        print(f"✅ OK : {chemin}")

if __name__ == "__main__":
    telecharger_modeles()