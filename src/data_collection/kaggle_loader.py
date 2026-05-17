import pandas as pd
import os
import warnings
warnings.filterwarnings('ignore')


def charger_dataset(chemin='data/raw/crop_recommendation.csv'):
    """
    Charge et vérifie le dataset Kaggle Crop Recommendation.
    
    Returns:
        DataFrame prêt à l'emploi
    """

    print("=" * 50)
    print("📊 CHARGEMENT DATASET KAGGLE")
    print("=" * 50)

    # --- Vérifier que le fichier existe ---
    if not os.path.exists(chemin):
        print(f"❌ Fichier non trouvé : {chemin}")
        print("   → Place ton fichier CSV dans data/raw/")
        return None

    # --- Chargement ---
    df = pd.read_csv(chemin)
    print(f"✅ Fichier chargé : {chemin}")

    # --- Informations générales ---
    print(f"\n📋 Dimensions :")
    print(f"   Lignes   : {df.shape[0]}")
    print(f"   Colonnes : {df.shape[1]}")

    print(f"\n📋 Colonnes disponibles :")
    for col in df.columns:
        print(f"   - {col} ({df[col].dtype})")

    # --- Aperçu ---
    print(f"\n📋 Aperçu des 5 premières lignes :")
    print(df.head().to_string(index=False))

    # --- Valeurs manquantes ---
    print(f"\n📋 Valeurs manquantes :")
    manquantes = df.isnull().sum()
    if manquantes.sum() == 0:
        print("   ✅ Aucune valeur manquante")
    else:
        print(manquantes[manquantes > 0])

    # --- Statistiques ---
    print(f"\n📋 Statistiques :")
    print(df.describe().round(2).to_string())

    # --- Cultures disponibles ---
    print(f"\n🌾 Cultures dans le dataset ({df['label'].nunique()} types) :")
    for i, culture in enumerate(sorted(df['label'].unique())):
        print(f"   {i+1:2d}. {culture}")

    # --- Vérification des colonnes attendues ---
    colonnes_attendues = ['N', 'P', 'K', 'temperature',
                          'humidity', 'ph', 'rainfall', 'label']
    colonnes_ok = all(col in df.columns for col in colonnes_attendues)

    if colonnes_ok:
        print(f"\n✅ Toutes les colonnes attendues sont présentes")
    else:
        manquantes = [c for c in colonnes_attendues if c not in df.columns]
        print(f"\n⚠️  Colonnes manquantes : {manquantes}")

    return df


# --- TEST DIRECT ---
if __name__ == "__main__":
    df = charger_dataset()

    if df is not None:
        print(f"\n📦 Dataset prêt — {len(df)} exemples d'entraînement")