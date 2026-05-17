import pandas as pd
import numpy as np
import sys
import os
import warnings
warnings.filterwarnings('ignore')

# --- Ajouter le dossier racine au path ---
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.data_collection.nasa_power   import get_meteo_summary
from src.data_collection.earth_engine import initialiser_ee, get_ndvi
from src.data_collection.soilgrids    import get_soilgrids_data
from src.data_collection.kaggle_loader import charger_dataset


def collecter_donnees_terrain(latitude, longitude, project_ee='agri-smart-495220'):
    """
    Collecte toutes les données pour un point GPS donné.
    Combine NASA + Earth Engine + SoilGrids en un seul dictionnaire.

    Returns:
        dict avec toutes les features du terrain
    """

    print("\n" + "="*55)
    print("🌍 COLLECTE COMPLÈTE DES DONNÉES TERRAIN")
    print("="*55)
    print(f"📍 Localisation : {latitude}, {longitude}")

    features = {'latitude': latitude, 'longitude': longitude}

    # ── 1. NASA POWER ──────────────────────────────────────
    print("\n[1/3] 📡 Météo NASA POWER...")
    meteo = get_meteo_summary(latitude, longitude, nb_jours=30)

    if meteo:
        features['temperature']    = meteo['temperature_moy']
        features['humidity']       = meteo['humidite_moy']
        features['rainfall']       = meteo['pluie_totale']
        features['rayonnement']    = meteo['rayonnement_moy']
    else:
        print("   ⚠️  Météo indisponible — valeurs par défaut")
        features['temperature']    = 25.0
        features['humidity']       = 70.0
        features['rainfall']       = 80.0
        features['rayonnement']    = 18.0

    # ── 2. EARTH ENGINE / NDVI ─────────────────────────────
    print("\n[2/3] 🛰️  NDVI Earth Engine...")
    try:
        initialiser_ee(project=project_ee)
        ndvi_data = get_ndvi(latitude, longitude, nb_jours=30)
        features['ndvi'] = ndvi_data['ndvi_moyen']
    except Exception as e:
        print(f"   ⚠️  NDVI indisponible : {e}")
        features['ndvi'] = 0.4

    # ── 3. SOILGRIDS ───────────────────────────────────────
    print("\n[3/3] 🌱 Données sol SoilGrids...")
    sol = get_soilgrids_data(latitude, longitude)

    if sol:
        features['ph']                 = sol['ph']
        features['argile']             = sol['argile_pct']
        features['sable']              = sol['sable_pct']
        features['limon']              = sol['limon_pct']
        features['carbone_organique']  = sol['carbone_organique']
    else:
        print("   ⚠️  Sol indisponible — valeurs par défaut")
        features['ph']                 = 6.5
        features['argile']             = 25.0
        features['sable']              = 40.0
        features['limon']              = 35.0
        features['carbone_organique']  = 10.0

    return features


def construire_dataset_entrainement(
        chemin_kaggle='data/raw/crop_recommendation.csv',
        chemin_sortie='data/processed/dataset_final.csv'):
    """
    Construit le dataset final d'entraînement en enrichissant
    le dataset Kaggle avec des features calculées.

    Returns:
        DataFrame final prêt pour XGBoost
    """

    print("\n" + "="*55)
    print("🔧 CONSTRUCTION DU DATASET FINAL")
    print("="*55)

    # ── Charger le dataset Kaggle ───────────────────────────
    df = charger_dataset(chemin_kaggle)
    if df is None:
        return None

    print(f"\n✅ Base : {len(df)} exemples Kaggle chargés")

    # ── Feature Engineering ─────────────────────────────────
    print("\n⚙️  Feature Engineering...")

    # 1. Stress hydrique estimé
    # ET0 simplifié (formule de Hargreaves simplifiée)
    df['ET0_estime']      = 0.0023 * (df['temperature'] + 17.8) * 0.5
    df['stress_hydrique'] = (df['ET0_estime'] - df['rainfall'] / 30).clip(lower=0)
    print("   ✅ Stress hydrique calculé")

    # 2. Ratio N/P/K normalisés
    total_npk             = df['N'] + df['P'] + df['K'] + 1
    df['ratio_N']         = df['N'] / total_npk
    df['ratio_P']         = df['P'] / total_npk
    df['ratio_K']         = df['K'] / total_npk
    print("   ✅ Ratios N/P/K calculés")

    # 3. Indice de fertilité sol (combinaison pH + NPK)
    df['fertilite_idx']   = (
        df['ph'] * 0.3 +
        df['N']  * 0.02 +
        df['P']  * 0.02 +
        df['K']  * 0.02
    )
    print("   ✅ Indice de fertilité calculé")

    # 4. Indice de confort climatique
    df['confort_clim']    = (
        (1 - abs(df['temperature'] - 25) / 25) * 0.5 +
        (df['humidity'] / 100) * 0.3 +
        (df['rainfall'].clip(0, 200) / 200) * 0.2
    ).clip(0, 1)
    print("   ✅ Indice de confort climatique calculé")

    # 5. NDVI simulé depuis les données Kaggle
    # (NDVI réel sera ajouté lors des prédictions terrain)
    df['ndvi_estime']     = (
        0.3 +
        (df['humidity'] / 100) * 0.3 +
        (df['rainfall'].clip(0, 200) / 200) * 0.2 +
        (df['N'] / 140) * 0.2
    ).clip(0, 1)
    print("   ✅ NDVI estimé calculé")

    # ── Encodage de la colonne label ────────────────────────
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    df['label_encoded'] = le.fit_transform(df['label'])

    # Sauvegarder le mapping
    mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    print(f"\n📋 Mapping cultures → codes :")
    for culture, code in mapping.items():
        print(f"   {code:2d} → {culture}")

    # ── Sauvegarder le dataset final ────────────────────────
    os.makedirs('data/processed', exist_ok=True)
    df.to_csv(chemin_sortie, index=False)
    print(f"\n✅ Dataset sauvegardé : {chemin_sortie}")

    # ── Résumé final ────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"📦 DATASET FINAL PRÊT")
    print(f"{'='*55}")
    print(f"   Lignes      : {df.shape[0]}")
    print(f"   Colonnes    : {df.shape[1]}")
    print(f"\n   Colonnes originales (Kaggle) :")
    kaggle_cols = ['N','P','K','temperature','humidity','ph','rainfall','label']
    for c in kaggle_cols:
        print(f"      - {c}")
    print(f"\n   Nouvelles features ajoutées :")
    new_cols = ['ET0_estime','stress_hydrique','ratio_N','ratio_P',
                'ratio_K','fertilite_idx','confort_clim','ndvi_estime']
    for c in new_cols:
        print(f"      + {c}")

    return df, le, mapping


# ── TEST DIRECT ─────────────────────────────────────────────
if __name__ == "__main__":

    print("="*55)
    print("TEST 1 — COLLECTE TERRAIN (point GPS réel)")
    print("="*55)
    features_terrain = collecter_donnees_terrain(33.97, -6.85)
    print("\n📦 Features terrain collectées :")
    for k, v in features_terrain.items():
        print(f"   {k:25s} : {v}")

    print("\n\n" + "="*55)
    print("TEST 2 — CONSTRUCTION DATASET ENTRAÎNEMENT")
    print("="*55)
    resultat = construire_dataset_entrainement()

    if resultat:
        df, le, mapping = resultat
        print(f"\n📋 Aperçu du dataset final :")
        print(df.head(3).to_string(index=False))