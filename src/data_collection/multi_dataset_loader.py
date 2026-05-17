import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')


def charger_et_fusionner_datasets(dossier='data/raw'):
    """
    Charge et fusionne plusieurs datasets agricoles
    en un dataset unifié et enrichi.
    """
    print("="*55)
    print("📊 CHARGEMENT MULTI-DATASETS")
    print("="*55)

    datasets = []

    # ── Dataset 1 : Kaggle original ──────────────────────
    chemin1 = f'{dossier}/crop_recommendation.csv'
    if os.path.exists(chemin1):
        df1 = pd.read_csv(chemin1)
        df1 = df1.rename(columns={'label': 'culture'})
        df1['source'] = 'kaggle_original'
        datasets.append(df1)
        print(f"✅ Dataset 1 : {len(df1)} lignes — Crop Recommendation")
    else:
        print(f"❌ Non trouvé : {chemin1}")

    # ── Dataset 2 : Fertilizer Dataset ───────────────────
    chemin2 = f'{dossier}/fertilizer_prediction.csv'
    if os.path.exists(chemin2):
        df2 = pd.read_csv(chemin2)
        print(f"   Colonnes disponibles : {list(df2.columns)}")
        df2 = _harmoniser_dataset(df2, 'fertilizer')
        if df2 is not None:
            datasets.append(df2)
            print(f"✅ Dataset 2 : {len(df2)} lignes — Fertilizer")
    else:
        print(f"⚠️  Dataset 2 non trouvé : {chemin2}")

    # ── Dataset 3 : Crop Fertilizer ──────────────────────
    chemin3 = f'{dossier}/crop_fertilizer.csv'
    if os.path.exists(chemin3):
        df3 = pd.read_csv(chemin3)
        print(f"   Colonnes disponibles : {list(df3.columns)}")
        df3 = _harmoniser_dataset(df3, 'crop_fertilizer')
        if df3 is not None:
            datasets.append(df3)
            print(f"✅ Dataset 3 : {len(df3)} lignes — Crop Fertilizer")
    else:
        print(f"⚠️  Dataset 3 non trouvé : {chemin3}")

    # ── Dataset Maroc synthétique ─────────────────────────
    df_maroc = _generer_dataset_maroc()
    datasets.append(df_maroc)
    print(f"✅ Dataset Maroc : {len(df_maroc)} lignes — Synthétique Maghreb")

    # ── Fusion ────────────────────────────────────────────
    if not datasets:
        print("❌ Aucun dataset chargé")
        return None

    df_final = pd.concat(datasets, ignore_index=True)

    # Colonnes obligatoires
    colonnes = ['N', 'P', 'K', 'temperature',
                'humidity', 'ph', 'rainfall', 'culture']
    df_final = df_final[colonnes + ['source']].dropna(subset=colonnes)

    print(f"\n{'='*55}")
    print(f"📦 DATASET FUSIONNÉ")
    print(f"{'='*55}")
    print(f"   Total lignes  : {len(df_final)}")
    print(f"\n   Cultures disponibles ({df_final['culture'].nunique()}) :")
    for c, n in df_final['culture'].value_counts().items():
        barre = '█' * (n // 10)
        print(f"   {c:20s} : {barre} ({n})")

    return df_final


def _harmoniser_dataset(df, type_dataset):
    mappings = {
        'N'          : ['N', 'Nitrogen', 'nitrogen'],
        'P'          : ['P', 'Phosphorus', 'Phosphorous', 'phosphorus'],
        'K'          : ['K', 'Potassium', 'potassium'],
        'temperature': ['temperature', 'Temperature', 'Temparature'],
        'humidity'   : ['humidity', 'Humidity', 'Humidity '],
        'ph'         : ['ph', 'pH', 'PH'],
        'rainfall'   : ['rainfall', 'Rainfall', 'Rain'],
        'culture'    : ['label', 'Crop', 'crop', 'CropType',
                        'Crop_Type', 'culture', 'Crop Type'],
    }

    df_new = pd.DataFrame()
    for col_cible, noms_possibles in mappings.items():
        for nom in noms_possibles:
            if nom in df.columns:
                df_new[col_cible] = df[nom]
                break

    # ── NOUVEAU : Estimer humidity si manquante ──
    if 'humidity' not in df_new.columns:
        if 'Moisture' in df.columns:
            # Moisture → humidité estimée
            df_new['humidity'] = df['Moisture'] * 1.3
            df_new['humidity'] = df_new['humidity'].clip(20, 99)
            print(f"   ℹ️  Humidity estimée depuis Moisture")
        else:
            # Valeur moyenne selon température
            df_new['humidity'] = 65.0
            print(f"   ℹ️  Humidity fixée à 65% (valeur par défaut)")

    # ── NOUVEAU : Estimer ph si manquant ──
    if 'ph' not in df_new.columns:
        df_new['ph'] = 6.5
        print(f"   ℹ️  pH fixé à 6.5 (valeur par défaut)")

    # ── NOUVEAU : Estimer rainfall si manquant ──
    if 'rainfall' not in df_new.columns:
        df_new['rainfall'] = 100.0
        print(f"   ℹ️  Rainfall fixé à 100mm")

    # Vérifier colonnes obligatoires
    cols_obligatoires = ['N', 'P', 'K', 'temperature',
                         'humidity', 'ph', 'rainfall', 'culture']
    manquantes = [c for c in cols_obligatoires
                  if c not in df_new.columns]

    if manquantes:
        print(f"   ⚠️  Colonnes encore manquantes : {manquantes}")
        return None

    df_new['source'] = type_dataset

    # Nettoyer valeurs aberrantes
    df_new['N'] = df_new['N'].clip(0, 300)
    df_new['P'] = df_new['P'].clip(0, 300)
    df_new['K'] = df_new['K'].clip(0, 300)
    df_new['ph'] = df_new['ph'].clip(3, 10)
    df_new['temperature'] = df_new['temperature'].clip(5, 50)
    df_new['humidity'] = df_new['humidity'].clip(20, 99)
    df_new['rainfall'] = df_new['rainfall'].clip(0, 3000)

    return df_new


def _generer_dataset_maroc():
    """
    Génère un dataset synthétique basé sur les
    conditions agricoles réelles du Maroc/Maghreb.
    Basé sur les recommandations de l'ONCA et FAO.
    """
    np.random.seed(42)
    lignes = []

    # Profils agricoles réels du Maroc
    # Source : FAO Morocco Country Profile + ONCA
    cultures_maroc = {
        # culture: (N_moy, N_std, P_moy, P_std, K_moy, K_std,
        #           temp_moy, temp_std, hum_moy, hum_std,
        #           ph_moy, ph_std, rain_moy, rain_std, nb_exemples)
        'blé': (
            120, 20, 60, 10, 40, 8,
            15, 4, 65, 15,
            7.0, 0.3, 400, 80, 150
        ),
        'orge': (
            80, 15, 50, 10, 30, 8,
            14, 4, 60, 15,
            7.2, 0.3, 350, 70, 100
        ),
        'maize': (
            150, 25, 80, 15, 60, 10,
            22, 3, 65, 10,
            6.8, 0.3, 500, 100, 100
        ),
        'tournesol': (
            80, 15, 60, 10, 80, 15,
            20, 3, 55, 10,
            6.5, 0.4, 400, 80, 80
        ),
        'tomate': (
            120, 20, 80, 15, 150, 25,
            22, 3, 70, 10,
            6.5, 0.3, 600, 100, 100
        ),
        'pomme_de_terre': (
            150, 25, 100, 20, 200, 30,
            18, 3, 75, 10,
            5.8, 0.3, 600, 100, 80
        ),
        'olive': (
            60, 15, 30, 8, 80, 15,
            18, 5, 55, 15,
            7.5, 0.4, 300, 100, 100
        ),
        'orange': (
            120, 20, 60, 12, 120, 20,
            20, 3, 65, 10,
            6.5, 0.3, 800, 150, 80
        ),
        'muskmelon': (
            50, 10, 50, 10, 50, 10,
            25, 3, 65, 10,
            7.0, 0.3, 80, 30, 100
        ),
        'watermelon': (
            60, 12, 40, 8, 40, 8,
            27, 3, 65, 10,
            6.8, 0.3, 50, 20, 100
        ),
        'pomegranate': (
            60, 12, 40, 8, 60, 12,
            22, 5, 50, 15,
            7.0, 0.4, 300, 80, 80
        ),
        'grapes': (
            40, 10, 30, 8, 60, 12,
            20, 4, 55, 15,
            6.5, 0.4, 400, 100, 80
        ),
        'lentil': (
            20, 5, 40, 8, 20, 5,
            18, 4, 60, 15,
            6.5, 0.3, 350, 70, 80
        ),
        'chickpea': (
            20, 5, 50, 10, 20, 5,
            20, 4, 55, 15,
            6.8, 0.3, 300, 60, 80
        ),
    }

    for culture, params in cultures_maroc.items():
        (N_m, N_s, P_m, P_s, K_m, K_s,
         t_m, t_s, h_m, h_s,
         ph_m, ph_s, r_m, r_s, n) = params

        for _ in range(n):
            lignes.append({
                'N'          : max(0, np.random.normal(N_m, N_s)),
                'P'          : max(0, np.random.normal(P_m, P_s)),
                'K'          : max(0, np.random.normal(K_m, K_s)),
                'temperature': max(5, np.random.normal(t_m, t_s)),
                'humidity'   : np.clip(np.random.normal(h_m, h_s), 20, 99),
                'ph'         : np.clip(np.random.normal(ph_m, ph_s), 4, 9),
                'rainfall'   : max(0, np.random.normal(r_m, r_s)),
                'culture'    : culture,
                'source'     : 'maroc_fao_onca',
            })

    return pd.DataFrame(lignes)


# ── TEST ─────────────────────────────────────────────────
if __name__ == "__main__":
    df = charger_et_fusionner_datasets()
    if df is not None:
        print(f"\n✅ Dataset prêt : {len(df)} exemples")
        print(f"   Cultures : {sorted(df['culture'].unique())}")