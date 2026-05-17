import pandas as pd
import numpy as np
import pickle
import os
import sys
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import LabelEncoder, StandardScaler
from sklearn.metrics         import (accuracy_score, f1_score,
                                     classification_report,
                                     mean_absolute_error, r2_score)
import xgboost as xgb

from src.data_collection.multi_dataset_loader import charger_et_fusionner_datasets


def entrainer_v2():
    print("="*55)
    print("🚀 ENTRAÎNEMENT V2 — MULTI-DATASETS")
    print("="*55)

    # ── 1. Charger les données fusionnées ────────────────
    df = charger_et_fusionner_datasets()
    if df is None:
        return

    # ── 2. Feature Engineering ───────────────────────────
    print("\n⚙️  Feature Engineering...")

    total_npk            = df['N'] + df['P'] + df['K'] + 1
    df['ratio_N']        = df['N'] / total_npk
    df['ratio_P']        = df['P'] / total_npk
    df['ratio_K']        = df['K'] / total_npk
    df['ET0_estime']     = 0.0023 * (df['temperature'] + 17.8) * 0.5
    df['stress_hydrique']= (df['ET0_estime'] - df['rainfall']/30).clip(lower=0)
    df['fertilite_idx']  = (
        df['ph'] * 0.3 + df['N'] * 0.02 +
        df['P']  * 0.02 + df['K'] * 0.02
    )
    df['confort_clim']   = (
        (1 - abs(df['temperature'] - 25) / 25) * 0.5 +
        (df['humidity'] / 100) * 0.3 +
        (df['rainfall'].clip(0, 200) / 200) * 0.2
    ).clip(0, 1)
    df['ndvi_estime']    = (
        0.3 + (df['humidity']/100)*0.3 +
        (df['rainfall'].clip(0,200)/200)*0.2 +
        (df['N']/140)*0.2
    ).clip(0, 1)

    print("   ✅ Features calculées")

    # ── 3. Préparer X et y ───────────────────────────────
    features = [
        'N', 'P', 'K', 'temperature', 'humidity',
        'ph', 'rainfall', 'ET0_estime', 'stress_hydrique',
        'ratio_N', 'ratio_P', 'ratio_K',
        'fertilite_idx', 'confort_clim', 'ndvi_estime'
    ]

    X = df[features]
    y = df['culture']

    le     = LabelEncoder()
    y_enc  = le.fit_transform(y)
    scaler = StandardScaler()
    X_sc   = pd.DataFrame(scaler.fit_transform(X), columns=features)

    print(f"   ✅ {len(df)} exemples, {len(le.classes_)} cultures")
    print(f"   Cultures : {list(le.classes_)}")

    # ── 4. Split ─────────────────────────────────────────
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_sc, y_enc, test_size=0.2,
        random_state=42, stratify=y_enc
    )
    print(f"\n   Train : {len(X_tr)} | Test : {len(X_te)}")

    # ── 5. Classification ────────────────────────────────
    print("\n🤖 Entraînement classification...")
    clf = xgb.XGBClassifier(
        n_estimators=400, max_depth=7,
        learning_rate=0.08, subsample=0.8,
        colsample_bytree=0.8, random_state=42,
        eval_metric='mlogloss', verbosity=0
    )
    clf.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)

    y_pred   = clf.predict(X_te)
    accuracy = accuracy_score(y_te, y_pred)
    f1       = f1_score(y_te, y_pred, average='weighted')

    print(f"   ✅ Accuracy : {accuracy*100:.2f}%")
    print(f"   ✅ F1-Score : {f1*100:.2f}%")
    print(f"\n   Rapport par culture :")
    print(classification_report(y_te, y_pred,
                                target_names=le.classes_))

    # ── 6. Régression N, P, K, irrigation ───────────────
    print("\n📈 Entraînement régression...")
    modeles_reg = {}
    idx_tr = X_tr.index
    idx_te = X_te.index

    for cible in ['N', 'P', 'K', 'rainfall']:
        reg = xgb.XGBRegressor(
            n_estimators=300, max_depth=6,
            learning_rate=0.1, random_state=42, verbosity=0
        )
        reg.fit(X_tr, df.loc[idx_tr, cible].values)
        preds  = reg.predict(X_te)
        mae    = mean_absolute_error(df.loc[idx_te, cible].values, preds)
        r2     = r2_score(df.loc[idx_te, cible].values, preds)
        print(f"   {cible:10s} → MAE={mae:.2f}  R²={r2:.4f}")
        modeles_reg[cible] = reg

    # ── 7. Sauvegarder ───────────────────────────────────
    print("\n💾 Sauvegarde des modèles V2...")
    os.makedirs('models', exist_ok=True)

    with open('models/model_classification.pkl', 'wb') as f:
        pickle.dump(clf, f)
    with open('models/modeles_regression.pkl', 'wb') as f:
        pickle.dump(modeles_reg, f)
    with open('models/label_encoder.pkl', 'wb') as f:
        pickle.dump(le, f)
    with open('models/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    with open('models/features.pkl', 'wb') as f:
        pickle.dump(features, f)

    print("   ✅ models/model_classification.pkl")
    print("   ✅ models/modeles_regression.pkl")
    print("   ✅ models/label_encoder.pkl")
    print("   ✅ models/scaler.pkl")
    print("   ✅ models/features.pkl")

    print(f"\n{'='*55}")
    print(f"🎉 ENTRAÎNEMENT V2 TERMINÉ")
    print(f"{'='*55}")
    print(f"   Accuracy       : {accuracy*100:.2f}%")
    print(f"   F1-Score       : {f1*100:.2f}%")
    print(f"   Total cultures : {len(le.classes_)}")
    print(f"   Total données  : {len(df)} exemples")
    print(f"\n   → Redémarre l'API FastAPI pour utiliser les nouveaux modèles")


if __name__ == "__main__":
    entrainer_v2()