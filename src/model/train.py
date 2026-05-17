import pandas as pd
import numpy as np
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import LabelEncoder, StandardScaler
from sklearn.metrics         import (accuracy_score, f1_score,
                                     classification_report,
                                     mean_absolute_error,
                                     mean_squared_error, r2_score)
import xgboost as xgb


# ═══════════════════════════════════════════════════════════
# 1. CHARGEMENT DES DONNÉES
# ═══════════════════════════════════════════════════════════

def charger_donnees(chemin='data/processed/dataset_final.csv'):
    print("="*55)
    print("📊 CHARGEMENT DES DONNÉES")
    print("="*55)

    df = pd.read_csv(chemin)
    print(f"✅ {len(df)} exemples chargés, {df.shape[1]} colonnes")
    return df


# ═══════════════════════════════════════════════════════════
# 2. PRÉPARATION DES FEATURES
# ═══════════════════════════════════════════════════════════

def preparer_features(df):
    print("\n" + "="*55)
    print("⚙️  PRÉPARATION DES FEATURES")
    print("="*55)

    # Features utilisées pour l'entraînement
    features = [
        'N', 'P', 'K',
        'temperature', 'humidity', 'ph', 'rainfall',
        'ET0_estime', 'stress_hydrique',
        'ratio_N', 'ratio_P', 'ratio_K',
        'fertilite_idx', 'confort_clim', 'ndvi_estime'
    ]

    X = df[features]
    y = df['label']

    print(f"✅ Features sélectionnées : {len(features)}")
    for f in features:
        print(f"   - {f}")

    # Encodage de la cible
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    print(f"\n✅ {le.classes_.shape[0]} cultures encodées")

    # Normalisation
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=features)
    print("✅ Normalisation appliquée (StandardScaler)")

    return X_scaled, y_encoded, le, scaler, features


# ═══════════════════════════════════════════════════════════
# 3. SPLIT TRAIN / TEST
# ═══════════════════════════════════════════════════════════

def split_donnees(X, y):
    print("\n" + "="*55)
    print("✂️  DIVISION TRAIN / TEST")
    print("="*55)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size    = 0.2,
        random_state = 42,
        stratify     = y      # garder la proportion de chaque culture
    )

    print(f"✅ Entraînement : {len(X_train)} exemples (80%)")
    print(f"✅ Test         : {len(X_test)} exemples (20%)")
    return X_train, X_test, y_train, y_test


# ═══════════════════════════════════════════════════════════
# 4. ENTRAÎNEMENT XGBOOST — CLASSIFICATION (culture)
# ═══════════════════════════════════════════════════════════

def entrainer_classification(X_train, X_test, y_train, y_test, le):
    print("\n" + "="*55)
    print("🤖 ENTRAÎNEMENT XGBOOST — CLASSIFICATION")
    print("    (Recommandation de culture)")
    print("="*55)

    model = xgb.XGBClassifier(
        n_estimators      = 300,
        max_depth         = 6,
        learning_rate     = 0.1,
        subsample         = 0.8,
        colsample_bytree  = 0.8,
        random_state      = 42,
        eval_metric       = 'mlogloss',
        verbosity         = 0
    )

    print("⏳ Entraînement en cours...")
    model.fit(
        X_train, y_train,
        eval_set              = [(X_test, y_test)],
        verbose               = False
    )
    print("✅ Entraînement terminé !")

    # --- Évaluation ---
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    f1       = f1_score(y_test, y_pred, average='weighted')

    print(f"\n📊 RÉSULTATS :")
    print(f"   Accuracy  : {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"   F1-Score  : {f1:.4f} ({f1*100:.2f}%)")

    print(f"\n📋 Rapport détaillé par culture :")
    print(classification_report(
        y_test, y_pred,
        target_names = le.classes_
    ))

    # --- Importance des features ---
    print("📊 Importance des features :")
    importances = pd.Series(
        model.feature_importances_,
        index=X_train.columns
    ).sort_values(ascending=False)

    for feat, score in importances.items():
        barre = "█" * int(score * 100)
        print(f"   {feat:20s} : {barre} {score:.4f}")

    return model, accuracy, f1


# ═══════════════════════════════════════════════════════════
# 5. ENTRAÎNEMENT XGBOOST — RÉGRESSION (N, P, K, irrigation)
# ═══════════════════════════════════════════════════════════

def entrainer_regression(X_train, X_test, y_train, y_test, df, features):
    print("\n" + "="*55)
    print("🤖 ENTRAÎNEMENT XGBOOST — RÉGRESSION")
    print("    (Quantités N, P, K et irrigation)")
    print("="*55)

    # Récupérer les index train/test
    idx_train = X_train.index
    idx_test  = X_test.index

    modeles_regression = {}
    cibles = {
        'N'          : 'Azote nécessaire (kg/ha)',
        'P'          : 'Phosphore nécessaire (kg/ha)',
        'K'          : 'Potassium nécessaire (kg/ha)',
        'rainfall'   : 'Irrigation nécessaire (mm)',
    }

    for cible, description in cibles.items():
        print(f"\n  📈 Modèle → {description}")

        y_reg_train = df.loc[idx_train, cible].values
        y_reg_test  = df.loc[idx_test,  cible].values

        model_reg = xgb.XGBRegressor(
            n_estimators     = 300,
            max_depth        = 6,
            learning_rate    = 0.1,
            subsample        = 0.8,
            colsample_bytree = 0.8,
            random_state     = 42,
            verbosity        = 0
        )

        model_reg.fit(X_train, y_reg_train, verbose=False)
        y_pred_reg = model_reg.predict(X_test)

        mae  = mean_absolute_error(y_reg_test, y_pred_reg)
        rmse = np.sqrt(mean_squared_error(y_reg_test, y_pred_reg))
        r2   = r2_score(y_reg_test, y_pred_reg)

        print(f"     MAE  : {mae:.3f}")
        print(f"     RMSE : {rmse:.3f}")
        print(f"     R²   : {r2:.4f} ({r2*100:.2f}%)")

        modeles_regression[cible] = model_reg

    return modeles_regression


# ═══════════════════════════════════════════════════════════
# 6. SAUVEGARDE DES MODÈLES
# ═══════════════════════════════════════════════════════════

def sauvegarder_modeles(model_clf, modeles_reg, le, scaler, features):
    print("\n" + "="*55)
    print("💾 SAUVEGARDE DES MODÈLES")
    print("="*55)

    os.makedirs('models', exist_ok=True)

    # Modèle classification
    with open('models/model_classification.pkl', 'wb') as f:
        pickle.dump(model_clf, f)
    print("✅ models/model_classification.pkl")

    # Modèles régression
    with open('models/modeles_regression.pkl', 'wb') as f:
        pickle.dump(modeles_reg, f)
    print("✅ models/modeles_regression.pkl")

    # LabelEncoder
    with open('models/label_encoder.pkl', 'wb') as f:
        pickle.dump(le, f)
    print("✅ models/label_encoder.pkl")

    # Scaler
    with open('models/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    print("✅ models/scaler.pkl")

    # Features list
    with open('models/features.pkl', 'wb') as f:
        pickle.dump(features, f)
    print("✅ models/features.pkl")

    print("\n✅ Tous les modèles sauvegardés dans models/")


# ═══════════════════════════════════════════════════════════
# 7. PIPELINE COMPLET
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("="*55)
    print("🚀 PIPELINE ENTRAÎNEMENT AGRI-SMART")
    print("="*55)

    # 1. Charger
    df = charger_donnees()

    # 2. Préparer
    X, y, le, scaler, features = preparer_features(df)

    # 3. Splitter
    X_train, X_test, y_train, y_test = split_donnees(X, y)

    # 4. Classification
    model_clf, accuracy, f1 = entrainer_classification(
        X_train, X_test, y_train, y_test, le
    )

    # 5. Régression
    modeles_reg = entrainer_regression(
        X_train, X_test, y_train, y_test, df, features
    )

    # 6. Sauvegarder
    sauvegarder_modeles(model_clf, modeles_reg, le, scaler, features)

    # ── Résumé final ────────────────────────────────────────
    print("\n" + "="*55)
    print("🎉 ENTRAÎNEMENT TERMINÉ")
    print("="*55)
    print(f"   Classification accuracy : {accuracy*100:.2f}%")
    print(f"   F1-Score                : {f1*100:.2f}%")
    print(f"   Modèles sauvegardés     : models/")
    print(f"\n   Prochaine étape → FastAPI 🚀")