import sys
import os
import pickle
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings('ignore')

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime

# Ajouter le dossier racine au path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.data_collection.nasa_power    import get_meteo_summary
from src.data_collection.earth_engine  import initialiser_ee, get_ndvi
from src.data_collection.soilgrids     import get_soilgrids_data
from src.data_collection.engrais_recommender import recommander_engrais


# ═══════════════════════════════════════════════════════════
# INITIALISATION FASTAPI
# ═══════════════════════════════════════════════════════════

app = FastAPI(
    title       = "🌾 AGRI-SMART API",
    description = "Système intelligent de recommandation irrigation et fertilisation",
    version     = "1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)


# ═══════════════════════════════════════════════════════════
# CHARGEMENT DES MODÈLES AU DÉMARRAGE
# ═══════════════════════════════════════════════════════════

print("⏳ Chargement des modèles...")

with open('models/model_classification.pkl', 'rb') as f:
    MODEL_CLF = pickle.load(f)

with open('models/modeles_regression.pkl', 'rb') as f:
    MODELES_REG = pickle.load(f)

with open('models/label_encoder.pkl', 'rb') as f:
    LABEL_ENCODER = pickle.load(f)

with open('models/scaler.pkl', 'rb') as f:
    SCALER = pickle.load(f)

with open('models/features.pkl', 'rb') as f:
    FEATURES = pickle.load(f)

# Initialiser Earth Engine
try:
    initialiser_ee(project='agri-smart-495220')
    EE_OK = True
except Exception:
    EE_OK = False

print("✅ Modèles chargés et prêts !")


# ═══════════════════════════════════════════════════════════
# SCHÉMAS DE DONNÉES
# ═══════════════════════════════════════════════════════════

class RequetePrediction(BaseModel):
    latitude  : float = Field(..., example=33.97,  description="Latitude GPS")
    longitude : float = Field(..., example=-6.85,  description="Longitude GPS")
    culture   : str   = Field(None, example="blé", description="Culture souhaitée (optionnel)")

class ReponsePrediction(BaseModel):
    culture_recommandee  : str
    confiance_pct        : float
    irrigation_mm        : float
    N_kg_ha              : float
    P_kg_ha              : float
    K_kg_ha              : float
    engrais              : list       # ← NOUVEAU
    engrais_complet      : dict | None  # ← NOUVEAU
    note_agronomique     : str        # ← NOUVEAU
    ndvi                 : float
    temperature          : float
    humidity             : float
    rainfall             : float
    ph                   : float
    source_sol           : str
    timestamp            : str

class RequeteNDVITile(BaseModel):
    latitude  : float = Field(..., example=33.97, description="Latitude GPS")
    longitude : float = Field(..., example=-6.85, description="Longitude GPS")
    polygon   : Optional[List[List[float]]] = Field(
        None,
        example=[[33.97, -6.85], [33.98, -6.86], [33.97, -6.87]],
        description="Liste de points [lat,lon] décrivant la parcelle"
    )


# ═══════════════════════════════════════════════════════════
# DÉTECTION DE ZONE GÉOGRAPHIQUE
# ═══════════════════════════════════════════════════════════

def _detecter_zone(latitude, longitude, temperature, rainfall, ph):
    """
    Estime N, P, K selon la zone agro-écologique.
    Basé sur les recommandations FAO par zone.
    """
    if temperature > 28 and rainfall < 30:
        return {'N': 60,  'P': 40, 'K': 40,  'zone': 'aride_chaud'}

    if 20 <= temperature <= 28 and rainfall < 50 and ph > 7.0:
        return {'N': 50,  'P': 50, 'K': 50,  'zone': 'mediterraneen_sec'}

    if 12 <= temperature <= 20 and 40 <= rainfall <= 80 and ph >= 7.0:
        return {'N': 120, 'P': 60, 'K': 40,  'zone': 'cerealier'}

    if temperature < 22 and rainfall > 50 and ph < 7.5:
        return {'N': 150, 'P': 80, 'K': 60,  'zone': 'maraicher_humide'}

    if 18 <= temperature <= 26 and ph < 7.0:
        return {'N': 120, 'P': 60, 'K': 120, 'zone': 'agrumicole'}

    if 15 <= temperature <= 25 and rainfall < 50 and ph > 7.2:
        return {'N': 60,  'P': 30, 'K': 80,  'zone': 'olivaraie'}

    if temperature > 25 and rainfall > 100:
        return {'N': 80,  'P': 40, 'K': 40,  'zone': 'tropical_humide'}

    if temperature < 15:
        return {'N': 100, 'P': 50, 'K': 50,  'zone': 'tempere'}

    return {'N': 50, 'P': 50, 'K': 50, 'zone': 'generique'}


# ═══════════════════════════════════════════════════════════
# PRÉPARATION DES FEATURES TERRAIN
# ═══════════════════════════════════════════════════════════

def preparer_features_terrain(latitude, longitude):
    """
    Collecte les données terrain et prépare les features pour le modèle.
    """
    features   = {}
    source_sol = 'defaut'

    # ── 1. NASA POWER ──────────────────────────────────
    meteo = get_meteo_summary(latitude, longitude, nb_jours=30)
    if meteo:
        features['temperature'] = meteo['temperature_moy']
        features['humidity']    = meteo['humidite_moy']
        features['rainfall']    = meteo['pluie_totale']
    else:
        features['temperature'] = 25.0
        features['humidity']    = 70.0
        features['rainfall']    = 80.0

    # ── 2. NDVI Earth Engine ────────────────────────────
    if EE_OK:
        try:
            ndvi_data        = get_ndvi(latitude, longitude, nb_jours=30)
            features['ndvi'] = ndvi_data['ndvi_moyen']
        except Exception:
            features['ndvi'] = 0.4
    else:
        features['ndvi'] = 0.4

    # ── 3. SoilGrids ────────────────────────────────────
    sol = get_soilgrids_data(latitude, longitude)
    if sol:
        features['ph']     = sol['ph']
        features['argile'] = sol.get('argile_pct', 25.0)
        features['sable']  = sol.get('sable_pct',  40.0)
        source_sol         = sol.get('source', 'inconnu')
    else:
        features['ph']     = 6.5
        features['argile'] = 25.0
        features['sable']  = 40.0
        source_sol         = 'defaut'

    # ── 4. N, P, K selon zone géographique ─────────────
    zone = _detecter_zone(
        latitude, longitude,
        features['temperature'],
        features['rainfall'],
        features['ph']
    )
    features['N'] = zone['N']
    features['P'] = zone['P']
    features['K'] = zone['K']

    # ── 5. Feature Engineering ──────────────────────────
    total_npk = features['N'] + features['P'] + features['K'] + 1

    features['ET0_estime']      = 0.0023 * (features['temperature'] + 17.8) * 0.5
    features['stress_hydrique'] = max(0, features['ET0_estime'] - features['rainfall'] / 30)
    features['ratio_N']         = features['N'] / total_npk
    features['ratio_P']         = features['P'] / total_npk
    features['ratio_K']         = features['K'] / total_npk
    features['fertilite_idx']   = (
        features['ph'] * 0.3 +
        features['N']  * 0.02 +
        features['P']  * 0.02 +
        features['K']  * 0.02
    )
    features['confort_clim'] = min(1, max(0,
        (1 - abs(features['temperature'] - 25) / 25) * 0.5 +
        (features['humidity'] / 100) * 0.3 +
        (min(features['rainfall'], 200) / 200) * 0.2
    ))
    features['ndvi_estime'] = features['ndvi']

    return features, source_sol


# ═══════════════════════════════════════════════════════════
# ROUTES API
# ═══════════════════════════════════════════════════════════

@app.get("/")
def accueil():
    return {
        "message"  : "🌾 AGRI-SMART API opérationnelle",
        "version"  : "1.0.0",
        "endpoints": ["/predict", "/ndvi_map", "/ndvi_tile", "/health", "/cultures", "/docs"]
    }


@app.get("/health")
def health_check():
    return {
        "status"          : "✅ OK",
        "modeles_charges" : True,
        "earth_engine"    : EE_OK,
        "timestamp"       : datetime.now().isoformat()
    }


@app.get("/cultures")
def liste_cultures():
    return {
        "cultures" : list(LABEL_ENCODER.classes_),
        "total"    : len(LABEL_ENCODER.classes_)
    }


@app.post("/predict", response_model=ReponsePrediction)
def predire(requete: RequetePrediction):
    """
    Endpoint principal : reçoit GPS + culture optionnelle,
    collecte les données terrain, retourne les recommandations.
    """
    try:
        print(f"\n🌍 Nouvelle requête : {requete.latitude}, {requete.longitude}")

        # 1. Collecter et préparer les features
        features_dict, source_sol = preparer_features_terrain(
            requete.latitude, requete.longitude
        )

        # 2. DataFrame dans le bon ordre
        X = pd.DataFrame([features_dict])[FEATURES]

        # 3. Normaliser
        X_scaled = SCALER.transform(X)

        # 4. Prédiction culture
        culture_code = MODEL_CLF.predict(X_scaled)[0]
        probas       = MODEL_CLF.predict_proba(X_scaled)[0]
        confiance    = round(float(probas.max()) * 100, 2)
        culture_nom  = LABEL_ENCODER.inverse_transform([culture_code])[0]

        # 5. Prédiction N, P, K, irrigation
        N   = round(float(MODELES_REG['N'].predict(X_scaled)[0]),        2)
        P   = round(float(MODELES_REG['P'].predict(X_scaled)[0]),        2)
        K   = round(float(MODELES_REG['K'].predict(X_scaled)[0]),        2)
        irr = round(float(MODELES_REG['rainfall'].predict(X_scaled)[0]), 2)

        print(f"✅ Recommandation : {culture_nom} ({confiance}%)")
        print(f"   N={N}  P={P}  K={K}  Irrigation={irr}mm")

        reco_engrais = recommander_engrais(
            N_kg_ha  = N,
            P_kg_ha  = P,
            K_kg_ha  = K,
            ph       = features_dict['ph'],
            culture  = culture_nom
        )

        return ReponsePrediction(
            culture_recommandee = culture_nom,
            confiance_pct       = confiance,
            irrigation_mm       = irr,
            N_kg_ha             = N,
            P_kg_ha             = P,
            K_kg_ha             = K,
            engrais             = reco_engrais.get('recommandations', []),
            engrais_complet     = reco_engrais.get('engrais_complet'),
            note_agronomique    = reco_engrais.get('note_agronomique', ''),
            ndvi                = features_dict['ndvi'],
            temperature         = features_dict['temperature'],
            humidity            = features_dict['humidity'],
            rainfall            = features_dict['rainfall'],
            ph                  = features_dict['ph'],
            source_sol          = source_sol,
            timestamp           = datetime.now().isoformat()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ═══════════════════════════════════════════════════════════
# NOUVEAU ENDPOINT — CARTE NDVI COLORÉE
# ═══════════════════════════════════════════════════════════

@app.post("/ndvi_map")
def carte_ndvi(requete: RequetePrediction):
    """
    Retourne une grille 5×5 de valeurs NDVI autour du point GPS.
    Utilisé pour la visualisation colorée sur la carte du dashboard.
    Chaque point a une couleur selon son taux de végétation.
    """
    if not EE_OK:
        raise HTTPException(status_code=503, detail="Earth Engine non disponible")

    try:
        import ee
        from datetime import timedelta

        date_fin   = datetime.today()
        date_debut = date_fin - timedelta(days=30)

        point = ee.Geometry.Point([requete.longitude, requete.latitude])
        zone  = point.buffer(2000)  # 2km autour du centre

        # Charger la collection Sentinel-2
        collection = (
            ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
            .filterBounds(zone)
            .filterDate(
                date_debut.strftime('%Y-%m-%d'),
                date_fin.strftime('%Y-%m-%d')
            )
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30))
        )

        nb_images = collection.size().getInfo()
        print(f"🛰️  NDVI Map — {nb_images} images disponibles")

        if nb_images == 0:
            return {
                "error"    : "Aucune image satellite disponible (trop de nuages)",
                "points"   : [],
                "nb_points": 0
            }

        # Calculer NDVI médian
        def calc_ndvi(image):
            return image.normalizedDifference(['B8', 'B4']).rename('NDVI')

        ndvi_median = collection.map(calc_ndvi).median()

        # Grille 5×5 = 25 points autour du centre
        points = []
        step   = 0.007   # ~1km entre chaque point

        for i in np.linspace(-3, 3, 7):
            for j in np.linspace(-3, 3, 7):
                lat_pt = requete.latitude  + i * step
                lon_pt = requete.longitude + j * step

                pt  = ee.Geometry.Point([lon_pt, lat_pt])
                val = ndvi_median.reduceRegion(
                    reducer   = ee.Reducer.mean(),
                    geometry  = pt.buffer(300),
                    scale     = 10,
                    maxPixels = 1e6
                ).getInfo()

                ndvi_val = val.get('NDVI')
                if ndvi_val is not None:
                    points.append({
                        'lat'  : round(lat_pt, 6),
                        'lon'  : round(lon_pt, 6),
                        'ndvi' : round(float(ndvi_val), 4)
                    })

        print(f"✅ NDVI Map — {len(points)} points calculés")

        return {
            "centre_lat" : requete.latitude,
            "centre_lon" : requete.longitude,
            "nb_images"  : nb_images,
            "nb_points"  : len(points),
            "points"     : points
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ndvi_tile")
def carte_ndvi_tile(requete: RequeteNDVITile):
    """
    Retourne une couche de tuiles NDVI tronquée à la parcelle dessinée.
    Utilisé pour un rendu couleur sur la carte avec Google Earth Engine.
    """
    if not EE_OK:
        raise HTTPException(status_code=503, detail="Earth Engine non disponible")

    try:
        import ee
        from datetime import timedelta

        date_fin   = datetime.today()
        date_debut = date_fin - timedelta(days=30)

        centre = ee.Geometry.Point([requete.longitude, requete.latitude])
        region = centre.buffer(2000)

        zone = region
        if requete.polygon:
            coords = [[ [pt[1], pt[0]] for pt in requete.polygon ]]
            if coords[0][0] != coords[0][-1]:
                coords[0].append(coords[0][0])
            zone = ee.Geometry.Polygon(coords)

        collection = (
            ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
            .filterBounds(zone)
            .filterDate(
                date_debut.strftime('%Y-%m-%d'),
                date_fin.strftime('%Y-%m-%d')
            )
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30))
        )

        nb_images = collection.size().getInfo()
        if nb_images == 0:
            raise HTTPException(status_code=404, detail="Aucune image satellite disponible (trop de nuages)")

        def calc_ndvi(image):
            return image.normalizedDifference(['B8', 'B4']).rename('NDVI')

        ndvi_median = collection.map(calc_ndvi).median().clip(zone)

        mean_result = ndvi_median.reduceRegion(
            reducer   = ee.Reducer.mean(),
            geometry  = zone,
            scale     = 10,
            maxPixels = 1e9
        ).getInfo()

        ndvi_mean = mean_result.get('NDVI')
        if ndvi_mean is None:
            raise HTTPException(status_code=500, detail="Impossible de calculer la moyenne NDVI")

        vis_params = {
            'min'    : 0.0,
            'max'    : 0.8,
            'palette': ['#d73027','#fdae61','#ffffbf','#a6d96a','#1a9850','#006837']
        }
        map_id = ndvi_median.getMapId(vis_params)
        tile_url = map_id['tile_fetcher'].url_format

        return {
            'tile_url'  : tile_url,
            'nb_images' : nb_images,
            'ndvi_mean' : round(float(ndvi_mean), 4),
            'min'       : vis_params['min'],
            'max'       : vis_params['max'],
            'palette'   : vis_params['palette']
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))