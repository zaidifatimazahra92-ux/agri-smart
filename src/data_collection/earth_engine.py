import ee
import os
import warnings
warnings.filterwarnings('ignore')


def initialiser_ee(project='agri-smart-495220'):
    """
    Initialise Earth Engine.
    - En local (dev)  : utilise les credentials utilisateur (~/.config/earthengine)
    - En production   : utilise le Service Account (variable d'environnement Railway)
    """
    try:
        # Chercher la clé Service Account dans les variables d'environnement
        sa_key = os.environ.get('EE_SERVICE_ACCOUNT_KEY')

        if sa_key:
            # ── Production Railway ──────────────────────────────
            import json
            key_data = json.loads(sa_key)
            credentials = ee.ServiceAccountCredentials(
                email    = key_data['client_email'],
                key_data = sa_key
            )
            ee.Initialize(credentials=credentials, project=project)
            print("✅ Earth Engine initialisé (Service Account)")

        elif os.path.exists('service_account.json'):
            # ── Développement avec fichier local ────────────────
            import json
            with open('service_account.json') as f:
                key_data = json.load(f)
            credentials = ee.ServiceAccountCredentials(
                email    = key_data['client_email'],
                key_file = 'service_account.json'
            )
            ee.Initialize(credentials=credentials, project=project)
            print("✅ Earth Engine initialisé (service_account.json)")

        else:
            # ── Développement local classique ───────────────────
            ee.Initialize(project=project)
            print("✅ Earth Engine initialisé (credentials utilisateur)")

    except Exception as e:
        print(f"❌ Erreur initialisation EE : {e}")
        raise


def get_ndvi(latitude, longitude, nb_jours=30):
    """
    Calcule le NDVI moyen via Sentinel-2 (Copernicus) sur les 30 derniers jours.

    Args:
        latitude  : ex. 33.97
        longitude : ex. -6.85
        nb_jours  : période de calcul

    Returns:
        dict avec ndvi_moyen, nb_images, statut
    """
    from datetime import datetime, timedelta

    date_fin   = datetime.today()
    date_debut = date_fin - timedelta(days=nb_jours)

    start = date_debut.strftime('%Y-%m-%d')
    end   = date_fin.strftime('%Y-%m-%d')

    print(f"\n🛰️  Calcul NDVI Sentinel-2")
    print(f"   Localisation : {latitude}, {longitude}")
    print(f"   Période      : {start} → {end}")

    # --- Définir le point géographique ---
    point = ee.Geometry.Point([longitude, latitude])

    # --- Zone autour du point (1km de rayon) ---
    zone = point.buffer(1000)

    # --- Charger la collection Sentinel-2 ---
    collection = (
        ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
        .filterBounds(zone)
        .filterDate(start, end)
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30))
    )

    # --- Compter le nombre d'images disponibles ---
    nb_images = collection.size().getInfo()
    print(f"   Images trouvées : {nb_images} (nuages < 30%)")

    if nb_images == 0:
        print("⚠️  Aucune image disponible — nuages ou zone sans données")
        print("   → Utilisation d'une valeur NDVI par défaut (0.4)")
        return {
            'ndvi_moyen' : 0.4,
            'nb_images'  : 0,
            'statut'     : 'defaut'
        }

    # --- Fonction de calcul NDVI sur une image ---
    def calculer_ndvi(image):
        ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
        return image.addBands(ndvi)

    # --- Appliquer le calcul sur toute la collection ---
    collection_ndvi = collection.map(calculer_ndvi)

    # --- NDVI médian (plus robuste que la moyenne face aux nuages) ---
    ndvi_median = collection_ndvi.select('NDVI').median()

    # --- Extraire la valeur au point ---
    valeur = ndvi_median.reduceRegion(
        reducer   = ee.Reducer.mean(),
        geometry  = zone,
        scale     = 10,
        maxPixels = 1e9
    ).getInfo()

    ndvi_val = valeur.get('NDVI')

    if ndvi_val is None:
        print("⚠️  NDVI non calculable pour cette zone")
        return {
            'ndvi_moyen' : 0.4,
            'nb_images'  : nb_images,
            'statut'     : 'non_calculable'
        }

    ndvi_val = round(ndvi_val, 4)

    # --- Interprétation ---
    if ndvi_val > 0.6:
        interpretation = "🌿 Végétation très dense et saine"
    elif ndvi_val > 0.3:
        interpretation = "🌱 Végétation modérée"
    elif ndvi_val > 0:
        interpretation = "🏜️  Végétation faible / sol partiellement nu"
    else:
        interpretation = "💧 Sol nu / eau / zone urbaine"

    print(f"\n📊 Résultat NDVI :")
    print(f"   NDVI moyen     : {ndvi_val}")
    print(f"   Interprétation : {interpretation}")
    print(f"   Nb images      : {nb_images}")
    print(f"   ✅ Calcul réussi")

    return {
        'ndvi_moyen'     : ndvi_val,
        'nb_images'      : nb_images,
        'interpretation' : interpretation,
        'statut'         : 'ok'
    }


# --- TEST DIRECT ---
if __name__ == "__main__":
    print("=" * 50)
    print("TEST GOOGLE EARTH ENGINE — NDVI")
    print("=" * 50)

    initialiser_ee(project='agri-smart-495220')

    LAT = 33.97
    LON = -6.85

    resultat = get_ndvi(LAT, LON, nb_jours=30)

    print("\n📦 Données retournées :")
    for k, v in resultat.items():
        print(f"   {k:20s} : {v}")