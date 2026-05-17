import requests
import pandas as pd
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


def get_nasa_power_data(latitude, longitude, nb_jours=30):
    """
    Récupère les données météo NASA POWER pour une localisation donnée.
    
    Args:
        latitude  : ex. 33.97 (Rabat, Maroc)
        longitude : ex. -6.85
        nb_jours  : nombre de jours à récupérer (défaut 30)
    
    Returns:
        DataFrame avec colonnes T2M, RH2M, PRECTOT, ALLSKY_SFC_SW_DWN
    """

    # --- Calcul des dates ---
    date_fin    = datetime.today()
    date_debut  = date_fin - timedelta(days=nb_jours)

    start = date_debut.strftime('%Y%m%d')
    end   = date_fin.strftime('%Y%m%d')

    print(f"📡 Récupération NASA POWER : {latitude}, {longitude}")
    print(f"   Période : {start} → {end}")

    # --- Construction de l'URL API ---
    url = (
        "https://power.larc.nasa.gov/api/temporal/daily/point"
        f"?parameters=T2M,RH2M,PRECTOTCORR,ALLSKY_SFC_SW_DWN"
        f"&community=AG"
        f"&longitude={longitude}"
        f"&latitude={latitude}"
        f"&start={start}"
        f"&end={end}"
        f"&format=JSON"
    )

    # --- Appel API ---
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        data = response.json()
    except requests.exceptions.RequestException as e:
        print(f"❌ Erreur API NASA : {e}")
        return None

    # --- Extraction des données ---
    params = data['properties']['parameter']

    df = pd.DataFrame({
        'date'                : list(params['T2M'].keys()),
        'T2M'                 : list(params['T2M'].values()),
        'RH2M'                : list(params['RH2M'].values()),
        'PRECTOT'             : list(params['PRECTOTCORR'].values()),
        'ALLSKY_SFC_SW_DWN'   : list(params['ALLSKY_SFC_SW_DWN'].values()),
    })

    # --- Nettoyage ---
    df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
    df = df.replace(-999.0, pd.NA)   # NASA utilise -999 pour les valeurs manquantes
    df = df.dropna()

    print(f"✅ {len(df)} jours récupérés avec succès")
    return df


def get_meteo_summary(latitude, longitude, nb_jours=30):
    """
    Retourne un résumé météo (moyennes) — ce qu'on passera au modèle ML.
    """
    df = get_nasa_power_data(latitude, longitude, nb_jours)

    if df is None or df.empty:
        return None

    summary = {
        'temperature_moy'   : round(df['T2M'].mean(), 2),
        'humidite_moy'      : round(df['RH2M'].mean(), 2),
        'pluie_totale'      : round(df['PRECTOT'].sum(), 2),
        'pluie_moy_jour'    : round(df['PRECTOT'].mean(), 2),
        'rayonnement_moy'   : round(df['ALLSKY_SFC_SW_DWN'].mean(), 2),
    }

    print("\n📊 Résumé météo 30 jours :")
    for k, v in summary.items():
        print(f"   {k:25s} : {v}")

    return summary


# --- TEST DIRECT ---
if __name__ == "__main__":
    # Test avec Rabat, Maroc
    LAT = 33.97
    LON = -6.85

    print("=" * 50)
    print("TEST NASA POWER API")
    print("=" * 50)

    # Test 1 : données brutes
    df = get_nasa_power_data(LAT, LON, nb_jours=30)
    if df is not None:
        print("\n📋 Aperçu des données :")
        print(df.head(10).to_string(index=False))

    # Test 2 : résumé
    print("\n" + "=" * 50)
    summary = get_meteo_summary(LAT, LON)