import requests
import warnings
warnings.filterwarnings('ignore')


def get_soilgrids_data(latitude, longitude):
    """
    Récupère les données du sol via OpenLandMap (basé sur SoilGrids ISRIC).
    Fallback automatique vers valeurs régionales si API indisponible.
    
    Args:
        latitude  : ex. 33.97
        longitude : ex. -6.85
    Returns:
        dict avec ph, argile, sable, limon, carbone_organique
    """

    print(f"\n🌱 Récupération données sol")
    print(f"   Localisation : {latitude}, {longitude}")

    # --- Tentative 1 : OpenLandMap API ---
    resultat = _get_openlandmap(latitude, longitude)
    if resultat and resultat.get('statut') == 'ok':
        return resultat

    # --- Tentative 2 : SoilGrids REST direct ---
    print("   ⚠️  OpenLandMap indisponible → essai SoilGrids REST...")
    resultat = _get_soilgrids_rest(latitude, longitude)
    if resultat and resultat.get('statut') == 'ok':
        return resultat

    # --- Fallback : valeurs régionales basées sur la zone climatique ---
    print("   ⚠️  APIs sol indisponibles → valeurs régionales estimées")
    return _get_valeurs_regionales(latitude, longitude)


def _get_openlandmap(latitude, longitude):
    """
    OpenLandMap REST API — données sol mondiales.
    """
    try:
        # pH du sol
        url_ph = (
            f"https://api.openlandmap.org/query/point"
            f"?lon={longitude}&lat={latitude}"
            f"&collection=sol"
            f"&variable=phh2o"
        )

        # On utilise leur endpoint de requête point
        base = "https://api.openlandmap.org/query/point"

        variables = {
            'phh2o'  : ('ph',                0.1),
            'clay'   : ('argile_pct',         0.1),
            'sand'   : ('sable_pct',          0.1),
            'silt'   : ('limon_pct',          0.1),
            'soc'    : ('carbone_organique',  0.1),
        }

        resultats = {}
        for var, (nom, facteur) in variables.items():
            resp = requests.get(
                f"{base}?lon={longitude}&lat={latitude}&variable={var}",
                timeout=10
            )
            if resp.status_code == 200:
                data = resp.json()
                val = data.get('result', {}).get('value')
                if val is not None:
                    resultats[nom] = round(float(val) * facteur, 2)

        if len(resultats) >= 3:
            resultats['statut'] = 'ok'
            resultats['source'] = 'openlandmap'
            _afficher_resultats(resultats)
            return resultats

    except Exception:
        pass

    return None


def _get_soilgrids_rest(latitude, longitude):
    """
    SoilGrids v2 avec plusieurs profondeurs.
    """
    try:
        url = "https://rest.isric.org/soilgrids/v2.0/properties/query"
        profondeurs = ['0-5cm', '5-15cm', '15-30cm']

        params = {
            'lon'      : longitude,
            'lat'      : latitude,
            'property' : ['phh2o', 'clay', 'sand', 'silt', 'soc'],
            'depth'    : profondeurs,
            'value'    : 'mean'
        }

        response = requests.get(url, params=params, timeout=20)
        response.raise_for_status()
        data = response.json()

        resultats = {}
        mapping = {
            'phh2o' : ('ph',               10),
            'clay'  : ('argile_pct',       10),
            'sand'  : ('sable_pct',        10),
            'silt'  : ('limon_pct',        10),
            'soc'   : ('carbone_organique', 10),
        }

        for layer in data['properties']['layers']:
            nom = layer['name']
            if nom not in mapping:
                continue
            cle, diviseur = mapping[nom]
            # Prendre la première profondeur non-nulle
            for depth in layer['depths']:
                val = depth['values'].get('mean')
                if val is not None:
                    resultats[cle] = round(val / diviseur, 2)
                    break

        if len(resultats) >= 3:
            resultats['statut'] = 'ok'
            resultats['source'] = 'soilgrids'
            _afficher_resultats(resultats)
            return resultats

    except Exception:
        pass

    return None


def _get_valeurs_regionales(latitude, longitude):
    """
    Estimation des propriétés sol selon la zone géographique.
    Basé sur les données climatiques et pédologiques régionales connues.
    """

    # --- Détermination de la zone ---
    # Maroc / Maghreb / Méditerranée
    if 28 <= latitude <= 37 and -14 <= longitude <= 2:
        zone = "Maghreb méditerranéen"
        valeurs = {
            'ph'                : 7.2,   # sols légèrement calcaires
            'argile_pct'        : 28.0,  # sol argilo-limoneux typique
            'sable_pct'         : 38.0,
            'limon_pct'         : 34.0,
            'carbone_organique' : 8.5,
        }

    # Europe méridionale
    elif 36 <= latitude <= 48 and -10 <= longitude <= 30:
        zone = "Europe méridionale"
        valeurs = {
            'ph'                : 6.8,
            'argile_pct'        : 25.0,
            'sable_pct'         : 42.0,
            'limon_pct'         : 33.0,
            'carbone_organique' : 12.0,
        }

    # Afrique subsaharienne
    elif -10 <= latitude <= 15:
        zone = "Afrique subsaharienne"
        valeurs = {
            'ph'                : 6.2,
            'argile_pct'        : 20.0,
            'sable_pct'         : 55.0,
            'limon_pct'         : 25.0,
            'carbone_organique' : 6.0,
        }

    # Asie du Sud
    elif 8 <= latitude <= 35 and 65 <= longitude <= 100:
        zone = "Asie du Sud"
        valeurs = {
            'ph'                : 6.5,
            'argile_pct'        : 32.0,
            'sable_pct'         : 35.0,
            'limon_pct'         : 33.0,
            'carbone_organique' : 9.0,
        }

    # Valeur globale par défaut
    else:
        zone = "Zone générique"
        valeurs = {
            'ph'                : 6.5,
            'argile_pct'        : 25.0,
            'sable_pct'         : 40.0,
            'limon_pct'         : 35.0,
            'carbone_organique' : 10.0,
        }

    valeurs['statut'] = 'regional'
    valeurs['source'] = f'estimation_{zone}'

    print(f"\n📊 Données sol (estimation régionale — {zone}) :")
    _afficher_resultats(valeurs)

    return valeurs


def _afficher_resultats(resultats):
    """
    Affiche et interprète les données sol.
    """
    print(f"\n📊 Données sol :")
    print(f"   pH                : {resultats.get('ph', 'N/A')}")
    print(f"   Argile            : {resultats.get('argile_pct', 'N/A')} %")
    print(f"   Sable             : {resultats.get('sable_pct', 'N/A')} %")
    print(f"   Limon             : {resultats.get('limon_pct', 'N/A')} %")
    print(f"   Carbone organique : {resultats.get('carbone_organique', 'N/A')} g/kg")
    print(f"   Source            : {resultats.get('source', 'N/A')}")

    # Interprétation pH
    ph = resultats.get('ph', 7)
    if ph < 5.5:
        print(f"   ⚠️  Sol acide — absorption phosphore difficile")
    elif ph <= 7.5:
        print(f"   ✅ pH idéal pour la plupart des cultures")
    else:
        print(f"   ⚠️  Sol basique — carence en fer possible")

    # Type de sol
    argile = resultats.get('argile_pct', 0)
    sable  = resultats.get('sable_pct', 0)
    if argile > 40:
        print(f"   🏔️  Sol argileux — retient bien l'eau")
    elif sable > 60:
        print(f"   🏜️  Sol sableux — se draine rapidement")
    else:
        print(f"   🌱 Sol équilibré (limono-argileux)")


# --- TEST DIRECT ---
if __name__ == "__main__":
    print("=" * 50)
    print("TEST DONNÉES SOL — MULTI-SOURCE")
    print("=" * 50)

    LAT = 33.97
    LON = -6.85

    resultat = get_soilgrids_data(LAT, LON)

    print("\n📦 Données finales retournées :")
    for k, v in resultat.items():
        print(f"   {k:25s} : {v}")