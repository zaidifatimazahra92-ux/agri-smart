"""
Recommandation du type d'engrais selon N, P, K, pH et culture.
Basé sur les recommandations FAO + ONCA Maroc.
"""

# ── Base de données des engrais disponibles ─────────────
ENGRAIS_DB = {
    # Engrais azotés (N)
    'uree': {
        'nom'        : 'Urée',
        'formule'    : '46-0-0',
        'nutriment'  : 'N',
        'concentration': 46,  # % N
        'usage'      : 'Engrais azoté de base, le plus concentré',
        'sol'        : 'Tous types',
        'ph_optimal' : (5.5, 8.0),
        'prix_rel'   : 1.0,  # prix relatif
        'emoji'      : '🟢'
    },
    'ammonitrate': {
        'nom'        : 'Ammonitrate 33.5%',
        'formule'    : '33.5-0-0',
        'nutriment'  : 'N',
        'concentration': 33.5,
        'usage'      : 'Azote à double effet (nitrique + ammoniacal)',
        'sol'        : 'Tous types sauf très acides',
        'ph_optimal' : (6.0, 8.5),
        'prix_rel'   : 1.2,
        'emoji'      : '🟢'
    },
    'sulfate_ammoniaque': {
        'nom'        : 'Sulfate d\'ammonium',
        'formule'    : '21-0-0 + 24S',
        'nutriment'  : 'N',
        'concentration': 21,
        'usage'      : 'Acidifie le sol, idéal pour sols basiques',
        'sol'        : 'Sols calcaires ou basiques (pH > 7)',
        'ph_optimal' : (7.0, 9.0),
        'prix_rel'   : 0.9,
        'emoji'      : '🟢'
    },

    # Engrais phosphatés (P)
    'superphosphate_triple': {
        'nom'        : 'Superphosphate Triple (TSP)',
        'formule'    : '0-45-0',
        'nutriment'  : 'P',
        'concentration': 45,
        'usage'      : 'Phosphore concentré, bonne solubilité',
        'sol'        : 'Sols neutres à légèrement acides',
        'ph_optimal' : (5.5, 7.5),
        'prix_rel'   : 1.3,
        'emoji'      : '🟡'
    },
    'dap': {
        'nom'        : 'DAP (Di-Ammonium Phosphate)',
        'formule'    : '18-46-0',
        'nutriment'  : 'P',
        'concentration': 46,
        'usage'      : 'Double apport N+P au semis, très utilisé',
        'sol'        : 'Tous types',
        'ph_optimal' : (5.5, 8.0),
        'prix_rel'   : 1.4,
        'emoji'      : '🟡'
    },
    'phosphate_naturel': {
        'nom'        : 'Phosphate naturel (Maroc)',
        'formule'    : '0-28-0',
        'nutriment'  : 'P',
        'concentration': 28,
        'usage'      : 'Action lente, idéal sols acides. Produit marocain.',
        'sol'        : 'Sols acides (pH < 6.5)',
        'ph_optimal' : (4.5, 6.5),
        'prix_rel'   : 0.7,
        'emoji'      : '🟡'
    },

    # Engrais potassiques (K)
    'chlorure_potassium': {
        'nom'        : 'Chlorure de potassium (KCl)',
        'formule'    : '0-0-60',
        'nutriment'  : 'K',
        'concentration': 60,
        'usage'      : 'Potassium le plus concentré et économique',
        'sol'        : 'Tous sauf cultures sensibles au chlore',
        'ph_optimal' : (5.5, 8.5),
        'prix_rel'   : 1.0,
        'emoji'      : '🔴'
    },
    'sulfate_potassium': {
        'nom'        : 'Sulfate de potassium (SOP)',
        'formule'    : '0-0-50 + 18S',
        'nutriment'  : 'K',
        'concentration': 50,
        'usage'      : 'Sans chlore, idéal fruits/légumes/tabac',
        'sol'        : 'Tous types, cultures sensibles au chlore',
        'ph_optimal' : (5.0, 8.5),
        'prix_rel'   : 1.8,
        'emoji'      : '🔴'
    },

    # Engrais complets NPK
    'npk_15_15_15': {
        'nom'        : 'Engrais complet NPK 15-15-15',
        'formule'    : '15-15-15',
        'nutriment'  : 'NPK',
        'concentration': 15,
        'usage'      : 'Apport équilibré N+P+K en une seule application',
        'sol'        : 'Tous types',
        'ph_optimal' : (5.5, 8.0),
        'prix_rel'   : 1.2,
        'emoji'      : '🌿'
    },
    'npk_17_17_17': {
        'nom'        : 'Engrais complet NPK 17-17-17',
        'formule'    : '17-17-17',
        'nutriment'  : 'NPK',
        'concentration': 17,
        'usage'      : 'Très utilisé au Maroc pour céréales et maraîchage',
        'sol'        : 'Tous types',
        'ph_optimal' : (5.5, 8.0),
        'prix_rel'   : 1.3,
        'emoji'      : '🌿'
    },
}


def calculer_quantite_produit(nutriment_kg_ha: float,
                              concentration: float) -> float:
    """
    Calcule la quantité de produit commercial à appliquer.

    Exemple : 120 kg N/ha avec Urée 46%
    → 120 / 0.46 = 260.9 kg d'Urée/ha
    """
    if concentration <= 0:
        return 0.0
    return round(nutriment_kg_ha / (concentration / 100), 1)


def recommander_engrais(N_kg_ha: float,
                        P_kg_ha: float,
                        K_kg_ha: float,
                        ph: float,
                        culture: str = None) -> dict:
    """
    Recommande les types d'engrais selon N, P, K, pH et culture.

    Returns:
        dict avec recommandations pour N, P, K et engrais complet
    """

    recommandations = []

    # ── Seuils pour décider si un nutriment est nécessaire ──
    SEUIL_N = 20   # kg/ha minimum pour recommander un engrais N
    SEUIL_P = 15
    SEUIL_K = 15

    # ── Recommandation AZOTE (N) ──────────────────────────
    if N_kg_ha >= SEUIL_N:
        # Choisir selon pH
        if ph > 7.2:
            # Sol basique/calcaire → sulfate d'ammonium pour acidifier
            engrais_n = ENGRAIS_DB['sulfate_ammoniaque']
        elif N_kg_ha > 100:
            # Besoins élevés → Urée (plus concentrée = moins de volume)
            engrais_n = ENGRAIS_DB['uree']
        else:
            # Besoins modérés → Ammonitrate (double effet)
            engrais_n = ENGRAIS_DB['ammonitrate']

        qte_produit = calculer_quantite_produit(N_kg_ha, engrais_n['concentration'])

        recommandations.append({
            'nutriment'      : 'Azote (N)',
            'besoin_kg_ha'   : N_kg_ha,
            'engrais'        : engrais_n['nom'],
            'formule'        : engrais_n['formule'],
            'quantite_kg_ha' : qte_produit,
            'usage'          : engrais_n['usage'],
            'moment'         : _moment_application('N', culture),
            'emoji'          : engrais_n['emoji'],
            'couleur'        : '#2d6a4f'
        })

    # ── Recommandation PHOSPHORE (P) ──────────────────────
    if P_kg_ha >= SEUIL_P:
        # Choisir selon pH et disponibilité locale
        if ph < 6.5:
            # Sol acide → phosphate naturel (marocain, moins cher)
            engrais_p = ENGRAIS_DB['phosphate_naturel']
        elif N_kg_ha >= SEUIL_N:
            # Si on apporte aussi N → DAP (couvre N+P ensemble)
            engrais_p = ENGRAIS_DB['dap']
        else:
            # Sol neutre → TSP (le plus concentré)
            engrais_p = ENGRAIS_DB['superphosphate_triple']

        qte_produit = calculer_quantite_produit(P_kg_ha, engrais_p['concentration'])

        recommandations.append({
            'nutriment'      : 'Phosphore (P)',
            'besoin_kg_ha'   : P_kg_ha,
            'engrais'        : engrais_p['nom'],
            'formule'        : engrais_p['formule'],
            'quantite_kg_ha' : qte_produit,
            'usage'          : engrais_p['usage'],
            'moment'         : _moment_application('P', culture),
            'emoji'          : engrais_p['emoji'],
            'couleur'        : '#e76f51'
        })

    # ── Recommandation POTASSIUM (K) ──────────────────────
    if K_kg_ha >= SEUIL_K:
        # Choisir selon culture (certaines sensibles au chlore)
        cultures_sensibles_chlore = [
            'pomme_de_terre', 'tomate', 'grapes', 'tobacco',
            'coffee', 'papaya', 'strawberry'
        ]
        if culture and culture.lower() in cultures_sensibles_chlore:
            engrais_k = ENGRAIS_DB['sulfate_potassium']
        else:
            engrais_k = ENGRAIS_DB['chlorure_potassium']

        qte_produit = calculer_quantite_produit(K_kg_ha, engrais_k['concentration'])

        recommandations.append({
            'nutriment'      : 'Potassium (K)',
            'besoin_kg_ha'   : K_kg_ha,
            'engrais'        : engrais_k['nom'],
            'formule'        : engrais_k['formule'],
            'quantite_kg_ha' : qte_produit,
            'usage'          : engrais_k['usage'],
            'moment'         : _moment_application('K', culture),
            'emoji'          : engrais_k['emoji'],
            'couleur'        : '#f4a261'
        })

    # ── Suggestion engrais complet si besoins équilibrés ──
    engrais_complet = None
    if N_kg_ha >= SEUIL_N and P_kg_ha >= SEUIL_P and K_kg_ha >= SEUIL_K:
        ratio_n = N_kg_ha / (N_kg_ha + P_kg_ha + K_kg_ha)
        ratio_p = P_kg_ha / (N_kg_ha + P_kg_ha + K_kg_ha)
        ratio_k = K_kg_ha / (N_kg_ha + P_kg_ha + K_kg_ha)

        # Si les ratios sont proches → engrais complet plus économique
        if max(ratio_n, ratio_p, ratio_k) < 0.5:
            npk = ENGRAIS_DB['npk_17_17_17']
            # Quantité basée sur le nutriment le plus limitant
            min_nutriment = min(N_kg_ha, P_kg_ha, K_kg_ha)
            qte_npk = calculer_quantite_produit(min_nutriment, npk['concentration'])
            engrais_complet = {
                'engrais'        : npk['nom'],
                'formule'        : npk['formule'],
                'quantite_kg_ha' : qte_npk,
                'usage'          : npk['usage'],
                'emoji'          : npk['emoji'],
            }

    return {
        'recommandations'  : recommandations,
        'engrais_complet'  : engrais_complet,
        'note_agronomique' : _generer_note(N_kg_ha, P_kg_ha, K_kg_ha, ph, culture)
    }


def _moment_application(nutriment: str, culture: str = None) -> str:
    """Recommande le moment d'application selon le nutriment."""
    if nutriment == 'N':
        return 'Fractionnez en 2-3 apports : 1/3 au semis, 2/3 en végétation'
    elif nutriment == 'P':
        return 'Application au semis ou en pré-semis (phosphore peu mobile)'
    elif nutriment == 'K':
        return 'Au semis ou en début de culture. Évitez périodes de pluie.'
    return 'Selon stade végétatif'


def _generer_note(N, P, K, ph, culture) -> str:
    """Génère une note agronomique personnalisée."""
    notes = []

    if ph > 7.5:
        notes.append("⚠️ Sol basique (pH {:.1f}) : le phosphore est moins disponible. "
                     "Préférez les engrais acidifiants.".format(ph))
    elif ph < 6.0:
        notes.append("⚠️ Sol acide (pH {:.1f}) : évitez l'urée. "
                     "Le sulfate d'ammonium est déconseillé.".format(ph))

    if N > 150:
        notes.append("⚠️ Dose d'azote élevée : fractionnez obligatoirement "
                     "pour éviter les pertes par lessivage.")

    if culture in ['rice', 'maize', 'blé', 'orge']:
        notes.append("💡 Céréale : apportez 50% de N au tallage, 50% à la montaison.")

    if culture in ['tomate', 'watermelon', 'muskmelon']:
        notes.append("💡 Maraîchage : privilégiez la fertigation (engrais solubles).")

    if not notes:
        notes.append("✅ Conditions favorables. Respectez les doses recommandées "
                     "et évitez d'épandre avant les pluies fortes.")

    return ' | '.join(notes)


# ── TEST DIRECT ──────────────────────────────────────────
if __name__ == "__main__":
    result = recommander_engrais(
        N_kg_ha  = 120,
        P_kg_ha  = 60,
        K_kg_ha  = 40,
        ph       = 7.2,
        culture  = 'blé'
    )

    print("="*55)
    print("🌱 RECOMMANDATION ENGRAIS")
    print("="*55)

    for rec in result['recommandations']:
        print(f"\n{rec['emoji']} {rec['nutriment']}")
        print(f"   Besoin         : {rec['besoin_kg_ha']} kg/ha")
        print(f"   Engrais        : {rec['engrais']} ({rec['formule']})")
        print(f"   Quantité       : {rec['quantite_kg_ha']} kg/ha de produit")
        print(f"   Application    : {rec['moment']}")

    if result['engrais_complet']:
        ec = result['engrais_complet']
        print(f"\n{ec['emoji']} Alternative engrais complet :")
        print(f"   {ec['engrais']} ({ec['formule']})")
        print(f"   Quantité : {ec['quantite_kg_ha']} kg/ha")

    print(f"\n📝 Note : {result['note_agronomique']}")