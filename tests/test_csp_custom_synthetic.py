"""
tests/test_csp_custom_synthetic.py

Niveau 1 de validation : données synthétiques où la direction
discriminante est connue à l'avance par construction.

On construit 2 canaux et 2 classes telles que :
    - classe A : forte variance sur le canal 0, faible sur le canal 1
    - classe B : l'inverse

Le CSP doit retrouver ces directions quasi exactement (aux erreurs
numériques et au signe/permutation près, qui n'ont pas d'importance
mathématique ici).

Usage:
    python -m tests.test_csp_custom_synthetic
"""

import numpy as np

from dimensionality_reduction.csp_custom import CSPCustom


def make_synthetic_data(n_epochs_per_class: int = 50, n_times: int = 200, seed: int = 42):
    """
    Génère des epochs synthétiques à 2 canaux, où le canal 0 est très
    variant pour la classe A et peu variant pour la classe B (et
    inversement pour le canal 1).
    """
    rng = np.random.default_rng(seed)

    # Classe A (label 0) : canal 0 fort (std=3), canal 1 faible (std=0.5)
    class_a = np.stack([
        rng.normal(0, 3.0, size=(n_epochs_per_class, n_times)),
        rng.normal(0, 0.5, size=(n_epochs_per_class, n_times)),
    ], axis=1)  # (n_epochs, 2, n_times)

    # Classe B (label 1) : canal 0 faible, canal 1 fort
    class_b = np.stack([
        rng.normal(0, 0.5, size=(n_epochs_per_class, n_times)),
        rng.normal(0, 3.0, size=(n_epochs_per_class, n_times)),
    ], axis=1)

    X = np.concatenate([class_a, class_b], axis=0)
    y = np.concatenate([
        np.zeros(n_epochs_per_class),
        np.ones(n_epochs_per_class),
    ])
    return X, y


def test_csp_recovers_discriminant_axes():
    X, y = make_synthetic_data()

    csp = CSPCustom(n_components=2)
    csp.fit(X, y)
    features = csp.transform(X)

    # --- Vérification 1 : les features doivent séparer nettement les classes ---
    mean_feature_a = features[y == 0].mean(axis=0)
    mean_feature_b = features[y == 1].mean(axis=0)

    print(f"Moyenne des features - classe A : {mean_feature_a}")
    print(f"Moyenne des features - classe B : {mean_feature_b}")

    separation = np.abs(mean_feature_a - mean_feature_b).sum()
    print(f"Séparation totale (somme des écarts absolus) : {separation:.4f}")

    assert separation > 1.0, (
        "Les features CSP ne séparent pas nettement les classes synthétiques : "
        "l'implémentation a probablement un bug."
    )

    # --- Vérification 2 : les filtres doivent privilégier des directions
    #     proches des axes bruts (canal 0 pur ou canal 1 pur), car les
    #     données synthétiques n'ont aucune corrélation croisée entre canaux ---
    filters = csp.filters_  # (n_components, n_channels)
    print(f"Filtres appris (chaque ligne = une composante) :\n{filters}")

    for filt in filters:
        # normalisation pour comparer uniquement la direction, pas l'échelle
        filt_normalized = np.abs(filt) / np.linalg.norm(filt)
        dominant_channel_weight = filt_normalized.max()
        assert dominant_channel_weight > 0.8, (
            "Le filtre appris ne privilégie pas nettement un seul canal, "
            "alors que les données synthétiques sont construites pour cela."
        )

    print("\n✅ Test synthétique réussi : le CSPCustom retrouve les axes discriminants attendus.")


if __name__ == "__main__":
    test_csp_recovers_discriminant_axes()
