"""
dimensionality_reduction/csp_reference.py

Wrapper autour de l'implémentation de référence mne.decoding.CSP.

Isoler cet import ici (plutôt que d'importer mne.decoding.CSP directement
dans build_pipeline.py) permet, à l'étape suivante, de brancher votre
propre implémentation maison (dimensionality_reduction/csp_custom.py)
sans changer la signature attendue par le reste du pipeline.
"""

from mne.decoding import CSP as MNECSP


def get_reference_csp(n_components: int = 4):
    """
    Retourne une instance du CSP de référence MNE, déjà configurée avec
    les hyperparamètres par défaut du projet.

    reg=None : pas de régularisation de la covariance à ce stade (le
    prototypage vise à obtenir un score de référence simple, la
    régularisation pourra être explorée plus tard si nécessaire).
    """
    return MNECSP(n_components=n_components, reg='ledoit_wolf', log=True, norm_trace=False)
