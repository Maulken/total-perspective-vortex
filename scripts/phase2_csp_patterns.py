# """
# scripts/phase2_csp_patterns.py

# Étape 2 du plan d'action Phase 2 : vérifier que le CSP de référence
# retrouve une structure spatiale cohérente avec la topographie observée
# en Phase 1 (asymétrie C3/C4 attendue en imagerie motrice).

# On visualise ici les PATTERNS du CSP (csp.patterns_), pas les filtres
# (csp.filters_) : les patterns représentent la contribution de chaque
# canal à une composante, et sont directement interprétables en topomap,
# contrairement aux filtres qui servent uniquement à la projection.

# Usage:
#     python3 -m scripts.phase2_csp_patterns --subjects 1 --runs 4 8 12
# """

# import argparse

# import matplotlib.pyplot as plt
# import mne

# from data.loader import load_raw
# from preprocessing.filtering import filter_raw
# from preprocessing.epoching import make_epochs
# from dimensionality_reduction.csp_reference import get_reference_csp

# mne.set_log_level("WARNING")


# def plot_csp_patterns(csp, info: mne.Info, n_components: int) -> None:
#     """
#     Affiche la topographie des patterns CSP pour les composantes
#     retenues. Les premières composantes sont les plus discriminantes
#     pour une classe, les dernières pour l'autre.
#     """
#     # patterns_ a la forme (n_channels, n_channels) ; on ne garde que
#     # les n_components colonnes réellement utilisées par transform().
#     patterns = csp.patterns_[:n_components]

#     fig, axes = plt.subplots(1, n_components, figsize=(4 * n_components, 4))
#     if n_components == 1:
#         axes = [axes]

#     for idx, ax in enumerate(axes):
#         mne.viz.plot_topomap(patterns[idx], info, axes=ax, show=False)
#         ax.set_title(f"Pattern CSP #{idx}")

#     fig.suptitle("Topographie des patterns CSP (à comparer à la Phase 1)")


# def main():
#     parser = argparse.ArgumentParser(description="Phase 2 - Vérification physiologique du CSP")
#     parser.add_argument("--subjects", type=int, nargs="+", default=[1])
#     parser.add_argument("--runs", type=int, nargs="+", default=[4, 8, 12])
#     parser.add_argument("--dataset", default="physionet")
#     parser.add_argument("--n-components", type=int, default=4)
#     args = parser.parse_args()

#     print(f"Chargement sujets={args.subjects}, runs={args.runs} ...")
#     raw = load_raw(args.subjects, args.runs, dataset_name=args.dataset)
#     raw_filtered = filter_raw(raw)
#     epochs = make_epochs(raw_filtered)

#     print(f"Nombre d'epochs : {len(epochs)}")
#     print(f"Classes : {epochs.event_id}")

#     X = epochs.get_data()
#     y = epochs.events[:, -1]

#     # On entraîne le CSP sur l'intégralité des données ici : l'objectif
#     # n'est pas d'évaluer une accuracy (déjà fait à l'étape 1), mais
#     # d'inspecter la structure spatiale apprise.
#     csp = get_reference_csp(n_components=args.n_components)
#     csp.fit(X, y)

#     plot_csp_patterns(csp, epochs.info, n_components=args.n_components)

#     print("Comparez ces topographies à celles obtenues en Phase 1 "
#           "(topomap de puissance mu par classe) : cherchez une "
#           "asymétrie cohérente sur les électrodes centrales (C3/C4).")

#     plt.show()


# if __name__ == "__main__":
#     main()

"""
scripts/phase2_csp_patterns.py

Étape 2 du plan d'action Phase 2 : vérifier que le CSP de référence
retrouve une structure spatiale cohérente avec la topographie observée
en Phase 1 (asymétrie C3/C4 attendue en imagerie motrice).

On visualise ici les PATTERNS du CSP (csp.patterns_), pas les filtres
(csp.filters_) : les patterns représentent la contribution de chaque
canal à une composante, et sont directement interprétables en topomap,
contrairement aux filtres qui servent uniquement à la projection.

Usage:
    python -m scripts.phase2_csp_patterns --subjects 1 --runs 4 8 12
"""

import argparse

import matplotlib.pyplot as plt
import mne

from data.loader import load_raw
from preprocessing.filtering import filter_raw
from preprocessing.epoching import make_epochs
from dimensionality_reduction.csp_reference import get_reference_csp

mne.set_log_level("WARNING")


def plot_csp_patterns(csp, info: mne.Info, n_components: int) -> None:
    """
    Affiche la topographie des patterns CSP pour les composantes
    retenues. Les premières composantes sont les plus discriminantes
    pour une classe, les dernières pour l'autre.
    """
    # patterns_ a la forme (n_channels, n_channels) ; on ne garde que
    # les n_components colonnes réellement utilisées par transform().
    patterns = csp.patterns_[:n_components]

    fig, axes = plt.subplots(1, n_components, figsize=(4 * n_components, 4))
    if n_components == 1:
        axes = [axes]

    for idx, ax in enumerate(axes):
        mne.viz.plot_topomap(patterns[idx], info, axes=ax, show=False)
        ax.set_title(f"Pattern CSP #{idx}")

    fig.suptitle("Topographie des patterns CSP (à comparer à la Phase 1)")


def main():
    parser = argparse.ArgumentParser(description="Phase 2 - Vérification physiologique du CSP")
    parser.add_argument("--subjects", type=int, nargs="+", default=[1])
    parser.add_argument("--runs", type=int, nargs="+", default=[4, 8, 12])
    parser.add_argument("--dataset", default="physionet")
    parser.add_argument("--n-components", type=int, default=4)
    args = parser.parse_args()

    print(f"Chargement sujets={args.subjects}, runs={args.runs} ...")
    if len(args.subjects) > 1:
        print(f"⚠️ Ce script travaille sur UN SEUL sujet à la fois. "
              f"Seul le premier ({args.subjects[0]}) sera utilisé, "
              f"les autres ({args.subjects[1:]}) sont ignorés.")
    raw = load_raw(args.subjects[0], args.runs, dataset_name=args.dataset)
    raw_filtered = filter_raw(raw)
    epochs = make_epochs(raw_filtered)

    print(f"Nombre d'epochs : {len(epochs)}")
    print(f"Classes : {epochs.event_id}")

    X = epochs.get_data()
    y = epochs.events[:, -1]

    # On entraîne le CSP sur l'intégralité des données ici : l'objectif
    # n'est pas d'évaluer une accuracy (déjà fait à l'étape 1), mais
    # d'inspecter la structure spatiale apprise.
    csp = get_reference_csp(n_components=args.n_components)
    csp.fit(X, y)

    plot_csp_patterns(csp, epochs.info, n_components=args.n_components)

    print("Comparez ces topographies à celles obtenues en Phase 1 "
          "(topomap de puissance mu par classe) : cherchez une "
          "asymétrie cohérente sur les électrodes centrales (C3/C4).")

    plt.show()


if __name__ == "__main__":
    main()
