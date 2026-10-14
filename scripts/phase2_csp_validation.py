# """
# scripts/phase2_csp_validation.py

# Étape 4 du plan d'action - niveaux 2 et 3 de validation :

#     Niveau 2 : comparaison visuelle des patterns appris par csp_custom
#                et csp_reference sur les mêmes données réelles
#     Niveau 3 : comparaison de l'accuracy (cross_val_score) des deux
#                pipelines, sur le même split de données

# Usage:
#     python -m scripts.phase2_csp_validation --subjects 1 --runs 4 8 12
# """

# import argparse

# import matplotlib.pyplot as plt
# import mne
# import numpy as np
# from sklearn.model_selection import cross_val_score

# from data.loader import load_raw
# from preprocessing.filtering import filter_raw
# from preprocessing.epoching import make_epochs
# from pipeline.build_pipeline import build_pipeline
# from dimensionality_reduction.csp_reference import get_reference_csp
# from dimensionality_reduction.csp_custom import CSPCustom

# mne.set_log_level("WARNING")


# def compare_accuracy(X, y, n_components: int, cv: int) -> None:
#     """Niveau 3 : compare cross_val_score entre les deux implémentations."""
#     pipeline_ref = build_pipeline(reducer_method="csp_reference", classifier_method="lda",
#                                    n_components=n_components)
#     pipeline_custom = build_pipeline(reducer_method="csp_custom", classifier_method="lda",
#                                       n_components=n_components)

#     scores_ref = cross_val_score(pipeline_ref, X, y, cv=cv)
#     scores_custom = cross_val_score(pipeline_custom, X, y, cv=cv)

#     print("\n--- Comparaison d'accuracy (cross_val_score) ---")
#     print(f"CSP référence (MNE) : {np.round(scores_ref, 4)} | moyenne = {scores_ref.mean():.4f}")
#     print(f"CSP maison           : {np.round(scores_custom, 4)} | moyenne = {scores_custom.mean():.4f}")

#     ecart = abs(scores_ref.mean() - scores_custom.mean())
#     print(f"Écart de moyenne : {ecart:.4f}")

#     if ecart < 0.10:
#         print("→ Écart raisonnable : l'implémentation maison est cohérente avec la référence.")
#     else:
#         print("→ ⚠️ Écart important : vérifier l'implémentation avant de la considérer validée.")


# def compare_patterns(X, y, info: mne.Info, n_components: int) -> None:
#     """Niveau 2 : compare visuellement les patterns des deux CSP."""
#     csp_ref = get_reference_csp(n_components=n_components)
#     csp_ref.fit(X, y)

#     csp_custom = CSPCustom(n_components=n_components)
#     csp_custom.fit(X, y)

#     fig, axes = plt.subplots(2, n_components, figsize=(4 * n_components, 8))

#     for idx in range(n_components):
#         mne.viz.plot_topomap(csp_ref.patterns_[idx], info, axes=axes[0, idx], show=False)
#         axes[0, idx].set_title(f"Référence - pattern #{idx}")

#         mne.viz.plot_topomap(csp_custom.patterns_[idx], info, axes=axes[1, idx], show=False)
#         axes[1, idx].set_title(f"Maison - pattern #{idx}")

#     fig.suptitle("Comparaison des patterns CSP : référence (haut) vs maison (bas)")


# def main():
#     parser = argparse.ArgumentParser(description="Phase 2 - Validation comparative CSP")
#     parser.add_argument("--subjects", type=int, nargs="+", default=[1])
#     parser.add_argument("--runs", type=int, nargs="+", default=[4, 8, 12])
#     parser.add_argument("--dataset", default="physionet")
#     parser.add_argument("--n-components", type=int, default=4)
#     parser.add_argument("--cv", type=int, default=5)
#     parser.add_argument("--no-plot", action="store_true")
#     args = parser.parse_args()

#     print(f"Chargement sujets={args.subjects}, runs={args.runs} ...")
#     raw = load_raw(args.subjects, args.runs, dataset_name=args.dataset)
#     raw_filtered = filter_raw(raw)
#     epochs = make_epochs(raw_filtered)

#     print(f"Nombre d'epochs : {len(epochs)}")
#     print(f"Classes : {epochs.event_id}")

#     X = epochs.get_data()
#     y = epochs.events[:, -1]

#     if len(np.unique(y)) != 2:
#         print(f"⚠️ {len(np.unique(y))} classes détectées au lieu de 2 — "
#               f"CSPCustom lèvera une erreur. Vérifiez make_epochs() avant de continuer.")
#         return

#     compare_accuracy(X, y, n_components=args.n_components, cv=args.cv)

#     if not args.no_plot:
#         compare_patterns(X, y, epochs.info, n_components=args.n_components)
#         plt.show()


# if __name__ == "__main__":
#     main()

"""
scripts/phase2_csp_validation.py

Étape 4 du plan d'action - niveaux 2 et 3 de validation :

    Niveau 2 : comparaison visuelle des patterns appris par csp_custom
               et csp_reference sur les mêmes données réelles
    Niveau 3 : comparaison de l'accuracy (cross_val_score) des deux
               pipelines, sur le même split de données

Usage:
    python -m scripts.phase2_csp_validation --subjects 1 --runs 4 8 12
"""

import argparse

import matplotlib.pyplot as plt
import mne
import numpy as np
from sklearn.model_selection import cross_val_score

from data.loader import load_raw
from preprocessing.filtering import filter_raw
from preprocessing.epoching import make_epochs
from pipeline.build_pipeline import build_pipeline
from dimensionality_reduction.csp_reference import get_reference_csp
from dimensionality_reduction.csp_custom import CSPCustom

mne.set_log_level("WARNING")


def compare_accuracy(X, y, n_components: int, cv: int) -> None:
    """Niveau 3 : compare cross_val_score entre les deux implémentations."""
    pipeline_ref = build_pipeline(reducer_method="csp_reference", classifier_method="lda",
                                   n_components=n_components)
    pipeline_custom = build_pipeline(reducer_method="csp_custom", classifier_method="lda",
                                      n_components=n_components)

    scores_ref = cross_val_score(pipeline_ref, X, y, cv=cv)
    scores_custom = cross_val_score(pipeline_custom, X, y, cv=cv)

    print("\n--- Comparaison d'accuracy (cross_val_score) ---")
    print(f"CSP référence (MNE) : {np.round(scores_ref, 4)} | moyenne = {scores_ref.mean():.4f}")
    print(f"CSP maison           : {np.round(scores_custom, 4)} | moyenne = {scores_custom.mean():.4f}")

    ecart = abs(scores_ref.mean() - scores_custom.mean())
    print(f"Écart de moyenne : {ecart:.4f}")

    if ecart < 0.10:
        print("→ Écart raisonnable : l'implémentation maison est cohérente avec la référence.")
    else:
        print("→ ⚠️ Écart important : vérifier l'implémentation avant de la considérer validée.")


def compare_patterns(X, y, info: mne.Info, n_components: int) -> None:
    """Niveau 2 : compare visuellement les patterns des deux CSP."""
    csp_ref = get_reference_csp(n_components=n_components)
    csp_ref.fit(X, y)

    csp_custom = CSPCustom(n_components=n_components)
    csp_custom.fit(X, y)

    fig, axes = plt.subplots(2, n_components, figsize=(4 * n_components, 8))

    for idx in range(n_components):
        mne.viz.plot_topomap(csp_ref.patterns_[idx], info, axes=axes[0, idx], show=False)
        axes[0, idx].set_title(f"Référence - pattern #{idx}")

        mne.viz.plot_topomap(csp_custom.patterns_[idx], info, axes=axes[1, idx], show=False)
        axes[1, idx].set_title(f"Maison - pattern #{idx}")

    fig.suptitle("Comparaison des patterns CSP : référence (haut) vs maison (bas)")


def main():
    parser = argparse.ArgumentParser(description="Phase 2 - Validation comparative CSP")
    parser.add_argument("--subjects", type=int, nargs="+", default=[1])
    parser.add_argument("--runs", type=int, nargs="+", default=[4, 8, 12])
    parser.add_argument("--dataset", default="physionet")
    parser.add_argument("--n-components", type=int, default=4)
    parser.add_argument("--cv", type=int, default=5)
    parser.add_argument("--no-plot", action="store_true")
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

    if len(np.unique(y)) != 2:
        print(f"⚠️ {len(np.unique(y))} classes détectées au lieu de 2 — "
              f"CSPCustom lèvera une erreur. Vérifiez make_epochs() avant de continuer.")
        return

    compare_accuracy(X, y, n_components=args.n_components, cv=args.cv)

    if not args.no_plot:
        compare_patterns(X, y, epochs.info, n_components=args.n_components)
        plt.show()


if __name__ == "__main__":
    main()
