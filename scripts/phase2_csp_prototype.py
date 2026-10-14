# """
# scripts/phase2_csp_prototype.py

# Étape 1 du plan d'action Phase 2 : prototypage avec le CSP de référence
# (mne.decoding.CSP) pour valider l'architecture globale du pipeline et
# obtenir un score de cross_val_score servant de référence pour la
# future comparaison avec le CSP maison.

# Usage:
#     python -m scripts.phase2_csp_prototype --subjects 1 --runs 4 8 12
# """

# import argparse

# import numpy as np
# from sklearn.model_selection import cross_val_score

# from data.loader import load_raw
# from preprocessing.filtering import filter_raw
# from preprocessing.epoching import make_epochs
# from pipeline.build_pipeline import build_pipeline


# def main():
#     parser = argparse.ArgumentParser(description="Phase 2 - Prototypage CSP de référence")
#     parser.add_argument("--subjects", type=int, nargs="+", default=[1])
#     parser.add_argument("--runs", type=int, nargs="+", default=[4, 8, 12])
#     parser.add_argument("--dataset", default="physionet")
#     parser.add_argument("--n-components", type=int, default=4)
#     parser.add_argument("--cv", type=int, default=5, help="Nombre de folds pour cross_val_score")
#     args = parser.parse_args()

#     print(f"Chargement sujets={args.subjects}, runs={args.runs} ")
#     raw = load_raw(args.subjects, args.runs, dataset_name=args.dataset)
#     raw_filtered = filter_raw(raw)

#     epochs = make_epochs(raw_filtered)
#     print(f"Nombre d'epochs : {len(epochs)}")
#     print(f"Classes : {epochs.event_id}")

#     # Le CSP travaille directement sur le signal temporel multi-canaux,
#     # pas sur des features déjà extraites (contrairement à la Phase 1).
#     X = epochs.get_data()   # (n_epochs, n_channels, n_times)
#     y = epochs.events[:, -1]

#     pipeline = build_pipeline(reducer_method="csp_reference", classifier_method="lda",
#                                n_components=args.n_components)

#     scores = cross_val_score(pipeline, X, y, cv=args.cv)

#     print(f"Scores par fold : {np.round(scores, 4)}")
#     print(f"cross_val_score (moyenne) : {scores.mean():.4f}")
#     print(f"+/- : {scores.std():.4f}")
#     print(f"Score de référence pour valider le futur CSP.")


# if __name__ == "__main__":
#     main()


"""
scripts/phase2_csp_prototype.py

Étape 1 du plan d'action Phase 2 : prototypage avec le CSP de référence
(mne.decoding.CSP) pour valider l'architecture globale du pipeline et
obtenir un score de cross_val_score servant de référence pour la
future comparaison avec le CSP maison.

Usage:
    python -m scripts.phase2_csp_prototype --subjects 1 --runs 4 8 12
"""

import argparse

import mne
import numpy as np
from sklearn.model_selection import cross_val_score

from data.loader import load_raw
from preprocessing.filtering import filter_raw
from preprocessing.epoching import make_epochs
from pipeline.build_pipeline import build_pipeline

# Coupe le logging interne de MNE (y compris celui de CSP.fit(), qui ne
# supporte pas verbose= directement en argument constructeur dans
# certaines versions). "WARNING" laisse passer les messages importants
# (erreurs, avertissements) tout en supprimant les logs INFO répétitifs.
mne.set_log_level("WARNING")


def main():
    parser = argparse.ArgumentParser(description="Phase 2 - Prototypage CSP de référence")
    parser.add_argument("--subjects", type=int, nargs="+", default=[1])
    parser.add_argument("--runs", type=int, nargs="+", default=[4, 8, 12])
    parser.add_argument("--dataset", default="physionet")
    parser.add_argument("--n-components", type=int, default=4)
    parser.add_argument("--cv", type=int, default=5, help="Nombre de folds pour cross_val_score")
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

    # Le CSP travaille directement sur le signal temporel multi-canaux,
    # pas sur des features déjà extraites (contrairement à la Phase 1).
    X = epochs.get_data()   # (n_epochs, n_channels, n_times)
    y = epochs.events[:, -1]

    pipeline = build_pipeline(reducer_method="csp_reference", classifier_method="lda",
                               n_components=args.n_components)

    scores = cross_val_score(pipeline, X, y, cv=args.cv)

    print(f"Scores par fold : {np.round(scores, 4)}")
    print(f"cross_val_score (moyenne) : {scores.mean():.4f}")
    print(f"Ce score sert de référence pour valider le futur CSP maison.")


if __name__ == "__main__":
    main()
