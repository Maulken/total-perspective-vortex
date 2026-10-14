# """
# scripts/phase1_analysis.py

# Script d'ANALYSE EXPLORATOIRE (pas d'entraînement) pour la Phase 1.
# Contrairement à phase1_demo.py (pensé pour être lancé en pipeline, avec
# possibilité de couper l'affichage), ce script affiche systématiquement
# toutes les visualisations utiles pour :

#     - vérifier la qualité du signal brut
#     - valider l'effet du filtrage passe-bande
#     - inspecter les epochs par classe
#     - comparer la puissance par bande/canal entre les deux classes de
#       mouvement (utile pour juger si les features Fourier sont
#       discriminantes avant de les brancher dans le CSP)
#     - visualiser la topographie de la puissance par bande (repère les
#       électrodes C3/C4, attendues comme discriminantes en imagerie
#       motrice main gauche/droite)

# Usage:
#     python -m scripts.phase1_analysis --subjects 1 --runs 4
#     python -m scripts.phase1_analysis --subjects 1 2 --runs 4 8 12 --features wavelet
# """

# import argparse

# import matplotlib.pyplot as plt
# import mne
# import numpy as np

# from data.loader import load_raw
# from preprocessing.filtering import filter_raw
# from preprocessing.epoching import make_epochs
# from preprocessing.features.fourier_features import FourierBandPowerExtractor
# from preprocessing.features.wavelet_features import WaveletEnergyExtractor


# def plot_raw_vs_filtered(raw: mne.io.Raw, raw_filtered: mne.io.Raw) -> None:
#     """Affiche signal temporel et PSD, avant et après filtrage."""
#     raw.plot(n_channels=10, duration=10, title="Signal brut", show=False)
#     raw_filtered.plot(n_channels=10, duration=10, title="Signal filtré", show=False)

#     raw.compute_psd(verbose=False).plot(show=False)
#     plt.gcf().suptitle("PSD - signal brut")

#     raw_filtered.compute_psd(verbose=False).plot(show=False)
#     plt.gcf().suptitle("PSD - signal filtré")


# def plot_epochs_by_class(epochs: mne.Epochs) -> None:
#     """
#     Affiche le potentiel moyen (evoked) par classe, ainsi que l'image
#     des epochs (utile pour repérer visuellement une désynchronisation
#     liée à l'événement, l'ERD/ERS attendu en imagerie motrice).
#     """
#     for label in epochs.event_id:
#         evoked = epochs[label].average()
#         evoked.plot(show=False, titles=f"Moyenne des epochs - classe {label}")


# def plot_band_power_by_class(X: np.ndarray, y: np.ndarray, event_id: dict, sfreq: float) -> None:
#     """
#     Compare la puissance par bande/canal entre les classes, sous forme
#     de boxplots. Permet de juger visuellement si la feature Fourier est
#     discriminante avant de la brancher dans le pipeline CSP.
#     """
#     extractor = FourierBandPowerExtractor(sfreq=sfreq)
#     features = extractor.fit_transform(X)  # (n_epochs, n_channels * n_bands)

#     labels = list(event_id.keys())
#     codes = list(event_id.values())

#     # On agrège la puissance sur tous les canaux/bandes pour une vue globale
#     fig, ax = plt.subplots()
#     data_per_class = [features[y == code].mean(axis=1) for code in codes]
#     ax.boxplot(data_per_class, tick_labels=labels)
#     ax.set_title("Puissance moyenne (toutes bandes/canaux) par classe")
#     ax.set_ylabel("Puissance")


# def plot_band_power_topomap(X: np.ndarray, y: np.ndarray, event_id: dict, epochs: mne.Epochs) -> None:
#     """
#     Affiche la topographie de la puissance en bande mu (8-12Hz) pour
#     chaque classe, afin de repérer visuellement la lateralisation
#     attendue sur les électrodes C3/C4 en imagerie motrice.
#     """
#     extractor = FourierBandPowerExtractor(sfreq=epochs.info["sfreq"], bands=((8.0, 12.0),))

#     for label, code in event_id.items():
#         features = extractor.fit_transform(X[y == code])  # (n_epochs_classe, n_channels)
#         mean_power = features.mean(axis=0)  # (n_channels,)

#         fig, ax = plt.subplots()
#         mne.viz.plot_topomap(mean_power, epochs.info, axes=ax, show=False)
#         ax.set_title(f"Topographie puissance mu (8-12Hz) - classe {label}")


# def main():
#     parser = argparse.ArgumentParser(description="Analyse exploratoire Phase 1")
#     parser.add_argument("--subjects", type=int, nargs="+", default=[1])
#     parser.add_argument("--runs", type=int, nargs="+", default=[4])
#     parser.add_argument("--features", choices=["fourier", "wavelet"], default="fourier")
#     parser.add_argument("--dataset", default="physionet")
#     args = parser.parse_args()

#     print(f"Chargement sujets={args.subjects}, runs={args.runs} ...")
#     raw = load_raw(args.subjects, args.runs, dataset_name=args.dataset)
#     raw_filtered = filter_raw(raw)

#     plot_raw_vs_filtered(raw, raw_filtered)

#     epochs = make_epochs(raw_filtered)
#     print(f"Nombre d'epochs : {len(epochs)}")
#     print(f"Classes : {epochs.event_id}")

#     plot_epochs_by_class(epochs)

#     X = epochs.get_data()
#     y = epochs.events[:, -1]

#     if args.features == "fourier":
#         plot_band_power_by_class(X, y, epochs.event_id, sfreq=epochs.info["sfreq"])
#         plot_band_power_topomap(X, y, epochs.event_id, epochs)
#     else:
#         extractor = WaveletEnergyExtractor()
#         features = extractor.fit_transform(X)
#         fig, ax = plt.subplots()
#         codes = list(epochs.event_id.values())
#         data_per_class = [features[y == code].mean(axis=1) for code in codes]
#         ax.boxplot(data_per_class, tick_labels=list(epochs.event_id.keys()))
#         ax.set_title("Énergie moyenne des coefficients d'ondelettes par classe")

#     print("Affichage des graphiques d'analyse ...")
#     plt.show()


# if __name__ == "__main__":
#     main()


"""
scripts/phase1_analysis.py

Script d'ANALYSE EXPLORATOIRE (pas d'entraînement) pour la Phase 1.
Contrairement à phase1_demo.py (pensé pour être lancé en pipeline, avec
possibilité de couper l'affichage), ce script affiche systématiquement
toutes les visualisations utiles pour :

    - vérifier la qualité du signal brut
    - valider l'effet du filtrage passe-bande
    - inspecter les epochs par classe
    - comparer la puissance par bande/canal entre les deux classes de
      mouvement (utile pour juger si les features Fourier sont
      discriminantes avant de les brancher dans le CSP)
    - visualiser la topographie de la puissance par bande (repère les
      électrodes C3/C4, attendues comme discriminantes en imagerie
      motrice main gauche/droite)

Usage:
    python -m scripts.phase1_analysis --subjects 1 --runs 4
    python -m scripts.phase1_analysis --subjects 1 2 --runs 4 8 12 --features wavelet
"""

import argparse

import matplotlib.pyplot as plt
import mne
import numpy as np

from data.loader import load_raw
from preprocessing.filtering import filter_raw
from preprocessing.epoching import make_epochs
from preprocessing.features.fourier_features import FourierBandPowerExtractor
from preprocessing.features.wavelet_features import WaveletEnergyExtractor


def plot_raw_vs_filtered(raw: mne.io.Raw, raw_filtered: mne.io.Raw) -> None:
    """Affiche signal temporel et PSD, avant et après filtrage."""
    raw.plot(n_channels=10, duration=10, title="Signal brut", show=False)
    raw_filtered.plot(n_channels=10, duration=10, title="Signal filtré", show=False)

    raw.compute_psd(verbose=False).plot(show=False)
    plt.gcf().suptitle("PSD - signal brut")

    raw_filtered.compute_psd(verbose=False).plot(show=False)
    plt.gcf().suptitle("PSD - signal filtré")


def plot_epochs_by_class(epochs: mne.Epochs) -> None:
    """
    Affiche le potentiel moyen (evoked) par classe, ainsi que l'image
    des epochs (utile pour repérer visuellement une désynchronisation
    liée à l'événement, l'ERD/ERS attendu en imagerie motrice).
    """
    for label in epochs.event_id:
        evoked = epochs[label].average()
        evoked.plot(show=False, titles=f"Moyenne des epochs - classe {label}")


def plot_band_power_by_class(X: np.ndarray, y: np.ndarray, event_id: dict, sfreq: float) -> None:
    """
    Compare la puissance par bande/canal entre les classes, sous forme
    de boxplots. Permet de juger visuellement si la feature Fourier est
    discriminante avant de la brancher dans le pipeline CSP.
    """
    extractor = FourierBandPowerExtractor(sfreq=sfreq)
    features = extractor.fit_transform(X)  # (n_epochs, n_channels * n_bands)

    labels = list(event_id.keys())
    codes = list(event_id.values())

    # On agrège la puissance sur tous les canaux/bandes pour une vue globale
    fig, ax = plt.subplots()
    data_per_class = [features[y == code].mean(axis=1) for code in codes]
    ax.boxplot(data_per_class, tick_labels=labels)
    ax.set_title("Puissance moyenne (toutes bandes/canaux) par classe")
    ax.set_ylabel("Puissance")


def plot_band_power_topomap(X: np.ndarray, y: np.ndarray, event_id: dict, epochs: mne.Epochs) -> None:
    """
    Affiche la topographie de la puissance en bande mu (8-12Hz) pour
    chaque classe, afin de repérer visuellement la lateralisation
    attendue sur les électrodes C3/C4 en imagerie motrice.
    """
    extractor = FourierBandPowerExtractor(sfreq=epochs.info["sfreq"], bands=((8.0, 12.0),))

    for label, code in event_id.items():
        features = extractor.fit_transform(X[y == code])  # (n_epochs_classe, n_channels)
        mean_power = features.mean(axis=0)  # (n_channels,)

        fig, ax = plt.subplots()
        mne.viz.plot_topomap(mean_power, epochs.info, axes=ax, show=False)
        ax.set_title(f"Topographie puissance mu (8-12Hz) - classe {label}")


def main():
    parser = argparse.ArgumentParser(description="Analyse exploratoire Phase 1")
    parser.add_argument("--subjects", type=int, nargs="+", default=[1])
    parser.add_argument("--runs", type=int, nargs="+", default=[4])
    parser.add_argument("--features", choices=["fourier", "wavelet"], default="fourier")
    parser.add_argument("--dataset", default="physionet")
    args = parser.parse_args()

    print(f"Chargement sujets={args.subjects}, runs={args.runs} ...")
    if len(args.subjects) > 1:
        print(f"⚠️ Ce script travaille sur UN SEUL sujet à la fois. "
              f"Seul le premier ({args.subjects[0]}) sera utilisé, "
              f"les autres ({args.subjects[1:]}) sont ignorés.")
    raw = load_raw(args.subjects[0], args.runs, dataset_name=args.dataset)
    raw_filtered = filter_raw(raw)

    plot_raw_vs_filtered(raw, raw_filtered)

    epochs = make_epochs(raw_filtered)
    print(f"Nombre d'epochs : {len(epochs)}")
    print(f"Classes : {epochs.event_id}")

    plot_epochs_by_class(epochs)

    X = epochs.get_data()
    y = epochs.events[:, -1]

    if args.features == "fourier":
        plot_band_power_by_class(X, y, epochs.event_id, sfreq=epochs.info["sfreq"])
        plot_band_power_topomap(X, y, epochs.event_id, epochs)
    else:
        extractor = WaveletEnergyExtractor()
        features = extractor.fit_transform(X)
        fig, ax = plt.subplots()
        codes = list(epochs.event_id.values())
        data_per_class = [features[y == code].mean(axis=1) for code in codes]
        ax.boxplot(data_per_class, tick_labels=list(epochs.event_id.keys()))
        ax.set_title("Énergie moyenne des coefficients d'ondelettes par classe")

    print("Affichage des graphiques d'analyse ...")
    plt.show()


if __name__ == "__main__":
    main()
