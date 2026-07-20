"""
scripts/phase1_demo.py

Démonstration de la Phase 1 (V.1.1 - Preprocessing, parsing et formating) :
chargement, visualisation, filtrage, découpage en epochs et extraction
de features, en s'appuyant sur les modules du projet.

Supporte :
    - le chargement de plusieurs sujets et/ou plusieurs runs (concaténation)
    - le choix de la méthode d'extraction de features : fourier (obligatoire)
      ou wavelet (bonus)

Usage:
    python -m scripts.phase1_demo --subjects 1 2 --runs 4 8 12 --features fourier
    python -m scripts.phase1_demo --subjects 1 --runs 4 --features wavelet --no-plot
"""

import argparse

from data.loader import load_raw
from preprocessing.filtering import filter_raw, plot_comparison, plot_psd_comparison
from preprocessing.epoching import make_epochs
from preprocessing.features.fourier_features import FourierBandPowerExtractor
from preprocessing.features.wavelet_features import WaveletEnergyExtractor


def get_feature_extractor(method: str, sfreq: float):
    if method == "fourier":
        return FourierBandPowerExtractor(sfreq=sfreq)
    elif method == "wavelet":
        return WaveletEnergyExtractor()
    else:
        raise ValueError(f"Méthode de features inconnue : {method}")


def main():
    parser = argparse.ArgumentParser(description="Phase 1 - Preprocessing EEG")
    parser.add_argument("--subjects", type=int, nargs="+", default=[1],
                         help="Liste des sujets à charger (ex : --subjects 1 2 3)")
    parser.add_argument("--runs", type=int, nargs="+", default=[4],
                         help="Liste des runs à charger (ex : --runs 4 8 12)")
    parser.add_argument("--features", choices=["fourier", "wavelet"], default="fourier",
                         help="Méthode d'extraction de features")
    parser.add_argument("--dataset", default="physionet", help="Nom du dataset (registre)")
    parser.add_argument("--no-plot", action="store_true", help="Désactive l'affichage graphique")
    args = parser.parse_args()

    print(f"Chargement sujets={args.subjects}, runs={args.runs}, dataset={args.dataset} ...")
    raw = load_raw(args.subjects, args.runs, dataset_name=args.dataset)

    print(raw.info)
    print(f"Canaux : {len(raw.ch_names)}")
    print(f"Fréquence d'échantillonnage : {raw.info['sfreq']} Hz")

    raw_filtered = filter_raw(raw)

    if not args.no_plot:
        plot_comparison(raw, raw_filtered)
        plot_psd_comparison(raw, raw_filtered)

    epochs = make_epochs(raw_filtered)
    print(f"Nombre d'epochs extraites : {len(epochs)}")
    print(f"Labels d'événements : {epochs.event_id}")

    X = epochs.get_data()  # (n_epochs, n_channels, n_times)
    y = epochs.events[:, -1]

    extractor = get_feature_extractor(args.features, sfreq=epochs.info["sfreq"])
    features = extractor.fit_transform(X)

    print(f"Méthode de features utilisée : {args.features}")
    print(f"Shape des features extraites : {features.shape}")
    print(f"Labels associés : {y[:10]} ...")


if __name__ == "__main__":
    main()
