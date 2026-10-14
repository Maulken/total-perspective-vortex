"""
load_edf_mne.py

Script de démonstration pour charger, explorer et découper en epochs
les fichiers du dataset PhysioNet "EEG Motor Movement/Imagery Dataset"
(fichiers .edf + annotations .edf.event) avec la librairie MNE.

Usage:
    python load_edf_mne.py --subject 1 --run 4

Dépendances:
    pip install mne matplotlib numpy
"""

import argparse
import matplotlib.pyplot as plt
import mne
from mne.datasets import eegbci


# ---------------------------------------------------------------------------
# 1. Téléchargement / localisation des fichiers
# ---------------------------------------------------------------------------
def get_raw_path(subject: int, run: int) -> str:
    """
    Télécharge (si nécessaire) et renvoie le chemin local du fichier .edf
    correspondant au sujet et au run demandés, via l'utilitaire MNE dédié
    au dataset PhysioNet eegbci (S001R01.edf, etc.).

    MNE gère automatiquement la récupération du fichier .edf ET des
    annotations associées (le contenu du .edf.event est intégré comme
    annotations dans le fichier EDF+ fourni par MNE).
    """
    paths = eegbci.load_data(subject, run, path=None, update_path=True)
    return paths[0]


# ---------------------------------------------------------------------------
# 2. Chargement du signal brut
# ---------------------------------------------------------------------------
def load_raw(subject: int, run: int) -> mne.io.Raw:
    """
    Charge le signal EEG brut pour un sujet/run donné.

    Retourne un objet mne.io.Raw contenant :
      - les données multi-canaux (64 électrodes, montage 10-10)
      - la fréquence d'échantillonnage (160 Hz)
      - les annotations d'événements (T0/T1/T2) issues du fichier .event
    """
    edf_path = get_raw_path(subject, run)
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)

    # Standardisation des noms de canaux (le dataset utilise une notation
    # legacy '.' qu'il faut nettoyer pour matcher les montages standards MNE)
    eegbci.standardize(raw)
    montage = mne.channels.make_standard_montage("standard_1005")
    raw.set_montage(montage)

    return raw


# ---------------------------------------------------------------------------
# 3. Filtrage passe-bande (bandes mu/beta utiles à l'imagerie motrice)
# ---------------------------------------------------------------------------
def filter_raw(raw: mne.io.Raw, l_freq: float = 7.0, h_freq: float = 30.0) -> mne.io.Raw:
    """
    Applique un filtre passe-bande pour ne conserver que les bandes
    mu (8-12 Hz) et beta (13-30 Hz), pertinentes pour l'imagerie motrice.
    """
    raw_filtered = raw.copy().filter(l_freq, h_freq, fir_design="firwin", verbose=False)
    return raw_filtered


# ---------------------------------------------------------------------------
# 4. Extraction des événements et découpage en epochs
# ---------------------------------------------------------------------------
def make_epochs(raw: mne.io.Raw, tmin: float = -1.0, tmax: float = 4.0) -> mne.Epochs:
    """
    Extrait les événements depuis les annotations du Raw et découpe
    le signal continu en epochs autour de chaque événement.

    Les labels typiques du dataset eegbci sont :
        T0 : repos
        T1 : mouvement/imagination (poing gauche ou deux poings selon le run)
        T2 : mouvement/imagination (poing droit ou deux pieds selon le run)
    """
    events, event_id = mne.events_from_annotations(raw, verbose=False)

    epochs = mne.Epochs(
        raw,
        events,
        event_id=event_id,
        tmin=tmin,
        tmax=tmax,
        baseline=None,
        preload=True,
        verbose=False,
    )
    return epochs


# ---------------------------------------------------------------------------
# 5. Visualisation avant / après filtrage
# ---------------------------------------------------------------------------
def plot_comparison(raw: mne.io.Raw, raw_filtered: mne.io.Raw) -> None:
    raw.plot(n_channels=10, duration=10, title="Signal brut", show=False)
    raw_filtered.plot(n_channels=10, duration=10, title="Signal filtré (7-30 Hz)", show=False)
    plt.show()


# ---------------------------------------------------------------------------
# Point d'entrée
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Chargement EEG PhysioNet avec MNE")
    parser.add_argument("--subject", type=int, default=1, help="Numéro du sujet (1-109)")
    parser.add_argument("--run", type=int, default=4, help="Numéro du run (1-14)")
    parser.add_argument("--no-plot", action="store_true", help="Désactive l'affichage graphique")
    args = parser.parse_args()

    print(f"Chargement sujet {args.subject:03d}, run {args.run:02d} ...")
    raw = load_raw(args.subject, args.run)

    print(raw.info)
    print(f"Canaux : {len(raw.ch_names)}")
    print(f"Fréquence d'échantillonnage : {raw.info['sfreq']} Hz")

    raw_filtered = filter_raw(raw)

    epochs = make_epochs(raw_filtered)
    print(f"Nombre d'epochs extraites : {len(epochs)}")
    print(f"Labels d'événements : {epochs.event_id}")

    if not args.no_plot:
        plot_comparison(raw, raw_filtered)


if __name__ == "__main__":
    main()
