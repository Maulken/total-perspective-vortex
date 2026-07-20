"""
preprocessing/filtering.py

Filtrage fréquentiel du signal EEG et visualisation avant/après.
"""

import matplotlib.pyplot as plt
import mne


def filter_raw(
    raw: mne.io.Raw,
    l_freq: float = 7.0,
    h_freq: float = 30.0,
    notch_freq: float = None,
) -> mne.io.Raw:
    """
    Applique un filtre passe-bande pour ne conserver que les bandes
    mu (8-12 Hz) et beta (13-30 Hz), pertinentes pour l'imagerie motrice.

    Un filtre coupe-bande (notch) optionnel permet d'éliminer le bruit
    secteur (50 ou 60 Hz selon la région d'enregistrement) si nécessaire.
    """
    raw_filtered = raw.copy()

    if notch_freq is not None:
        raw_filtered.notch_filter(notch_freq, verbose=False)

    raw_filtered.filter(l_freq, h_freq, fir_design="firwin", verbose=False)
    return raw_filtered


def plot_comparison(raw: mne.io.Raw, raw_filtered: mne.io.Raw, duration: float = 10.0) -> None:
    """Affiche côte à côte le signal brut et le signal filtré."""
    raw.plot(n_channels=10, duration=duration, title="Signal brut", show=False)
    raw_filtered.plot(n_channels=10, duration=duration, title="Signal filtré", show=False)
    plt.show()


def plot_psd_comparison(raw: mne.io.Raw, raw_filtered: mne.io.Raw) -> None:
    """Affiche la densité spectrale de puissance avant/après filtrage."""
    raw.compute_psd(verbose=False).plot(show=False)
    plt.gcf().suptitle("PSD - signal brut")

    raw_filtered.compute_psd(verbose=False).plot(show=False)
    plt.gcf().suptitle("PSD - signal filtré")

    plt.show()
