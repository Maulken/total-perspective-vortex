"""
preprocessing/features/fourier_features.py

Extraction de features via la puissance du signal par bande de
fréquence et par canal (transformée de Fourier / Welch), telle que
suggérée en exemple par le sujet.

Compatible avec l'API scikit-learn (BaseEstimator + TransformerMixin)
afin d'être interchangeable dans le Pipeline avec l'extracteur wavelet
(bonus) sans modifier le reste de l'architecture.
"""

import numpy as np
from mne.time_frequency import psd_array_welch
from sklearn.base import BaseEstimator, TransformerMixin


class FourierBandPowerExtractor(BaseEstimator, TransformerMixin):
    """
    Transformer sklearn qui convertit des epochs EEG (n_epochs, n_channels,
    n_times) en un vecteur de features (n_epochs, n_channels * n_bands),
    où chaque feature est la puissance moyenne du signal dans une bande
    de fréquence donnée, pour un canal donné.
    """

    def __init__(self, sfreq: float, bands=((8.0, 12.0), (13.0, 30.0)), n_fft: int = 256):
        # bands par défaut : mu (8-12 Hz) et beta (13-30 Hz)
        self.sfreq = sfreq
        self.bands = bands
        self.n_fft = n_fft

    def fit(self, X, y=None):
        # Aucun paramètre à apprendre : l'extraction de features est
        # déterministe et ne dépend pas des données d'entraînement.
        return self

    def transform(self, X):
        """
        X : array (n_epochs, n_channels, n_times)
        Retourne : array (n_epochs, n_channels * n_bands)
        """
        n_epochs, n_channels, n_times = X.shape
        n_fft = min(self.n_fft, n_times)

        psds, freqs = psd_array_welch(
            X, sfreq=self.sfreq, fmin=self.bands[0][0], fmax=self.bands[-1][1],
            n_fft=n_fft, verbose=False,
        )
        # psds shape: (n_epochs, n_channels, n_freqs)

        features = []
        for (fmin, fmax) in self.bands:
            band_mask = (freqs >= fmin) & (freqs <= fmax)
            band_power = psds[:, :, band_mask].mean(axis=2)  # (n_epochs, n_channels)
            features.append(band_power)

        # Concatène les bandes -> (n_epochs, n_channels * n_bands)
        return np.concatenate(features, axis=1)
