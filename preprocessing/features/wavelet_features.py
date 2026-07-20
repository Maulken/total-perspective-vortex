"""
preprocessing/features/wavelet_features.py

BONUS : extraction de features via décomposition en ondelettes discrètes
(DWT), utilisée comme alternative à l'approche Fourier/Welch.

Compatible avec l'API scikit-learn, interchangeable avec
FourierBandPowerExtractor dans le pipeline via build_pipeline().

Dépendance additionnelle : pip install PyWavelets
"""

import numpy as np
import pywt
from sklearn.base import BaseEstimator, TransformerMixin


class WaveletEnergyExtractor(BaseEstimator, TransformerMixin):
    """
    Transformer sklearn qui convertit des epochs EEG (n_epochs, n_channels,
    n_times) en un vecteur de features (n_epochs, n_channels * n_levels),
    où chaque feature est l'énergie des coefficients d'ondelettes à un
    niveau de décomposition donné, pour un canal donné.

    L'énergie par niveau de décomposition joue un rôle analogue à la
    puissance par bande de fréquence dans l'approche Fourier, mais avec
    une résolution temps-fréquence adaptative propre aux ondelettes.
    """

    def __init__(self, wavelet: str = "db4", level: int = 4):
        self.wavelet = wavelet
        self.level = level

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """
        X : array (n_epochs, n_channels, n_times)
        Retourne : array (n_epochs, n_channels * (level + 1))
        """
        n_epochs, n_channels, n_times = X.shape
        n_bands = self.level + 1  # coefficients d'approximation + détails

        features = np.zeros((n_epochs, n_channels * n_bands))

        for i in range(n_epochs):
            for c in range(n_channels):
                coeffs = pywt.wavedec(X[i, c, :], wavelet=self.wavelet, level=self.level)
                energies = [np.sum(np.square(coef)) for coef in coeffs]
                start = c * n_bands
                features[i, start:start + n_bands] = energies

        return features
