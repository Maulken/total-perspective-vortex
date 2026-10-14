"""
dimensionality_reduction/csp_custom.py

Implémentation maison du Common Spatial Patterns (CSP), respectant
l'API scikit-learn (BaseEstimator + TransformerMixin) afin d'être
interchangeable avec dimensionality_reduction/csp_reference.py dans
le pipeline sklearn.

Ce fichier traduit directement la formalisation mathématique établie
à l'étape 3 du plan d'action :

    1. Covariance spatiale par epoch
    2. Moyenne des covariances PAR CLASSE (pas sur les données concaténées)
    3. Résolution du problème de valeurs propres généralisé :
           Cov_A . v = lambda . Cov_B . v
    4. Tri des vecteurs propres, sélection des extrêmes (les plus discriminants)
    5. Construction de la matrice de filtres W
    6. transform() : projection + log-variance comme feature finale

NOTE : conformément au périmètre validé, l'eigendecomposition et
l'estimation de covariance NE SONT PAS recodées à la main (bonus exclu).
On utilise scipy.linalg.eigh, dont l'usage est explicitement autorisé
par le sujet ("Are also allowed Numpy or scipy functions to find
eigenvalues, singular values, and covariance matrix estimation.").
"""

import numpy as np
from scipy.linalg import eigh
from sklearn.base import BaseEstimator, TransformerMixin


class CSPCustom(BaseEstimator, TransformerMixin):
    """
    Common Spatial Patterns fait maison, pour une classification binaire.

    Attributs appris (suffixe "_", convention sklearn) :
        filters_  : matrice de projection complète (n_channels, n_channels)
        patterns_ : inverse des filtres, interprétable physiologiquement
                    (n_channels, n_channels)
    """

    def __init__(self, n_components: int = 4):
        # Nombre de composantes CSP à conserver au final (doit être pair :
        # la moitié la plus discriminante pour la classe A, l'autre
        # moitié pour la classe B).
        self.n_components = n_components

    # ------------------------------------------------------------------
    # Étape 1 : covariance spatiale d'une seule epoch
    # ------------------------------------------------------------------
    @staticmethod
    def _epoch_covariance(epoch: np.ndarray) -> np.ndarray:
        """
        Calcule la matrice de covariance spatiale d'une epoch unique.

        epoch : array (n_channels, n_times)
        Retourne : array (n_channels, n_channels)

        On utilise numpy.cov, autorisé explicitement par le sujet pour
        l'estimation de covariance. rowvar=True car chaque LIGNE de
        `epoch` est un canal (convention EEG standard), pas une colonne.
        """
        return np.cov(epoch, rowvar=True)

    # ------------------------------------------------------------------
    # Étape 2 : covariance moyenne PAR CLASSE
    # ------------------------------------------------------------------
    def _class_mean_covariance(self, X: np.ndarray) -> np.ndarray:
        """
        Calcule la covariance moyenne sur un ensemble d'epochs
        appartenant à UNE SEULE classe.

        X : array (n_epochs_classe, n_channels, n_times)
        Retourne : array (n_channels, n_channels)

        Point de vigilance (piège classique) : on moyenne les matrices
        de covariance INDIVIDUELLES de chaque epoch, on ne calcule PAS
        la covariance sur les données concaténées de toutes les epochs.
        Ces deux approches donnent des résultats mathématiquement
        différents ; seule la première est correcte pour le CSP.
        """
        covariances = np.array([self._epoch_covariance(epoch) for epoch in X])
        return covariances.mean(axis=0)

    # ------------------------------------------------------------------
    # fit : étapes 2 à 5 de la formalisation
    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray) -> "CSPCustom":
        """
        X : array (n_epochs, n_channels, n_times)
        y : array (n_epochs,) — exactement 2 classes attendues

        Cette méthode a le droit de regarder y (contrairement à
        transform), conformément au contrat fit/transform de sklearn.
        """
        classes = np.unique(y)
        if len(classes) != 2:
            raise ValueError(
                f"CSPCustom ne supporte que 2 classes, {len(classes)} reçues : {classes}. "
                "Vérifiez le filtrage de vos epochs (ex : exclusion de la classe de repos)."
            )

        class_a, class_b = classes

        # --- Étape 2 : covariance moyenne par classe ---
        cov_a = self._class_mean_covariance(X[y == class_a])
        cov_b = self._class_mean_covariance(X[y == class_b])

        # --- Étape 3 : problème de valeurs propres généralisé ---
        # eigh(A, B) résout A.v = lambda.B.v et retourne des valeurs
        # propres triées par ordre croissant, avec des vecteurs propres
        # déjà normalisés. On utilise (cov_a, cov_a + cov_b) plutôt que
        # (cov_a, cov_b) directement : c'est la formulation classique du
        # CSP, numériquement plus stable, et mathématiquement équivalente
        # (le rapport de discrimination entre les deux classes est
        # préservé, seule l'échelle de la valeur propre change).
        eigvals, eigvecs = eigh(cov_a, cov_a + cov_b)

        # --- Étape 4 : sélection des composantes les plus discriminantes ---
        # eigvals est trié croissant. Les valeurs proches de 1 signifient
        # "toute la variance est expliquée par la classe A" (peu de
        # variance pour la classe B), les valeurs proches de 0 signifient
        # l'inverse. On garde donc les extrêmes des deux côtés du spectre.
        n_pairs = self.n_components // 2
        selected_indices = np.concatenate([
            np.arange(n_pairs),                      # extrémité basse (discriminant classe B)
            np.arange(len(eigvals) - n_pairs, len(eigvals)),  # extrémité haute (discriminant classe A)
        ])

        # --- Étape 5 : construction de la matrice de filtres ---
        # Convention : les colonnes de eigvecs sont les vecteurs propres.
        # On transpose pour obtenir des filtres au format (n_components, n_channels),
        # cohérent avec l'application W^T X du sujet.
        self.filters_ = eigvecs[:, selected_indices].T

        # Patterns : inverse des filtres, utilisés pour l'interprétation
        # physiologique (topomap), PAS pour la projection elle-même.
        self.patterns_ = np.linalg.pinv(self.filters_).T

        return self

    # ------------------------------------------------------------------
    # transform : étape 6 de la formalisation
    # ------------------------------------------------------------------
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        X : array (n_epochs, n_channels, n_times)
        Retourne : array (n_epochs, n_components) — la feature finale
        exploitable par un classifieur.

        IMPORTANT : cette méthode n'utilise PAS y, conformément au
        contrat sklearn (pas de fuite de données lors de la validation
        croisée).
        """
        if not hasattr(self, "filters_"):
            raise RuntimeError("CSPCustom doit être entraîné (fit) avant transform.")

        n_epochs = X.shape[0]
        features = np.zeros((n_epochs, self.filters_.shape[0]))

        for i, epoch in enumerate(X):
            # Projection : W^T X, conformément à la notation du sujet.
            projected = self.filters_ @ epoch  # (n_components, n_times)

            # Log-variance : stabilise la distribution de la variance,
            # qui est souvent très asymétrique en EEG. C'est la feature
            # finale classiquement utilisée avec le CSP, plus discriminante
            # qu'une simple projection brute non résumée.
            variance = np.var(projected, axis=1)
            features[i] = np.log(variance + 1e-10)  # epsilon pour éviter log(0)

        return features
