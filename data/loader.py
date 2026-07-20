"""
data/loader.py

Chargement des données EEG et abstraction du dataset source.

- Gère le chargement d'un ou plusieurs sujets / runs.
- Concatène automatiquement plusieurs Raw en un seul objet MNE.
- Expose une interface générique (BaseDatasetLoader) afin qu'un autre
  dataset que PhysioNet eegbci puisse être branché sans modifier le
  reste du pipeline (bonus "autres datasets").
"""

from abc import ABC, abstractmethod
from typing import List

import mne
from mne.datasets import eegbci


# ---------------------------------------------------------------------------
# Interface générique (permet le bonus "autres datasets")
# ---------------------------------------------------------------------------
class BaseDatasetLoader(ABC):
    """
    Interface commune que doit respecter tout loader de dataset EEG.
    Un nouveau dataset (bonus) s'intègre en implémentant cette interface,
    sans toucher au reste du pipeline (preprocessing, features, CSP...).
    """

    @abstractmethod
    def load(self, subjects: List[int], runs: List[int]) -> mne.io.Raw:
        """Retourne un objet mne.io.Raw concaténé pour les sujets/runs donnés."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Implémentation pour PhysioNet eegbci (dataset demandé par le sujet)
# ---------------------------------------------------------------------------
class PhysionetLoader(BaseDatasetLoader):
    """
    Loader pour le "EEG Motor Movement/Imagery Dataset" (PhysioNet, eegbci),
    fichiers SxxxRyy.edf + annotations associées.
    """

    MONTAGE_NAME = "standard_1005"

    def load(self, subjects: List[int], runs: List[int]) -> mne.io.Raw:
        """
        Télécharge (si nécessaire) et concatène les Raw de plusieurs
        sujets et runs en un seul objet MNE, prêt pour le filtrage
        et le découpage en epochs.

        Note : concaténer des sujets différents dans un même Raw n'a de
        sens que si vous entraînez un modèle "cross-subject". Pour un
        entraînement classique sujet par sujet, appelez cette méthode
        avec une liste subjects=[un seul sujet] et plusieurs runs.
        """
        raw_list = []
        for subject in subjects:
            paths = eegbci.load_data(subject, runs, path=None, update_path=True)
            for path in paths:
                raw = mne.io.read_raw_edf(path, preload=True, verbose=False)
                eegbci.standardize(raw)
                raw_list.append(raw)

        raw_concat = mne.concatenate_raws(raw_list, verbose=False)

        montage = mne.channels.make_standard_montage(self.MONTAGE_NAME)
        raw_concat.set_montage(montage)

        return raw_concat


# ---------------------------------------------------------------------------
# Registre de datasets : point d'extension pour le bonus "autres datasets"
# ---------------------------------------------------------------------------
DATASET_REGISTRY = {
    "physionet": PhysionetLoader,
    # "mon_autre_dataset": MonAutreDatasetLoader,  # bonus : ajout futur
}


def get_loader(dataset_name: str = "physionet") -> BaseDatasetLoader:
    """Factory renvoyant le loader correspondant au dataset demandé."""
    if dataset_name not in DATASET_REGISTRY:
        raise ValueError(
            f"Dataset '{dataset_name}' inconnu. "
            f"Disponibles : {list(DATASET_REGISTRY.keys())}"
        )
    return DATASET_REGISTRY[dataset_name]()


def load_raw(subjects: List[int], runs: List[int], dataset_name: str = "physionet") -> mne.io.Raw:
    """Point d'entrée simple utilisé par le reste du pipeline."""
    loader = get_loader(dataset_name)
    return loader.load(subjects, runs)
