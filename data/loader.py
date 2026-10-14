# """
# data/loader.py

# Chargement des données EEG et abstraction du dataset source.

# - Gère le chargement d'un ou plusieurs sujets / runs.
# - Concatène automatiquement plusieurs Raw en un seul objet MNE.
# - Expose une interface générique (BaseDatasetLoader) afin qu'un autre
#   dataset que PhysioNet eegbci puisse être branché sans modifier le
#   reste du pipeline (bonus "autres datasets").
# """

# from abc import ABC, abstractmethod
# from typing import List

# import mne
# from mne.datasets import eegbci


# # ---------------------------------------------------------------------------
# # Interface générique (permet le bonus "autres datasets")
# # ---------------------------------------------------------------------------
# class BaseDatasetLoader(ABC):
#     """
#     Interface commune que doit respecter tout loader de dataset EEG.
#     Un nouveau dataset (bonus) s'intègre en implémentant cette interface,
#     sans toucher au reste du pipeline (preprocessing, features, CSP...).
#     """

#     @abstractmethod
#     def load(self, subjects: List[int], runs: List[int]) -> mne.io.Raw:
#         """Retourne un objet mne.io.Raw concaténé pour les sujets/runs donnés."""
#         raise NotImplementedError


# # ---------------------------------------------------------------------------
# # Implémentation pour PhysioNet eegbci (dataset demandé par le sujet)
# # ---------------------------------------------------------------------------
# class PhysionetLoader(BaseDatasetLoader):
#     """
#     Loader pour le "Et" (PhysioNet, eegbci),
#     fichiers SxxxRyy.edf + annotations associées.
#     """

#     MONTAGE_NAME = "standard_1005"

#     def load(self, subjects: List[int], runs: List[int]) -> mne.io.Raw:
#         """
#         Télécharge (si nécessaire) et concatène les Raw de plusieurs
#         sujets et runs en un seul objet MNE, prêt pour le filtrage
#         et le découpage en epochs.

#         Note : concaténer des sujets différents dans un même Raw n'a de
#         sens que si vous entraînez un modèle "cross-subject". Pour un
#         entraînement classique sujet par sujet, appelez cette méthode
#         avec une liste subjects=[un seul sujet] et plusieurs runs.
#         """
#         raw_list = []
#         for subject in subjects:
#             paths = eegbci.load_data(subject, runs, path=None, update_path=True)
#             for path in paths:
#                 raw = mne.io.read_raw_edf(path, preload=True, verbose=False)
#                 eegbci.standardize(raw)
#                 raw_list.append(raw)

#         raw_concat = mne.concatenate_raws(raw_list, verbose=False)

#         montage = mne.channels.make_standard_montage(self.MONTAGE_NAME)
#         raw_concat.set_montage(montage)

#         return raw_concat


# # ---------------------------------------------------------------------------
# # Registre de datasets : point d'extension pour le bonus "autres datasets"
# # ---------------------------------------------------------------------------
# DATASET_REGISTRY = {
#     "physionet": PhysionetLoader,
#     # "mon_autre_dataset": MonAutreDatasetLoader,  # bonus : ajout futur
# }


# def get_loader(dataset_name: str = "physionet") -> BaseDatasetLoader:
#     """Factory renvoyant le loader correspondant au dataset demandé."""
#     if dataset_name not in DATASET_REGISTRY:
#         raise ValueError(
#             f"Dataset '{dataset_name}' inconnu. "
#             f"Disponibles : {list(DATASET_REGISTRY.keys())}"
#         )
#     return DATASET_REGISTRY[dataset_name]()


# def load_raw(subjects: List[int], runs: List[int], dataset_name: str = "physionet") -> mne.io.Raw:
#     """Point d'entrée simple utilisé par le reste du pipeline."""
#     loader = get_loader(dataset_name)
#     return loader.load(subjects, runs)

"""
data/loader.py

Chargement des données EEG et abstraction du dataset source.

CORRECTIF IMPORTANT : la version précédente concaténait sujets ET runs
ensemble dans un seul Raw, ce qui mélange les propriétés spatiales
individuelles de chaque sujet et empêche le CSP de trouver une
combinaison de canaux discriminante (validé empiriquement : accuracy
retombée au niveau du hasard en concaténant 4 sujets).

Règle désormais appliquée :
    - Concaténer plusieurs RUNS d'un même sujet : valide, encouragé
      (plus d'exemples pour le même modèle spatial).
    - Concaténer plusieurs SUJETS dans un même Raw : jamais fait ici.
      Chaque sujet reste un Raw séparé ; c'est au niveau de
      l'ÉVALUATION (Phase 3, boucle multi-sujets) que les résultats
      de plusieurs sujets sont agrégés, jamais leurs données brutes.
"""

from abc import ABC, abstractmethod
from typing import Dict, List

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
    def load_subject(self, subject: int, runs: List[int]) -> mne.io.Raw:
        """
        Retourne un objet mne.io.Raw pour UN SEUL sujet, avec ses runs
        concaténés entre eux (jamais avec un autre sujet).
        """
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

    def load_subject(self, subject: int, runs: List[int]) -> mne.io.Raw:
        """
        Télécharge (si nécessaire) et concatène UNIQUEMENT les runs
        d'un même sujet en un seul objet MNE. C'est une concaténation
        valide : les runs partagent la même géométrie de canaux et le
        même sujet, donc la même signature spatiale à apprendre par le CSP.
        """
        paths = eegbci.load_data(subject, runs, path=None, update_path=True)

        raw_list = []
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


# ---------------------------------------------------------------------------
# Points d'entrée utilisés par le reste du pipeline
# ---------------------------------------------------------------------------
def load_raw(subject: int, runs: List[int], dataset_name: str = "physionet") -> mne.io.Raw:
    """
    Charge UN SEUL sujet (runs concaténés). C'est la fonction à utiliser
    pour entraîner/évaluer un modèle CSP sur un sujet donné.
    """
    loader = get_loader(dataset_name)
    return loader.load_subject(subject, runs)


def load_raw_multi_subjects(
    subjects: List[int], runs: List[int], dataset_name: str = "physionet"
) -> Dict[int, mne.io.Raw]:
    """
    Charge plusieurs sujets, chacun dans SON PROPRE Raw (dictionnaire
    {subject_id: Raw}), sans jamais les concaténer entre eux.

    À utiliser pour la boucle d'évaluation multi-sujets (Phase 3) :
    un CSP + un classifieur sont entraînés séparément pour chaque
    sujet, puis les accuracies obtenues sont moyennées — jamais les
    données brutes.
    """
    loader = get_loader(dataset_name)
    return {subject: loader.load_subject(subject, runs) for subject in subjects}
