"""
preprocessing/epoching.py

Extraction des événements depuis les annotations et découpage
du signal continu en epochs labellisées.
"""

import mne


def make_epochs(
    raw: mne.io.Raw,
    tmin: float = 0.0,
    tmax: float = 4.0,
    exclude_rest: bool = True,
) -> mne.Epochs:
    """
    Découpe le signal continu en epochs autour de chaque événement.

    Labels du dataset eegbci :
        T0 : repos
        T1 : mouvement/imagination (poing gauche, ou deux poings selon le run)
        T2 : mouvement/imagination (poing droit, ou deux pieds selon le run)

    Si exclude_rest=True, la classe de repos (T0) est retirée pour ne garder
    que les deux classes de mouvement (classification binaire A vs B,
    conformément à l'énoncé du sujet).
    """
    events, event_id = mne.events_from_annotations(raw, verbose=False)

    if exclude_rest and "T0" in event_id:
        event_id = {k: v for k, v in event_id.items() if k != "T0"}

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
