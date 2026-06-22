"""
baselines.py – thin re-export facade.

Each algorithm lives in its own module:

    models/akom.py          AKOM  (Pitkow & Pirolli 1999)
    models/cpt.py           CPT+  (Gueniche et al. 2013)
    models/mogen.py         MOGen (Gote et al. 2023)
    models/iohmm.py         IOHMM (Mo et al. 2022)
    models/simple_markov.py SimpleMarkov (fixed-order Markov, no back-off)

Importing from this module preserves backwards compatibility.
"""

from __future__ import annotations

from ._base       import _BaselineModel
from .akom        import AKOM
from .cpt         import CPTPlus
from .mogen       import MOGen
from .iohmm       import IOHMM
from .simple_markov import SimpleMarkov
from typing import List


def get_baseline(name: str, **kwargs) -> _BaselineModel:
    """
    Instantiate a baseline model by name.

    Parameters
    ----------
    name : str
        One of ``'akom'``, ``'cpt'``, ``'mogen'``, ``'iohmm'``.
    **kwargs
        Keyword arguments forwarded to the model constructor.

    Returns
    -------
    _BaselineModel
    """
    registry = {
        "akom":  AKOM,
        "cpt":   CPTPlus,
        "cptplus": CPTPlus,
        "mogen": MOGen,
        "iohmm": IOHMM,
        "mc1":   lambda **kw: SimpleMarkov(order=1),
        "mc1_ge": lambda **kw: SimpleMarkov(order=1),
        "mc5":   lambda **kw: SimpleMarkov(order=5),
        "mc10":  lambda **kw: SimpleMarkov(order=10),
    }
    key = name.lower().strip()
    if key not in registry:
        raise ValueError(
            f"Unknown baseline '{name}'. Choose from: {list(registry.keys())}"
        )
    return registry[key](**kwargs)


__all__ = ["AKOM", "CPTPlus", "MOGen", "IOHMM", "SimpleMarkov", "get_baseline"]
