"""The 12 uncontrolled searcher configurations of section 5 (control C0).

Each entry maps a configuration name to (run function, keyword arguments). Every run function has
the signature run(evaluator, K, rng, **kwargs) -> SearchResult, where rng is a numpy Generator
created from the search seed.
"""

from . import bpso, de, ga, gwo, hill_climb, one_plus_one, random_search, woa
from ._base import SearchResult

CONFIGS = {
    "random": (random_search.run, {}),
    "hill_climb": (hill_climb.run, {}),
    "one_plus_one": (one_plus_one.run, {}),
    "ga": (ga.run, {}),
    "bpso_S": (bpso.run, {"transfer": "S"}),
    "bpso_V": (bpso.run, {"transfer": "V"}),
    "de_S": (de.run, {"transfer": "S"}),
    "de_V": (de.run, {"transfer": "V"}),
    "gwo_S": (gwo.run, {"transfer": "S"}),
    "gwo_V": (gwo.run, {"transfer": "V"}),
    "woa_S": (woa.run, {"transfer": "S"}),
    "woa_V": (woa.run, {"transfer": "V"}),
}

CONTINUOUS_CORES = {"bpso": bpso.minimize, "de": de.minimize, "gwo": gwo.minimize, "woa": woa.minimize}


def run_config(name, evaluator, K, rng):
    fn, kwargs = CONFIGS[name]
    return fn(evaluator, K, rng, **kwargs)


__all__ = ["CONFIGS", "CONTINUOUS_CORES", "SearchResult", "run_config"]
