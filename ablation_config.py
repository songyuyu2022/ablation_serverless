# ablation_config.py
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class AblationConfig:
    """Central switch for ablation experiments.

    ABLATION_MODE:
      - full:           original method (hot/cold + async cold accumulation + hybrid scheduler + NSGA-II for grad selection)
      - no_hotcold:     disable expert hot/cold identification; comm treated as unified http
      - sync_update:    force all experts to update synchronously every step (no cold accumulation)
      - heuristic_only: keep heuristic scheduler only (disable NN predictions + online update)
      - predictor_only: keep NN scheduler only (disable heuristic base score)
      - no_nsga2:       disable NSGA-II; use scheduler selection for grad instance selection
    """

    mode: str = "full"

    @property
    def disable_hotcold(self) -> bool:
        return self.mode == "no_hotcold"

    @property
    def force_sync_update(self) -> bool:
        return self.mode == "sync_update"

    @property
    def heuristic_only(self) -> bool:
        return self.mode == "heuristic_only"

    @property
    def predictor_only(self) -> bool:
        return self.mode == "predictor_only"

    @property
    def disable_nsga2(self) -> bool:
        return self.mode == "no_nsga2"


def load_ablation_from_env() -> AblationConfig:
    mode = (os.getenv("ABLATION_MODE", "full") or "full").strip().lower()
    valid = {"full", "no_hotcold", "sync_update", "heuristic_only", "predictor_only", "no_nsga2"}
    if mode not in valid:
        raise ValueError(f"Invalid ABLATION_MODE={mode}, valid={sorted(valid)}")
    return AblationConfig(mode=mode)
