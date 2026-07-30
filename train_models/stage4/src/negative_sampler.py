"""Epoch-wise level-matched Stage4 negative sampling."""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path
from typing import Any


def _item_key(item: dict[str, Any]) -> tuple[str, str]:
    return str(item["study_uid"]), str(item["vertebra"])


class NegativeRegionSampler:
    """Sample one level-matched negative bag for every strong positive bag."""

    def __init__(
        self,
        strong_items: list[dict[str, Any]],
        negative_items: list[dict[str, Any]],
        manifest_dir: Path,
        seed: int = 42,
        write_manifest: bool = True,
    ) -> None:
        if any(item.get("region_supervision") != "strong" for item in strong_items):
            raise ValueError("strong_items contains non-strong supervision")
        if any(item.get("region_supervision") != "negative" for item in negative_items):
            raise ValueError("negative_items contains non-negative supervision")
        negative_keys = [_item_key(item) for item in negative_items]
        if len(negative_keys) != len(set(negative_keys)):
            raise ValueError("negative_items contains duplicate bags")
        self.strong_items = list(strong_items)
        self.negative_items = list(negative_items)
        self.manifest_dir = manifest_dir
        self.seed = seed
        self.write_manifest = write_manifest

    def sample(self, epoch: int) -> list[dict[str, Any]]:
        """Sample without replacement and persist the selected bag manifest."""
        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        epoch_seed = self.seed + epoch
        random_generator = random.Random(epoch_seed)
        required = Counter(str(item["vertebra"]) for item in self.strong_items)
        candidates: dict[str, list[dict[str, Any]]] = {
            level: [
                item for item in self.negative_items if str(item["vertebra"]) == level
            ]
            for level in required
        }
        for level, count in required.items():
            if len(candidates[level]) < count:
                raise ValueError(
                    f"not enough negative {level} bags: "
                    f"required={count} available={len(candidates[level])}"
                )
            random_generator.shuffle(candidates[level])

        selected: list[dict[str, Any]] = []
        selected_keys: set[tuple[str, str]] = set()
        used_patients: set[str] = set()
        remaining = Counter(required)
        levels = sorted(required)

        while any(remaining.values()):
            random_generator.shuffle(levels)
            progressed = False
            for level in levels:
                if remaining[level] == 0:
                    continue
                candidate = next(
                    (
                        item
                        for item in candidates[level]
                        if _item_key(item) not in selected_keys
                        and str(item["study_uid"]) not in used_patients
                    ),
                    None,
                )
                if candidate is None:
                    continue
                selected.append(candidate)
                selected_keys.add(_item_key(candidate))
                used_patients.add(str(candidate["study_uid"]))
                remaining[level] -= 1
                progressed = True
            if not progressed:
                break

        for level in levels:
            if remaining[level] == 0:
                continue
            available = [
                item
                for item in candidates[level]
                if _item_key(item) not in selected_keys
            ]
            chosen = available[: remaining[level]]
            selected.extend(chosen)
            selected_keys.update(_item_key(item) for item in chosen)
            remaining[level] -= len(chosen)

        if any(remaining.values()):
            raise RuntimeError(f"negative sampling did not satisfy counts: {remaining}")
        if self.write_manifest:
            self._save_manifest(epoch, epoch_seed, selected)
        return selected

    def _save_manifest(
        self,
        epoch: int,
        epoch_seed: int,
        selected: list[dict[str, Any]],
    ) -> None:
        self.manifest_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "epoch": epoch,
            "seed": epoch_seed,
            "n_strong": len(self.strong_items),
            "n_negative": len(selected),
            "bags": [
                {
                    "study_uid": str(item["study_uid"]),
                    "vertebra": str(item["vertebra"]),
                }
                for item in selected
            ],
        }
        path = self.manifest_dir / f"negative_manifest_epoch{epoch}.json"
        with path.open("w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2)
