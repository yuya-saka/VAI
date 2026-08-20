"""natural・mixup・annotated streamの乱数状態管理。"""

from __future__ import annotations

import copy
import hashlib
import random
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import torch
from torch import Tensor


def sample_seed(
    base_seed: int,
    outer_fold: int,
    epoch: int,
    stream: str,
    ordinal: int,
) -> int:
    """sample位置からAlbumentations用の安定した32bit seedを作る。"""
    if min(base_seed, outer_fold, epoch, ordinal) < 0:
        raise ValueError("seed構成値は0以上である必要があります")
    payload = f"{base_seed}:{outer_fold}:{epoch}:{stream}:{ordinal}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little")


@dataclass(frozen=True)
class GlobalRngState:
    """checkpoint対象のglobal RNG状態。"""

    python: tuple[Any, ...]
    numpy: tuple[Any, ...]
    torch_cpu: Tensor
    torch_cuda: tuple[Tensor, ...]

    @classmethod
    def capture(cls) -> GlobalRngState:
        """現在のglobal RNG状態を複製する。"""
        cuda_states = (
            tuple(state.clone() for state in torch.cuda.get_rng_state_all())
            if torch.cuda.is_available()
            else ()
        )
        return cls(
            python=copy.deepcopy(random.getstate()),
            numpy=cast(tuple[Any, ...], copy.deepcopy(np.random.get_state())),
            torch_cpu=torch.get_rng_state().clone(),
            torch_cuda=cuda_states,
        )

    def restore(self) -> None:
        """保存したglobal RNG状態を復元する。"""
        random.setstate(self.python)
        np.random.set_state(self.numpy)
        torch.set_rng_state(self.torch_cpu)
        if self.torch_cuda:
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA RNG stateをCPU環境では復元できません")
            torch.cuda.set_rng_state_all(list(self.torch_cuda))

    def state_dict(self) -> dict[str, object]:
        """torch.save可能な辞書を返す。"""
        return {
            "python": self.python,
            "numpy": self.numpy,
            "torch_cpu": self.torch_cpu.clone(),
            "torch_cuda": [state.clone() for state in self.torch_cuda],
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, object]) -> GlobalRngState:
        """checkpoint辞書から状態を復元する。"""
        torch_cpu = state.get("torch_cpu")
        torch_cuda = state.get("torch_cuda", [])
        if not isinstance(torch_cpu, Tensor) or not isinstance(torch_cuda, list):
            raise TypeError("RNG checkpointのtensor stateが不正です")
        if not all(isinstance(value, Tensor) for value in torch_cuda):
            raise TypeError("CUDA RNG checkpointが不正です")
        numpy_state = state.get("numpy")
        python_state = state.get("python")
        if not isinstance(numpy_state, tuple) or not isinstance(python_state, tuple):
            raise TypeError("Python/NumPy RNG checkpointが不正です")
        return cls(
            python=python_state,
            numpy=numpy_state,
            torch_cpu=_cpu_rng_state(torch_cpu, "torch CPU RNG"),
            torch_cuda=tuple(
                _cpu_rng_state(value, "torch CUDA RNG") for value in torch_cuda
            ),
        )


def global_rng_states_equal(left: GlobalRngState, right: GlobalRngState) -> bool:
    """2つのglobal RNG stateが完全一致するか返す。"""
    numpy_equal = (
        left.numpy[0] == right.numpy[0]
        and np.array_equal(left.numpy[1], right.numpy[1])
        and left.numpy[2:] == right.numpy[2:]
    )
    return bool(
        left.python == right.python
        and numpy_equal
        and torch.equal(left.torch_cpu, right.torch_cpu)
        and len(left.torch_cuda) == len(right.torch_cuda)
        and all(
            torch.equal(left_state, right_state)
            for left_state, right_state in zip(
                left.torch_cuda, right.torch_cuda, strict=True
            )
        )
    )


class TrainingRngStreams:
    """mixup generatorとannotated専用torch RNGを保持する。"""

    def __init__(self, mixup_seed: int, annotated_seed: int) -> None:
        self.mixup = torch.Generator(device="cpu")
        self.mixup.manual_seed(mixup_seed)
        self._annotated_cpu = _seeded_cpu_state(annotated_seed)
        self._annotated_cuda = _seeded_cuda_states(annotated_seed)

    @contextmanager
    def annotated(self) -> Iterator[None]:
        """global torch RNGを汚さずannotated専用stateだけを進める。"""
        global_cpu = torch.get_rng_state().clone()
        global_cuda = (
            tuple(state.clone() for state in torch.cuda.get_rng_state_all())
            if torch.cuda.is_available()
            else ()
        )
        torch.set_rng_state(self._annotated_cpu)
        if self._annotated_cuda:
            torch.cuda.set_rng_state_all(list(self._annotated_cuda))
        try:
            yield
        finally:
            self._annotated_cpu = torch.get_rng_state().clone()
            if torch.cuda.is_available():
                self._annotated_cuda = tuple(
                    state.clone() for state in torch.cuda.get_rng_state_all()
                )
            torch.set_rng_state(global_cpu)
            if global_cuda:
                torch.cuda.set_rng_state_all(list(global_cuda))

    def state_dict(self) -> dict[str, object]:
        """checkpointへ保存する専用stream状態を返す。"""
        return {
            "mixup": self.mixup.get_state().clone(),
            "annotated_cpu": self._annotated_cpu.clone(),
            "annotated_cuda": [state.clone() for state in self._annotated_cuda],
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        """専用stream状態をcheckpointから復元する。"""
        mixup = state.get("mixup")
        annotated_cpu = state.get("annotated_cpu")
        annotated_cuda = state.get("annotated_cuda", [])
        if not isinstance(mixup, Tensor) or not isinstance(annotated_cpu, Tensor):
            raise TypeError("専用RNG checkpointが不正です")
        if not isinstance(annotated_cuda, list) or not all(
            isinstance(value, Tensor) for value in annotated_cuda
        ):
            raise TypeError("annotated CUDA RNG checkpointが不正です")
        self.mixup.set_state(_cpu_rng_state(mixup, "mixup RNG"))
        self._annotated_cpu = _cpu_rng_state(annotated_cpu, "annotated CPU RNG")
        self._annotated_cuda = tuple(
            _cpu_rng_state(value, "annotated CUDA RNG") for value in annotated_cuda
        )


def checkpoint_rng_state(streams: TrainingRngStreams) -> dict[str, object]:
    """globalと専用streamをまとめたcheckpoint payloadを返す。"""
    return {
        "global": GlobalRngState.capture().state_dict(),
        "streams": streams.state_dict(),
    }


def restore_checkpoint_rng_state(
    state: Mapping[str, object], streams: TrainingRngStreams
) -> None:
    """checkpoint payloadからglobalと専用streamを復元する。"""
    global_state = state.get("global")
    stream_state = state.get("streams")
    if not isinstance(global_state, Mapping) or not isinstance(stream_state, Mapping):
        raise TypeError("RNG checkpoint payloadが不正です")
    GlobalRngState.from_state_dict(global_state).restore()
    streams.load_state_dict(stream_state)


def _seeded_cpu_state(seed: int) -> Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return generator.get_state()


def _cpu_rng_state(state: Tensor, label: str) -> Tensor:
    if state.dtype != torch.uint8:
        raise TypeError(f"{label} stateはtorch.uint8である必要があります")
    return state.detach().cpu().contiguous().clone()


def _seeded_cuda_states(seed: int) -> tuple[Tensor, ...]:
    if not torch.cuda.is_available():
        return ()
    global_states = tuple(state.clone() for state in torch.cuda.get_rng_state_all())
    try:
        torch.cuda.manual_seed_all(seed)
        return tuple(state.clone() for state in torch.cuda.get_rng_state_all())
    finally:
        torch.cuda.set_rng_state_all(list(global_states))
