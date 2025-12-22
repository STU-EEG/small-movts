from pathlib import Path
import json
from typing import Any, Callable, Dict, Final, List

import numpy as np

from signals import AbstractSignal


class MultimodalDataset:
    _signals: Dict[str, AbstractSignal]

    def load(
        self, config: Dict[str, Any], signals_getter: Callable[[str], AbstractSignal]
    ) -> None:
        signals_key: Final[str] = "signals"
        title_key: Final[str] = "title"

        if signals_key not in config:
            raise ValueError(f"a key {signals_key} must exist in the config")
        signals_descs = config[signals_key]

        if not isinstance(signals_descs, list):
            raise ValueError(f"config[{signals_key}] must be an array")

        self._signals = {}
        for s_desc in signals_descs:
            if not isinstance(s_desc, dict):
                raise ValueError(
                    f"each element of config[{signals_key}] must be an object"
                )

            if title_key not in s_desc:
                raise ValueError(f"key {title_key} must exist in each signals entry")

            title = s_desc[title_key]
            if title in self._signals:
                raise ValueError(f"duplicate key {title} not allowed")

            self._signals[title] = signals_getter(title)

    def __init__(
        self, directory: Path, signals_getter: Callable[[str], AbstractSignal]
    ) -> None:
        if not directory.is_dir():
            raise ValueError("`directory` must be an existing directory")

        config_path = directory / "dataset_config.json"
        if not config_path.exists():
            raise ValueError(f"file {config_path} must exist")

        with config_path.open() as config_file:
            config_dict = json.load(config_file)

        if not isinstance(config_dict, dict):
            raise ValueError("expecting an object at the top level of JSON config")

        self.load(config_dict, signals_getter)

    def raw_signals(self, start: float, end: float) -> Dict[str, np.floating]:
        return {k: v.get_raw(start, end) for k, v in self._signals.items()}
