from pathlib import Path
import json


ROOT = Path(__file__).resolve().parents[1]


def load_defaults(path=None):
    path = Path(path) if path else ROOT / "default.yaml"
    with path.open("r", encoding="utf-8") as handle:
        # JSON is a valid subset of YAML. Keeping default.yaml JSON-compatible
        # avoids adding a parser dependency to the reproduction environment.
        return json.load(handle)


def load_experiment_configs(path=None):
    path = Path(path) if path else ROOT / "configs.json"
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)
