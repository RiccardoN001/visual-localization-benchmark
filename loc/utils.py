import ast
import inspect
import json
import logging
import os
from pathlib import Path


def setup_logging(level=logging.INFO):
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path):
    """Atomic write: readers see either the complete file or no file at all.

    Parallel jobs share the retrieval cache, and a job killed by its walltime must not leave a
    truncated result that --resume would later treat as valid.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, default=str)
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def check_options(cls, options):
    """Reject options that are not named arguments of the constructor, instead of ignoring them."""
    params = inspect.signature(cls.__init__).parameters
    accepted = {n for n, p in params.items() if n not in ("self", "device")
                and p.kind not in (p.VAR_KEYWORD, p.VAR_POSITIONAL)}
    unknown = set(options) - accepted
    if unknown:
        raise SystemExit(f"{cls.__name__} does not accept {sorted(unknown)}: "
                         f"valid options are {sorted(accepted)}")


def parse_options(pairs):
    """['max_kpts=1024', 'resize=None'] -> {'max_kpts': 1024, 'resize': None}."""
    options = {}
    for pair in pairs or []:
        key, sep, value = pair.partition("=")
        if not sep or not key:
            raise SystemExit(f"invalid option '{pair}': expected key=value")
        try:
            options[key.strip()] = ast.literal_eval(value.strip())
        except (ValueError, SyntaxError):
            options[key.strip()] = value.strip()
    return options
