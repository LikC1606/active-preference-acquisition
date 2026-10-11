"""Load frozen public embeddings and trusted private rewards separately."""
import errno
import hashlib
import json
from pathlib import Path

SCHEMAS = {
    "train-embeddings.npy": ([10, 33500, 2048], "float32", 2744320128),
    "dev-embeddings.npy": ([10, 4188, 2048], "float32", 343081088),
    "train-rewards.npy": ([10, 33500], "float64", 2680128),
    "dev-rewards.npy": ([10, 4188], "float64", 335168),
}


def _json(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 262144:
        raise ValueError("Oracle manifest must be a bounded regular file")
    def pairs(rows):
        value = {}
        for key, item in rows:
            if key in value:
                raise ValueError("Duplicate oracle manifest key")
            value[key] = item
        return value
    return json.loads(path.read_text(), object_pairs_hook=pairs,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def _sha(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
            result.update(chunk)
    return result.hexdigest()


def _members(manifest, names):
    files = manifest.get("files")
    if (not isinstance(files, list) or len(files) != len(names)
            or any(not isinstance(item, dict) for item in files)
            or {item.get("file") for item in files} != set(names)):
        raise ValueError("Oracle/public dataset allowlist changed")
    return {item["file"]: item for item in files}


def arrays(root, public):
    import numpy as np
    root, public = Path(root), Path(public)
    frozen_public = _json(root / "public_manifest.json")
    if _json(public / "data_manifest.json") != frozen_public:
        raise ValueError("Public embedding manifest differs from its frozen identity")
    if {path.name for path in public.iterdir()} != {
        "train-embeddings.npy", "dev-embeddings.npy", "data_manifest.json", "protocol.json",
    }:
        raise ValueError("Public projection contains unexpected data or rewards")
    oracle = root / "oracle"
    frozen_oracle = _json(oracle / "manifest.json")
    if frozen_oracle.get("schema") != "arm.trusted-oracle-data.v12":
        raise ValueError("Private oracle manifest schema changed")
    result = {}
    for directory, members in (
        (public, _members(frozen_public, ("train-embeddings.npy", "dev-embeddings.npy"))),
        (oracle, _members(frozen_oracle, ("train-rewards.npy", "dev-rewards.npy"))),
    ):
        for name, item in members.items():
            shape, dtype, size = SCHEMAS[name]
            if (item.get("shape"), item.get("dtype"), item.get("bytes")) != (shape, dtype, size):
                raise ValueError("Frozen dataset dimensions changed: " + name)
            path = directory / name
            if path.is_symlink() or not path.is_file() or path.stat().st_size != size:
                raise ValueError("Dataset file identity changed: " + name)
            if _sha(path) != item.get("sha256"):
                raise ValueError("Dataset digest changed: " + name)
            try:
                array = np.load(path, mmap_mode="r", allow_pickle=False)
            except OSError as exc:
                if exc.errno != errno.ENODEV:
                    raise
                array = np.load(path, allow_pickle=False)
            if list(array.shape) != shape or str(array.dtype) != dtype:
                raise ValueError("Dataset array schema changed: " + name)
            array.flags.writeable = False
            result[name.removesuffix(".npy")] = array
    return result


def hidden_arrays(directory):
    """Load the independent held-out split only inside the final verifier."""
    import numpy as np
    directory = Path(directory)
    manifest = _json(directory / "manifest.json")
    if manifest.get("schema") != "arm.hidden-data.v1" or manifest.get("split") != "hidden":
        raise ValueError("Invalid Hidden manifest")
    names = ("hidden-embeddings.npy", "hidden-rewards.npy")
    result = {}
    for name, item in _members(manifest, names).items():
        shape, dtype, size = (SCHEMAS["dev-embeddings.npy"] if name.endswith("embeddings.npy")
                              else SCHEMAS["dev-rewards.npy"])
        path = directory / name
        if ((item.get("shape"), item.get("dtype"), item.get("bytes")) != (shape, dtype, size)
                or path.is_symlink() or not path.is_file() or path.stat().st_size != size
                or _sha(path) != item.get("sha256")):
            raise ValueError("Hidden data identity changed: " + name)
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        if list(array.shape) != shape or str(array.dtype) != dtype:
            raise ValueError("Hidden array schema changed: " + name)
        array.flags.writeable = False
        result[name.removesuffix(".npy")] = array
    return result
