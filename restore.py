"""Restore the complete three-directory delivery from local distribution files."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import stat
import tempfile
import zipfile


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def relative(root, name):
    path = PurePosixPath(name)
    if path.is_absolute() or '..' in path.parts or not path.parts:
        raise ValueError('Invalid package path: ' + name)
    return root.joinpath(*path.parts)


def read_block(stream, length):
    pieces = []
    remaining = length
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            break
        pieces.append(chunk)
        remaining -= len(chunk)
    return b''.join(pieces)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path('delivery-restored'))
    args = parser.parse_args()
    kit = Path(__file__).resolve().parent
    destination = args.output.resolve()
    if destination.exists() or destination.is_symlink():
        raise ValueError('Choose a new output directory; existing results are not overwritten.')
    index = json.loads((kit / 'distribution.json').read_text())
    core = relative(kit, index['main']['file'])
    if digest(core) != index['main']['sha256']:
        raise ValueError('Main package hash differs from the distribution manifest.')
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.restore-', dir=destination.parent) as work:
        restored = Path(work) / 'delivery'
        restored.mkdir()
        with zipfile.ZipFile(core) as archive:
            seen = set()
            for item in archive.infolist():
                if item.filename in seen:
                    raise ValueError('Duplicate ZIP entry: ' + item.filename)
                seen.add(item.filename)
                target = relative(restored, item.filename)
                mode = item.external_attr >> 16
                if item.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                if mode and not stat.S_ISREG(mode):
                    raise ValueError('Non-regular ZIP entry: ' + item.filename)
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(item) as source, target.open('xb') as output:
                    shutil.copyfileobj(source, output, 4 * 1024 * 1024)
                target.chmod(stat.S_IMODE(mode) or 0o644)
        for asset in index['assets']:
            first = relative(restored, asset['targets'][0]['path'])
            first.parent.mkdir(parents=True, exist_ok=True)
            identity = hashlib.sha256()
            total = 0
            with first.open('xb') as output:
                for part in asset['parts']:
                    source = relative(kit, part['file'])
                    if digest(source) != part['sha256'] or source.stat().st_size != part['bytes']:
                        raise ValueError('Asset part hash or size mismatch: ' + part['file'])
                    count = 0
                    with gzip.open(source, 'rb') as decoded:
                        while block := read_block(decoded, index['block_bytes']):
                            if asset['encoding'] == 'byte_shuffle_4':
                                if len(block) % 4:
                                    raise ValueError('Invalid shuffled block length')
                                plane = len(block) // 4
                                raw = bytearray(len(block))
                                for lane in range(4):
                                    raw[lane::4] = block[lane * plane:(lane + 1) * plane]
                            elif asset['encoding'] == 'identity':
                                raw = block
                            else:
                                raise ValueError('Unsupported asset encoding')
                            count += len(raw)
                            total += len(raw)
                            identity.update(raw)
                            output.write(raw)
                    if count != part['raw_bytes']:
                        raise ValueError('Decoded asset part size mismatch')
            if total != asset['raw_bytes'] or identity.hexdigest() != asset['sha256']:
                raise ValueError('Restored asset hash mismatch')
            first.chmod(asset['targets'][0]['mode'])
            # Independent copies preserve separate Docker build contexts and edit semantics.
            for entry in asset['targets'][1:]:
                target = relative(restored, entry['path'])
                if target.exists():
                    raise ValueError('Overlapping distribution entries: ' + entry['path'])
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(first, target)
                target.chmod(entry['mode'])
        manifest = relative(restored, index['delivery_manifest']['path'])
        if digest(manifest) != index['delivery_manifest']['sha256']:
            raise ValueError('Delivery manifest changed')
        expected = json.loads(manifest.read_text())
        expected[index['delivery_manifest']['path']] = index['delivery_manifest']
        files = {p.relative_to(restored).as_posix(): p for p in restored.rglob('*') if p.is_file()}
        if set(files) != set(expected):
            raise ValueError('Restored delivery file set differs from the original manifest')
        for name, record in expected.items():
            path = files[name]
            if path.stat().st_size != record['bytes'] or digest(path) != record['sha256']:
                raise ValueError('Final file identity mismatch: ' + name)
            if stat.S_IMODE(path.stat().st_mode) != index['file_modes'][name]:
                raise ValueError('Restored permission mismatch: ' + name)
        if {p.name for p in restored.iterdir()} != {'workspace', 'expert_evidence', 'optimization_evidence'}:
            raise ValueError('Restored top-level layout differs')
        restored.rename(destination)
    print(json.dumps({'status': 'complete_delivery_restored', 'output': str(destination),
                      'files': len(expected), 'all_file_hashes_passed': True,
                      'all_file_modes_passed': True, 'training_runs': 0}, ensure_ascii=False))


if __name__ == '__main__':
    main()
