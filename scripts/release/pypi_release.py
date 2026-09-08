"""Fail closed when Git's package and an existing PyPI version disagree."""
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from urllib.error import HTTPError
from urllib.request import urlopen
import zipfile

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / 'src/python'
MARKER = 'fedops/_release.json'


def source_digest():
    names = subprocess.check_output(
        ['git', 'ls-files', '-z', 'src/python/fedops', 'src/python/setup.py',
         'src/python/setup.cfg', 'src/python/README.md'], cwd=ROOT,
    ).decode().split('\0')
    digest = hashlib.sha256()
    for name in sorted(filter(None, names)):
        if '__pycache__' in Path(name).parts or name.endswith(('.pyc', '_release.json')):
            continue
        content = (ROOT / name).read_bytes()
        digest.update(name.encode() + b'\0' + hashlib.sha256(content).digest())
    return digest.hexdigest()


def stamp():
    identity = {'source_sha256': source_digest(), 'git_commit': subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()}
    (PACKAGE / MARKER).write_text(json.dumps(identity, sort_keys=True) + '\n')


def wheel_identity(content):
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        identity = json.loads(archive.read(MARKER))
        metadata_name, = [n for n in archive.namelist() if n.endswith('.dist-info/METADATA')]
        from email.parser import BytesParser
        metadata = BytesParser().parsebytes(archive.read(metadata_name))
        if metadata['Name'].lower() != 'fedops':
            raise ValueError('Unexpected distribution name')
        return metadata['Version'], identity


def release_info(version):
    try:
        # Avoid a pre-publication 404 cached by an intermediary/CDN when the
        # immediately following verify reads the newly published version.
        with urlopen(f'https://pypi.org/pypi/fedops/{version}/json?verify={time.time_ns()}', timeout=30) as response:
            return json.load(response)
    except HTTPError as error:
        if error.code == 404:
            return None
        raise


def check_identity(local, published):
    if local.get('source_sha256') != published.get('source_sha256'):
        raise ValueError('This PyPI version contains different source. Bump setup.py version; never overwrite or silently skip it.')
    if not local.get('source_sha256'):
        raise ValueError('Missing source identity')


def verified_download(entry):
    # Only public PyPI distribution files, never arbitrary redirect input.
    if not entry['url'].startswith('https://files.pythonhosted.org/'):
        raise ValueError('Unexpected PyPI artifact host')
    if entry.get('yanked'):
        raise ValueError('Published artifact is yanked')
    with urlopen(entry['url'], timeout=60) as response:
        content = response.read()
    if hashlib.sha256(content).hexdigest() != entry['digests']['sha256']:
        raise ValueError('PyPI artifact checksum mismatch')
    return content


def payload(content):
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        return {name: archive.read(name) for name in archive.namelist()
                if name != MARKER and not name.endswith(('.dist-info/RECORD', '.dist-info/WHEEL'))}


def check_published(info, version, identity, distributions, exact=False):
    remote_wheel, = [entry for entry in info['urls'] if entry['filename'].endswith('.whl')]
    remote_content = verified_download(remote_wheel)
    remote_version, remote_identity = wheel_identity(remote_content)
    if remote_version != version:
        raise ValueError('Published version mismatch')
    check_identity(identity, remote_identity)
    local_wheel, = [path for path in distributions if path.suffix == '.whl']
    if payload(remote_content) != payload(local_wheel.read_bytes()):
        raise ValueError('Published wheel contents differ from this build; bump version')
    entries = {entry['filename']: entry for entry in info['urls']}
    for path in distributions:
        entry = entries.get(path.name)
        if not entry or entry.get('yanked'):
            raise ValueError('Incomplete or yanked PyPI release')
        # Fresh publication must match every tested artifact byte for byte.
        # An already-published rerun compares source and wheel payload instead:
        # ZIP timestamps and the producing commit can legitimately differ.
        if exact:
            if hashlib.sha256(path.read_bytes()).hexdigest() != entry['digests']['sha256']:
                raise ValueError('Uploaded distribution differs from the tested artifact')


def main(command):
    if command == 'stamp':
        stamp()
        return
    distributions = sorted((ROOT / 'dist').glob('*'))
    wheel, = [path for path in distributions if path.suffix == '.whl']
    version, identity = wheel_identity(wheel.read_bytes())
    info = release_info(version)
    if command == 'verify':
        for _ in range(12):
            if info is not None:
                break
            time.sleep(10)
            info = release_info(version)
        if info is None:
            raise ValueError('Release did not appear on public PyPI')
    if info is not None:
        check_published(info, version, identity, distributions,
                        exact=command == 'verify' and os.environ.get('EXACT_ARTIFACTS') == 'true')
    if command == 'plan':
        if identity['source_sha256'] != source_digest():
            raise ValueError('Wheel does not match the checked out package source')
        output = f'version={version}\npublish={str(info is None).lower()}\n'
        if os.environ.get('GITHUB_OUTPUT'):
            with open(os.environ['GITHUB_OUTPUT'], 'a') as stream:
                stream.write(output)
        print(output, end='')
    elif command == 'verify':
        print(f'PyPI fedops {version}: package source and distribution checks passed')
    else:
        raise ValueError('Expected stamp, plan or verify')


if __name__ == '__main__':
    main(sys.argv[1])
