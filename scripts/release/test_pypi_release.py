import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import pypi_release as release


def wheel(code=b'example', commit='abc', source='source-a'):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, 'w') as archive:
        archive.writestr(release.MARKER, json.dumps({'source_sha256': source, 'git_commit': commit}))
        archive.writestr('fedops/core.py', code)
        archive.writestr('fedops-1.2.3.dist-info/METADATA', 'Name: fedops\nVersion: 1.2.3\n')
    return stream.getvalue()


class ReleaseGuardTests(unittest.TestCase):
    def test_identity(self):
        version, identity = release.wheel_identity(wheel())
        self.assertEqual(version, '1.2.3')
        self.assertEqual(identity['git_commit'], 'abc')

    def test_source_mismatch_and_missing_provenance_fail(self):
        for left, right in [({'source_sha256': 'a'}, {'source_sha256': 'b'}), ({}, {})]:
            with self.assertRaises(ValueError):
                release.check_identity(left, right)
        with self.assertRaises(KeyError):
            release.wheel_identity(self.legacy_wheel())

    def legacy_wheel(self):
        stream = io.BytesIO()
        with zipfile.ZipFile(stream, 'w') as archive:
            archive.writestr('fedops/core.py', b'old')
        return stream.getvalue()

    def test_same_source_different_commit_is_allowed_but_changed_payload_is_not(self):
        self.assertEqual(release.payload(wheel(commit='abc')), release.payload(wheel(commit='def')))
        self.assertNotEqual(release.payload(wheel()), release.payload(wheel(code=b'changed')))

    def test_existing_release_content_and_exact_artifacts(self):
        local = wheel()
        remote = wheel(commit='def')
        entry = {'filename': 'fedops-1.2.3-py3-none-any.whl',
                 'digests': {'sha256': release.hashlib.sha256(remote).hexdigest()}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, entry['filename'])
            path.write_bytes(local)
            version, identity = release.wheel_identity(local)
            with patch.object(release, 'verified_download', return_value=remote):
                release.check_published({'urls': [entry]}, version, identity, [path])
                with self.assertRaisesRegex(ValueError, 'tested artifact'):
                    release.check_published({'urls': [entry]}, version, identity, [path], exact=True)
            with patch.object(release, 'verified_download', return_value=wheel(code=b'changed')):
                with self.assertRaisesRegex(ValueError, 'contents differ'):
                    release.check_published({'urls': [entry]}, version, identity, [path])

    def test_untrusted_host_and_yanked_file_fail(self):
        for entry in [{'url': 'https://example.com/a.whl'},
                      {'url': 'https://files.pythonhosted.org/a.whl', 'yanked': True}]:
            with self.assertRaises(ValueError):
                release.verified_download(entry)


if __name__ == '__main__':
    unittest.main()
