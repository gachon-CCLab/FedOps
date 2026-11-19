"""Optional endpoint inputs; unchanged AWS destination when no endpoint is set."""
import os
from urllib.parse import urlsplit
from botocore.config import Config


def s3_options():
    options = {}
    endpoint = os.getenv('S3_ENDPOINT_URL', '').strip()
    if endpoint:
        url = urlsplit(endpoint)
        if url.scheme not in ('http', 'https') or not url.hostname or url.username or url.password or url.query or url.fragment:
            raise ValueError('S3_ENDPOINT_URL must be an HTTP(S) URL without credentials, query, or fragment')
        options['endpoint_url'] = endpoint.rstrip('/')
    style = os.getenv('S3_FORCE_PATH_STYLE', 'true' if endpoint else None)
    if style is not None:
        if style not in ('true', 'false'):
            raise ValueError('S3_FORCE_PATH_STYLE must be true or false')
        options['config'] = Config(s3={'addressing_style': 'path' if style == 'true' else 'virtual'})
    return options
