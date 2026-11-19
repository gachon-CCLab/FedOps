"""Read required internal service URLs from the environment."""
import os
from urllib.parse import urlsplit


def internal_url(key):
    value = os.getenv(key, '').strip().rstrip('/')
    parsed = urlsplit(value)
    if parsed.scheme not in ('http', 'https') or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError(f'{key} must be an HTTP(S) base URL without credentials, query, or fragment')
    return value
