"""Verified on-demand inputs; at most two checkpoint files per host, including transfer."""
from contextlib import contextmanager
import fcntl
import hashlib
import os
from pathlib import Path
import sys


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(2**20),b''):h.update(block)
    return h.hexdigest()


def client():
    runtime=Path(os.environ['MECHANISM_CONTROL_RUNTIME'])
    sys.path.insert(0,str(runtime))
    from control_client import ControlClient
    return ControlClient(runtime=runtime)


def validate_digest(value):
    if len(value)!=64 or any(c not in '0123456789abcdef' for c in value):raise ValueError('invalid asset digest')


def fetch_locked(sha,kind,cache,transport):
    validate_digest(sha)
    folder=cache/('weights' if kind=='weight' else 'assets');folder.mkdir(parents=True,exist_ok=True)
    target=folder/(sha+'.bin')
    if target.is_file():
        if digest(target)!=sha:raise ValueError('cached input hash mismatch')
        os.utime(target,None);return target
    if kind=='weight':
        # Evict before transfer: one resident plus one .partial is the maximum of two.
        for old in sorted(folder.glob('*.bin'),key=lambda p:p.stat().st_mtime)[:-1]:old.unlink()
        for old in folder.glob('*.partial'):old.unlink()
    temporary=target.with_suffix('.partial')
    try:
        with temporary.open('wb') as stream:
            transport.request('GET','/api/mechanism/asset/'+sha,destination=stream,timeout=1200)
            stream.flush();os.fsync(stream.fileno())
        if digest(temporary)!=sha:raise ValueError('downloaded mechanism asset hash mismatch')
        temporary.replace(target);return target
    finally:temporary.unlink(missing_ok=True)


@contextmanager
def materialized(source,transport=None):
    if not source.get('remote_asset'):
        yield Path(source['path']);return
    cache=Path(os.environ['MECHANISM_CACHE']);cache.mkdir(parents=True,exist_ok=True)
    # Hold the lock until torch.load finishes: another worker cannot evict an open source.
    with (cache/'weights.lock').open('a+b') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        try:yield fetch_locked(source['sha256'],'weight',cache,transport or client())
        finally:fcntl.flock(lock,fcntl.LOCK_UN)


def fetch_asset(sha,transport=None):
    cache=Path(os.environ['MECHANISM_CACHE']);cache.mkdir(parents=True,exist_ok=True)
    with (cache/'assets.lock').open('a+b') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        try:return fetch_locked(sha,'data',cache,transport or client())
        finally:fcntl.flock(lock,fcntl.LOCK_UN)
