"""Corpus source abstraction for the arbez benchmark (S-061).

Pre-S-061, ``arbez_benchmark.py`` had a hard-coded local directory
(``DEFAULT_CORPUS``) and used ``Path.iterdir()`` to enumerate the
top-level files only. This module replaces both: a corpus is now
identified by a URI string the user supplies via ``--corpus``, and
each backend walks recursively.

Supported URI schemes:

* ``/local/path`` or ``./relative/path`` — a local filesystem
  directory. The bare-path form (no scheme) is the default; for
  explicitness it's also accepted as ``file:///abs/path``.
* ``s3://bucket-name/optional/prefix/`` — an AWS S3 bucket
  (optionally with a key prefix). Auth via boto3's standard
  credential provider chain (env vars, ``~/.aws/credentials``,
  IAM instance profile, etc.). ``boto3`` must be installed; the
  benchmark's ``examples/`` directory is dev-only so the dep is
  loaded lazily — pip-install ``boto3`` separately when needed.
* ``b2://bucket-name/optional/prefix/`` — a Backblaze B2 bucket.
  Auth via b2sdk's standard credential lookup: env vars
  (``B2_APPLICATION_KEY_ID`` + ``B2_APPLICATION_KEY``) take
  priority, falling back to the persistent account-info file
  populated by the ``b2 authorize-account`` CLI. Requires
  ``b2sdk``.

Remote items are downloaded to a per-source local cache the first
time they're materialized; subsequent runs reuse the cache. The
cache lives under ``~/.cache/arbez-benchmark/<scheme>/<bucket>/``.

Side effect: importing this module DOES NOT trigger any HEIC/AVIF
plugin registration — callers should do that themselves (the
benchmark does so via the existing ``_ensure_pillow_plugins``
path). This keeps the module dependency-light and reusable in
tools/scripts that don't need image-format probing.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Iterable


# ── Accepted file extensions (case-insensitive) ───────────────────────

#: Core image formats Pillow decodes without optional plugins.
CORE_EXTENSIONS: frozenset[str] = frozenset({
    ".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff",
    ".bmp", ".gif", ".ico", ".ppm",
})

#: HEIC formats — enabled when ``pillow-heif`` is installed.
HEIC_EXTENSIONS: frozenset[str] = frozenset({".heic", ".heif"})

#: AVIF formats — enabled when ``pillow-avif-plugin`` is installed.
AVIF_EXTENSIONS: frozenset[str] = frozenset({".avif"})


def accepted_extensions(
    include_heic: bool = True, include_avif: bool = True,
) -> frozenset[str]:
    """Return the set of file extensions a corpus source should include.

    Defaults to core + HEIC + AVIF. Callers that want to constrain
    (e.g. for a specific test or when a plugin isn't installed) can
    pass ``include_heic=False`` / ``include_avif=False``.
    """
    accepted = set(CORE_EXTENSIONS)
    if include_heic:
        accepted |= HEIC_EXTENSIONS
    if include_avif:
        accepted |= AVIF_EXTENSIONS
    return frozenset(accepted)


# ── CorpusItem ────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class CorpusItem:
    """One image in a corpus, addressable across backends.

    ``key`` is the source-relative identifier (e.g. ``subdir/img.jpg``
    for local + S3 + B2). ``name`` is the leaf filename used in
    benchmark CSV output for human readability. Callers materialize
    the item to a local on-disk path via :meth:`local_path`, which
    is the only operation the inner scan loop needs.
    """

    name: str
    key: str
    _source: CorpusSource

    def local_path(self) -> Path:
        """Return an on-disk path that points at this item's bytes.

        For local sources this is the original path; for remote
        sources the item is downloaded into the per-source cache
        directory on first call and the cached path is returned on
        subsequent calls.
        """
        return self._source.materialize(self)


# ── CorpusSource Protocol ─────────────────────────────────────────────


@runtime_checkable
class CorpusSource(Protocol):
    """The backend-agnostic corpus interface used by the benchmark.

    Implementations: :class:`LocalCorpusSource`, :class:`S3CorpusSource`,
    :class:`B2CorpusSource`. Dispatch by URI happens in
    :func:`open_corpus`.
    """

    uri: str

    @property
    def kind(self) -> str:
        """Short backend identifier: ``"local"`` / ``"s3"`` / ``"b2"``.

        Used in the benchmark's methodology block and progress output
        so the operator knows which backend the corpus is being
        served from.
        """
        ...

    def list_items(
        self, accepted_exts: frozenset[str] | None = None,
    ) -> list[CorpusItem]:
        """Enumerate all matching images in the corpus, recursively.

        ``accepted_exts`` filters by file extension (case-insensitive).
        If ``None``, defaults to ``accepted_extensions()``. Results
        are sorted by key for deterministic ordering.
        """
        ...

    def materialize(self, item: CorpusItem) -> Path:
        """Return an on-disk path containing the item's bytes.

        For local sources this is the original path; for remote
        sources this triggers a (cached) download.
        """
        ...


# ── LocalCorpusSource ─────────────────────────────────────────────────


@dataclass
class LocalCorpusSource:
    """A local filesystem directory, walked recursively (S-061).

    The pre-S-061 ``discover_images`` walked only the top-level files
    via ``Path.iterdir()``. ``LocalCorpusSource`` recurses via
    ``Path.rglob`` instead, so corpora organized into subdirectories
    (per-day, per-camera, per-symbology, etc.) are fully enumerated
    without the user needing to flatten first.
    """

    uri: str  # original URI string (may include a scheme prefix or not)
    root: Path  # resolved absolute root directory

    @property
    def kind(self) -> str:
        """Backend identifier — used in benchmark methodology block."""
        return "local"

    def list_items(
        self, accepted_exts: frozenset[str] | None = None,
    ) -> list[CorpusItem]:
        exts = accepted_exts or accepted_extensions()
        items: list[CorpusItem] = []
        for p in self.root.rglob("*"):
            if not p.is_file():
                continue
            if p.suffix.lower() not in exts:
                continue
            try:
                rel = p.relative_to(self.root)
            except ValueError:  # symlink escapes root, etc. — skip defensively
                continue
            items.append(CorpusItem(
                name=p.name,
                # Always forward-slash regardless of OS, so keys are
                # portable + sort stably across Linux / macOS / Windows.
                # Windows ``str(rel)`` would yield ``"subdir\\foo.jpg"``
                # which both breaks the test and breaks cross-engine
                # CSV diffs when corpora migrate between OS hosts.
                key=rel.as_posix(),
                _source=self,
            ))
        items.sort(key=lambda c: c.key)
        return items

    def materialize(self, item: CorpusItem) -> Path:
        return self.root / item.key


# ── S3CorpusSource ────────────────────────────────────────────────────


@dataclass
class S3CorpusSource:
    """An AWS S3 bucket, accessed via boto3 (S-061).

    URI shape: ``s3://bucket-name`` or ``s3://bucket-name/optional/prefix``.

    Auth: boto3's standard credential chain. boto3 itself decides
    where credentials come from; the benchmark just calls
    ``boto3.client('s3')`` and lets the SDK do its lookup. Supports
    env vars, ``~/.aws/credentials``, IAM instance profile, ECS
    task role, etc.
    """

    uri: str
    bucket: str
    prefix: str  # may be empty
    cache_dir: Path = field(init=False)
    _client: object | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        # Per-bucket cache directory keeps separate corpora isolated.
        self.cache_dir = (
            Path.home() / ".cache" / "arbez-benchmark" / "s3" / self.bucket
        )

    @property
    def kind(self) -> str:
        return "s3"

    def _ensure_client(self) -> object:
        if self._client is None:
            try:
                import boto3  # type: ignore[import-not-found, import-untyped, unused-ignore]
            except ImportError as e:
                raise RuntimeError(
                    "boto3 is required for s3:// corpora. Install with "
                    "`pip install boto3` in the benchmark venv."
                ) from e
            self._client = boto3.client("s3")
        return self._client

    def list_items(
        self, accepted_exts: frozenset[str] | None = None,
    ) -> list[CorpusItem]:
        exts = accepted_exts or accepted_extensions()
        client = self._ensure_client()
        items: list[CorpusItem] = []
        # S3's list_objects_v2 is paginated; iterate cleanly.
        paginator = client.get_paginator("list_objects_v2")  # type: ignore[attr-defined]
        kwargs: dict[str, str] = {"Bucket": self.bucket}
        if self.prefix:
            kwargs["Prefix"] = self.prefix
        for page in paginator.paginate(**kwargs):
            for obj in page.get("Contents", []):
                key: str = obj["Key"]
                # Strip the prefix so `key` is corpus-relative,
                # matching the local backend's semantics.
                rel = key
                if self.prefix and rel.startswith(self.prefix):
                    rel = rel[len(self.prefix):].lstrip("/")
                if not rel:
                    continue
                if Path(rel).suffix.lower() not in exts:
                    continue
                items.append(CorpusItem(
                    name=Path(rel).name,
                    key=rel,
                    _source=self,
                ))
        items.sort(key=lambda c: c.key)
        return items

    def materialize(self, item: CorpusItem) -> Path:
        local = self.cache_dir / item.key
        if local.is_file() and local.stat().st_size > 0:
            return local
        local.parent.mkdir(parents=True, exist_ok=True)
        client = self._ensure_client()
        # Recompose the absolute S3 key (cache stores corpus-relative).
        s3_key = f"{self.prefix.rstrip('/')}/{item.key}" if self.prefix else item.key
        client.download_file(self.bucket, s3_key, str(local))  # type: ignore[attr-defined]
        return local


# ── B2CorpusSource ────────────────────────────────────────────────────


@dataclass
class B2CorpusSource:
    """A Backblaze B2 bucket, accessed via b2sdk (S-061).

    URI shape: ``b2://bucket-name`` or ``b2://bucket-name/optional/prefix``.

    Auth lookup order (b2sdk standard):

    1. ``B2_APPLICATION_KEY_ID`` + ``B2_APPLICATION_KEY`` env vars
       → authorize a fresh ``InMemoryAccountInfo`` session.
    2. Otherwise fall back to ``SqliteAccountInfo`` (the persistent
       file populated by ``b2 authorize-account`` CLI runs).

    This matches the conventions of the ``b2`` command-line tool —
    if you've already run ``b2 authorize-account``, this Just Works.
    """

    uri: str
    bucket_name: str
    prefix: str  # may be empty
    cache_dir: Path = field(init=False)
    _api: object | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.cache_dir = (
            Path.home() / ".cache" / "arbez-benchmark" / "b2" / self.bucket_name
        )

    @property
    def kind(self) -> str:
        return "b2"

    def _ensure_api(self) -> object:
        if self._api is None:
            try:
                # b2sdk's v3 is current; v2 still around for compatibility.
                # Prefer v3 if available, fall back to v2.
                try:
                    from b2sdk.v3 import (  # type: ignore[import-not-found, import-untyped, unused-ignore]
                        B2Api,
                        InMemoryAccountInfo,
                        SqliteAccountInfo,
                    )
                except ImportError:
                    from b2sdk.v2 import (  # type: ignore[import-not-found, import-untyped, unused-ignore]
                        B2Api,
                        InMemoryAccountInfo,
                        SqliteAccountInfo,
                    )
            except ImportError as e:
                raise RuntimeError(
                    "b2sdk is required for b2:// corpora. Install with "
                    "`pip install b2sdk` in the benchmark venv."
                ) from e

            key_id = os.environ.get("B2_APPLICATION_KEY_ID")
            app_key = os.environ.get("B2_APPLICATION_KEY")
            if key_id and app_key:
                # Env-var auth path: short-lived in-memory session.
                info = InMemoryAccountInfo()
                api = B2Api(info)
                api.authorize_account(
                    "production", key_id, app_key,
                )
                self._api = api
            else:
                # Persistent file path: rely on `b2 authorize-account`
                # having been run previously. SqliteAccountInfo reads
                # ~/.b2_account_info by default.
                info = SqliteAccountInfo()
                api = B2Api(info)
                if not info.is_master_key():
                    # Will raise if the file doesn't exist or is empty.
                    pass
                self._api = api
        return self._api

    def _get_bucket(self) -> object:
        api = self._ensure_api()
        return api.get_bucket_by_name(self.bucket_name)  # type: ignore[attr-defined]

    def list_items(
        self, accepted_exts: frozenset[str] | None = None,
    ) -> list[CorpusItem]:
        exts = accepted_exts or accepted_extensions()
        bucket = self._get_bucket()
        items: list[CorpusItem] = []
        # b2sdk's bucket.ls() yields (FileVersion, folder_name) tuples;
        # recursive=True walks subdirectories. The folder_name is None
        # for files (we ignore folders).
        # API has shifted slightly between b2sdk versions; this form
        # works across v2 + v3.
        try:
            ls_iter: Iterable[object] = bucket.ls(  # type: ignore[attr-defined]
                folder_to_list=self.prefix, recursive=True,
            )
        except TypeError:
            # Older b2sdk takes the prefix as a positional arg.
            ls_iter = bucket.ls(self.prefix, recursive=True)  # type: ignore[attr-defined]

        for entry in ls_iter:
            # entry is (FileVersion, folder_name) in v2/v3.
            if not isinstance(entry, tuple) or len(entry) < 1:
                continue
            file_version = entry[0]
            file_name: str = getattr(file_version, "file_name", "")
            if not file_name:
                continue
            # Strip the prefix so `key` is corpus-relative.
            rel = file_name
            if self.prefix and rel.startswith(self.prefix):
                rel = rel[len(self.prefix):].lstrip("/")
            if not rel:
                continue
            if Path(rel).suffix.lower() not in exts:
                continue
            items.append(CorpusItem(
                name=Path(rel).name,
                key=rel,
                _source=self,
            ))
        items.sort(key=lambda c: c.key)
        return items

    def materialize(self, item: CorpusItem) -> Path:
        local = self.cache_dir / item.key
        if local.is_file() and local.stat().st_size > 0:
            return local
        local.parent.mkdir(parents=True, exist_ok=True)
        bucket = self._get_bucket()
        b2_key = f"{self.prefix.rstrip('/')}/{item.key}" if self.prefix else item.key
        downloaded = bucket.download_file_by_name(b2_key)  # type: ignore[attr-defined]
        downloaded.save_to(str(local))
        return local


# ── URI dispatch ──────────────────────────────────────────────────────


def open_corpus(uri: str) -> CorpusSource:
    """Open a corpus by URI; dispatch to the appropriate backend.

    Examples::

        open_corpus("/Users/me/images")          # local
        open_corpus("./relative/path")           # local
        open_corpus("file:///abs/path")          # local (explicit scheme)
        open_corpus("s3://my-bucket/photos/")    # S3
        open_corpus("b2://my-bucket/2026/")      # B2

    Raises ``ValueError`` if the scheme isn't recognized.
    """
    # Normalize: strip whitespace + collapse trailing slashes (except
    # for bare-root cases that would become empty).
    uri = uri.strip()
    if not uri:
        raise ValueError("empty corpus URI")

    if uri.startswith("s3://"):
        bucket, _, prefix = uri.removeprefix("s3://").partition("/")
        if not bucket:
            raise ValueError(f"s3:// URI missing bucket name: {uri!r}")
        return S3CorpusSource(uri=uri, bucket=bucket, prefix=prefix)

    if uri.startswith("b2://"):
        bucket, _, prefix = uri.removeprefix("b2://").partition("/")
        if not bucket:
            raise ValueError(f"b2:// URI missing bucket name: {uri!r}")
        return B2CorpusSource(uri=uri, bucket_name=bucket, prefix=prefix)

    if uri.startswith("file://"):
        # Strip the scheme; leave a normal Path.
        path = uri.removeprefix("file://")
        root = Path(path).expanduser().resolve()
    else:
        # Bare local path (default).
        root = Path(uri).expanduser().resolve()

    if not root.exists():
        raise FileNotFoundError(f"corpus directory does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"corpus path is not a directory: {root}")

    return LocalCorpusSource(uri=uri, root=root)
