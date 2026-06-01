"""Tests for ``examples/_corpus_source.py`` (S-061).

The corpus source abstraction lives in ``examples/`` because it's
benchmark-only (not part of the SDK's user-facing surface). It's
still worth a test suite — the URI dispatch + extension filtering
+ recursive walk are easy to regress otherwise.

Backend coverage in this file:

* :class:`LocalCorpusSource` — full coverage via tempdir fixtures.
* :class:`S3CorpusSource` — URI parsing only; live S3 tests are
  out of scope for unit tests (require AWS credentials + a bucket).
* :class:`B2CorpusSource` — same as S3; URI parsing only.

The S3/B2 backends' ``list_items`` and ``materialize`` methods
defer their imports of ``boto3`` and ``b2sdk`` to first call, so
tests that don't exercise those paths run without either dep
installed.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

# The module lives under ``examples/`` which isn't on the
# default Python path. Add it explicitly so the import resolves.
_EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"
sys.path.insert(0, str(_EXAMPLES_DIR))

from _corpus_source import (  # type: ignore[import-not-found]  # noqa: E402
    B2CorpusSource,
    CorpusItem,
    CorpusSource,
    LocalCorpusSource,
    S3CorpusSource,
    accepted_extensions,
    open_corpus,
)

# ── Extension filtering ───────────────────────────────────────────────


def test_accepted_extensions_defaults_include_all() -> None:
    """Default = core + HEIC + AVIF."""
    exts = accepted_extensions()
    assert ".jpg" in exts
    assert ".png" in exts
    assert ".heic" in exts
    assert ".avif" in exts


def test_accepted_extensions_can_disable_optional_formats() -> None:
    """Callers can opt out of HEIC / AVIF (e.g. if plugin isn't installed)."""
    exts = accepted_extensions(include_heic=False, include_avif=False)
    assert ".jpg" in exts
    assert ".heic" not in exts
    assert ".avif" not in exts


# ── LocalCorpusSource ─────────────────────────────────────────────────


def test_local_walks_recursively(tmp_path: Path) -> None:
    """Recursive walk picks up images in subdirectories."""
    (tmp_path / "subdir" / "nested").mkdir(parents=True)
    (tmp_path / "top.jpg").write_bytes(b"fake")
    (tmp_path / "subdir" / "mid.png").write_bytes(b"fake")
    (tmp_path / "subdir" / "nested" / "deep.webp").write_bytes(b"fake")
    # Non-image at top level — should be excluded.
    (tmp_path / "notes.txt").write_bytes(b"fake")

    src = LocalCorpusSource(uri=str(tmp_path), root=tmp_path)
    items = src.list_items()

    keys = [i.key for i in items]
    # String sort: 'm' < 'n' so "subdir/mid.png" precedes
    # "subdir/nested/...", and 's' < 't' so "subdir/..." precedes
    # "top.jpg".
    assert keys == [
        "subdir/mid.png",
        "subdir/nested/deep.webp",
        "top.jpg",
    ]
    # Names are leaf filenames, not the full key.
    names = [i.name for i in items]
    assert names == ["mid.png", "deep.webp", "top.jpg"]


def test_local_keys_use_forward_slash_on_all_platforms(tmp_path: Path) -> None:
    """Keys MUST use POSIX forward-slash separators regardless of OS.

    Otherwise Windows ``str(rel)`` yields ``"subdir\\foo.jpg"`` which
    (a) breaks the cross-OS portability the benchmark CSV diffs assume,
    (b) breaks alphabetical sort stability across hosts,
    (c) breaks the test above silently when run on Windows CI.

    Regression guard for the bug that surfaced on PR #32 CI: 5 Windows
    matrix cells failed on ``test_local_walks_recursively`` until the
    LocalCorpusSource started using ``rel.as_posix()`` instead of
    ``str(rel)``.
    """
    (tmp_path / "a" / "b").mkdir(parents=True)
    (tmp_path / "a" / "b" / "deep.jpg").write_bytes(b"fake")
    src = LocalCorpusSource(uri=str(tmp_path), root=tmp_path)
    items = src.list_items()
    assert len(items) == 1
    # Forward slash only — no backslash, no os.sep.
    assert items[0].key == "a/b/deep.jpg"
    assert "\\" not in items[0].key
    # materialize() round-trips the forward-slash key back to a real
    # on-disk path even when the OS native separator is backslash.
    assert items[0].local_path().read_bytes() == b"fake"


def test_local_filters_by_extension(tmp_path: Path) -> None:
    """Only files matching the accepted extension set are included."""
    (tmp_path / "good.jpg").write_bytes(b"fake")
    (tmp_path / "good.PNG").write_bytes(b"fake")  # case-insensitive
    (tmp_path / "bad.exe").write_bytes(b"fake")
    (tmp_path / "bad.html").write_bytes(b"fake")

    src = LocalCorpusSource(uri=str(tmp_path), root=tmp_path)
    keys = sorted(i.key for i in src.list_items())
    assert keys == ["good.PNG", "good.jpg"]


def test_local_materialize_returns_original_path(tmp_path: Path) -> None:
    """For local sources, ``local_path`` returns the actual on-disk path."""
    (tmp_path / "photo.jpg").write_bytes(b"hello")
    src = LocalCorpusSource(uri=str(tmp_path), root=tmp_path)
    items = src.list_items()
    assert len(items) == 1
    path = items[0].local_path()
    assert path == tmp_path / "photo.jpg"
    assert path.read_bytes() == b"hello"


def test_local_sorted_deterministic_ordering(tmp_path: Path) -> None:
    """``list_items()`` returns a sorted list — runs reproduce."""
    for name in ("z.jpg", "a.jpg", "m.jpg"):
        (tmp_path / name).write_bytes(b"fake")
    src = LocalCorpusSource(uri=str(tmp_path), root=tmp_path)
    keys = [i.key for i in src.list_items()]
    assert keys == ["a.jpg", "m.jpg", "z.jpg"]


def test_local_satisfies_corpus_source_protocol(tmp_path: Path) -> None:
    """LocalCorpusSource is a structural subtype of CorpusSource."""
    src = LocalCorpusSource(uri=str(tmp_path), root=tmp_path)
    assert isinstance(src, CorpusSource)


# ── open_corpus URI dispatch ──────────────────────────────────────────


def test_open_corpus_local_bare_path(tmp_path: Path) -> None:
    src = open_corpus(str(tmp_path))
    assert isinstance(src, LocalCorpusSource)
    assert src.root == tmp_path.resolve()
    assert src.kind == "local"


def test_open_corpus_local_file_scheme(tmp_path: Path) -> None:
    """``file://`` scheme dispatches to LocalCorpusSource."""
    src = open_corpus(f"file://{tmp_path}")
    assert isinstance(src, LocalCorpusSource)
    assert src.root == tmp_path.resolve()


def test_open_corpus_s3_uri_parses() -> None:
    """``s3://`` URI dispatches to S3CorpusSource (no network call)."""
    src = open_corpus("s3://my-bucket/photos/2026/")
    assert isinstance(src, S3CorpusSource)
    assert src.bucket == "my-bucket"
    assert src.prefix == "photos/2026/"
    assert src.kind == "s3"


def test_open_corpus_s3_uri_without_prefix() -> None:
    src = open_corpus("s3://my-bucket")
    assert isinstance(src, S3CorpusSource)
    assert src.bucket == "my-bucket"
    assert src.prefix == ""


def test_open_corpus_b2_uri_parses() -> None:
    """``b2://`` URI dispatches to B2CorpusSource (no network call)."""
    src = open_corpus("b2://archive-bucket/cameras/cam01/")
    assert isinstance(src, B2CorpusSource)
    assert src.bucket_name == "archive-bucket"
    assert src.prefix == "cameras/cam01/"
    assert src.kind == "b2"


def test_open_corpus_b2_uri_without_prefix() -> None:
    src = open_corpus("b2://archive-bucket")
    assert isinstance(src, B2CorpusSource)
    assert src.bucket_name == "archive-bucket"
    assert src.prefix == ""


def test_open_corpus_rejects_empty_uri() -> None:
    with pytest.raises(ValueError, match="empty"):
        open_corpus("")


def test_open_corpus_rejects_s3_without_bucket() -> None:
    with pytest.raises(ValueError, match="missing bucket"):
        open_corpus("s3://")


def test_open_corpus_rejects_b2_without_bucket() -> None:
    with pytest.raises(ValueError, match="missing bucket"):
        open_corpus("b2://")


def test_open_corpus_rejects_missing_local_dir() -> None:
    with pytest.raises(FileNotFoundError):
        open_corpus("/nonexistent/path/that/should/not/exist/xyz123")


def test_open_corpus_rejects_local_file_not_directory(tmp_path: Path) -> None:
    """Passing a regular file (not a directory) is a clear error."""
    f = tmp_path / "not_a_dir.txt"
    f.write_bytes(b"")
    with pytest.raises(NotADirectoryError):
        open_corpus(str(f))


# ── CorpusItem ─────────────────────────────────────────────────────────


def test_corpus_item_is_frozen() -> None:
    """CorpusItem is immutable — try to mutate it, expect FrozenInstanceError."""
    from dataclasses import FrozenInstanceError

    src = LocalCorpusSource(uri="/tmp", root=Path("/tmp"))
    item = CorpusItem(name="a.jpg", key="a.jpg", _source=src)
    with pytest.raises(FrozenInstanceError):
        item.name = "renamed.jpg"  # type: ignore[misc, unused-ignore]


# ── Lazy dep loading (the S3/B2 backends shouldn't import their SDKs
#    until first use) ─────────────────────────────────────────────────


def test_s3_source_construct_without_boto3() -> None:
    """Constructing S3CorpusSource doesn't import boto3 — only
    ``list_items``/``materialize`` do. This lets the benchmark module
    import cleanly in environments without boto3 installed."""
    src = open_corpus("s3://b/p/")
    assert isinstance(src, S3CorpusSource)
    assert src.bucket == "b"
    # _client stays None until first use.
    assert src._client is None


def test_b2_source_construct_without_b2sdk() -> None:
    """Same as above for B2 / b2sdk."""
    src = open_corpus("b2://b/p/")
    assert isinstance(src, B2CorpusSource)
    assert src.bucket_name == "b"
    assert src._api is None
