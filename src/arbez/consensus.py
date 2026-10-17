"""Multi-engine consensus voting (S-032, locked from v0.0.18; S-093, 0.2.0).

Public function: :func:`run_consensus`. Used by multi-engine
:class:`~arbez.scanner.Scanner` paths — bare ``Scanner()`` (union,
``consensus=1``) and ``Scanner(consensus=N)`` (require >= N engines per
code). Engines run in parallel (one thread per engine per S-018); their
detections are grouped by IoU and merged into a single
:class:`Detection` per physical barcode.

Voting policy (S-032)
---------------------
For each detection group (overlapping bboxes from different engines):

* **min_votes** — keep the group iff at least N unique engines
  contributed. ``Scanner`` maps its integer ``consensus`` threshold onto
  this parameter (default bare ``Scanner()`` → ``min_votes=1`` union;
  ``Scanner(consensus=2)`` → ``min_votes=2``). Set to ``len(engines)``
  for "all-agree" mode.
* **bbox** — per-coord median across group members. Robust against
  one engine reporting a slightly off bbox.
* **symbology** — the winning decoder's symbology when a payload is
  present; otherwise the most common symbology (score, then value).
* **payload** — most common canonical payload among decoded members.
  Ties break by score, then symbology value, then the canonical
  string. Not by group order. ``None`` if no engine decoded.
* **score** — mean of group members' scores.
* **engine** — fixed string ``"consensus"`` on the output Detection.
* **extras** carries ``voted_by`` (tuple of engine names that
  contributed), ``vote_count`` (int), and ``agreed_payloads``
  (tuple of distinct non-None payloads seen in the group).

Thread-safety
-------------
Each engine instance must be S-012 thread-safe for the concurrent-
parallel dispatch. The four built-in engines all satisfy this when
shared across threads.

Stability contract (S-032, locked from v0.0.18)
-----------------------------------------------
* ``run_consensus`` signature locked: ``(pil_image, engines, *,
  min_votes, iou_threshold)``. S-099 adds an optional keyword-only
  ``symbologies`` allow-list (default ``None``, previous behaviour).
  The filter runs after the vote and uses the decoding engine's
  symbology, not the voted cluster label. A kept detection is
  labelled with that same symbology when a payload is present.
  ``per_engine`` stays raw. ``agreed_payloads`` drops strings whose
  member symbology is outside the allow-list.
* Output ``engine`` field is always ``"consensus"`` (not the source
  engine's name).
* ``extras`` keys ``voted_by``, ``vote_count``, ``agreed_payloads``
  are locked semantics; new keys may be added.
"""

from __future__ import annotations

import logging
import statistics
from collections import Counter
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import TYPE_CHECKING

from arbez._canon import canonicalize_payload
from arbez.types import Detection, Symbology

if TYPE_CHECKING:
    from collections.abc import Mapping

    from PIL.Image import Image as PILImage

    from arbez.engines.base import Engine

_log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ConsensusResult:
    """Return of :func:`run_consensus_detailed` (S-093).

    * ``detections`` — the merged, per-code consensus result (each
      ``Detection`` tagged ``engine="consensus"`` with ``extras["voted_by"]``).
    * ``per_engine`` — ``{engine_name: that engine's own raw detections}``,
      i.e. what each voter saw BEFORE clustering/voting. Lets a caller see
      detections that didn't reach the vote threshold.
    """

    detections: tuple[Detection, ...]
    per_engine: Mapping[str, tuple[Detection, ...]]


def _coerce_symbology_allowlist(
    symbologies: Iterable[Symbology | str] | None,
) -> frozenset[Symbology] | None:
    """Normalize ``symbologies=`` to a frozenset, or ``None`` for "all".

    A bare string or a single :class:`Symbology` is rejected: both are
    iterable, and a string would otherwise allow one character at a time.
    ``Symbology`` is a ``str`` enum, so members are checked before strings.
    """
    if symbologies is None:
        return None
    if isinstance(symbologies, (str, Symbology)):
        raise TypeError(
            "symbologies must be an iterable of Symbology members or "
            "value strings (for example {Symbology.QR} or {'qr'}), not "
            f"a single {type(symbologies).__name__}."
        )
    allowed: set[Symbology] = set()
    for item in symbologies:
        if isinstance(item, Symbology):
            allowed.add(item)
            continue
        if isinstance(item, str):
            try:
                allowed.add(Symbology(item))
            except ValueError as e:
                known = tuple(s.value for s in Symbology)
                raise ValueError(
                    f"unknown symbology {item!r}; expected a Symbology member or one of {known}."
                ) from e
            continue
        raise TypeError(
            "symbologies entries must be Symbology members or value "
            f"strings; got {type(item).__name__}."
        )
    if not allowed:
        raise ValueError("symbologies allow-list is empty; pass None to allow every symbology.")
    return frozenset(allowed)


def run_consensus_detailed(
    pil_image: PILImage,
    engines: dict[str, Engine],
    *,
    min_votes: int = 2,
    iou_threshold: float = 0.5,
    symbologies: Iterable[Symbology | str] | None = None,
) -> ConsensusResult:
    """Vote across multiple engines on one image; return merged + per-engine.

    Same voting logic as :func:`run_consensus`, but returns a
    :class:`ConsensusResult` carrying both the merged consensus
    ``detections`` and the ``per_engine`` breakdown (S-093). ``Scanner``
    uses this so ``Result.per_engine`` can expose each voter's raw finds.

    Args:
        pil_image: RGB PIL image (already coerced by Scanner).
        engines: ``{engine_name: Engine instance}``. Each engine runs
            in its own thread (multi-engine path) or synchronously
            when ``len(engines) == 1`` (S-077 short-circuit). The
            names are used to build ``voted_by`` in the output
            Detection's ``extras``.
        min_votes: minimum number of unique engines that must agree on
            an overlapping bbox cluster for it to survive the vote.
        iou_threshold: bbox IoU >= this groups two detections as the
            same physical barcode.
        symbologies: optional allow-list applied after the vote, on
            the symbology of the engine that supplied the winning
            payload (S-099). ``None`` (default) keeps every cluster
            that passes ``min_votes``. The merged detection uses that
            same symbology when a payload is present. ``per_engine``
            is not filtered. ``agreed_payloads`` omits raw strings
            from symbologies outside the list. A bare string is a
            ``TypeError``; an empty iterable is a ``ValueError``.

    Returns:
        A :class:`ConsensusResult`: ``.detections`` is the merged
        consensus tuple (sorted by descending score, each tagged
        ``engine="consensus"``; empty if no cluster reaches ``min_votes``),
        and ``.per_engine`` maps each voter to its own raw detections.

    Per-engine exception handling:
        If an individual engine's ``detect_and_decode`` raises any
        Exception, the engine is **treated as having returned zero
        detections** and the failure is logged at WARNING via the
        ``arbez.consensus`` logger. The other engines proceed
        unchanged. This is deliberate ("best-effort consensus")
        but means a silently-failing engine won't surface in the
        vote — inspect logger output to diagnose voter dropouts.
        Applies to both the multi-engine parallel path AND the
        single-engine fast path.

    Raises:
        ValueError: empty ``engines`` dict, or ``min_votes < 1``, or
            ``min_votes > len(engines)``, or ``iou_threshold`` not in
            ``[0, 1]``.
    """
    if not engines:
        raise ValueError("run_consensus: engines dict is empty")
    if min_votes < 1:
        raise ValueError(f"run_consensus: min_votes must be >= 1; got {min_votes}")
    # Mirror of Scanner.__init__'s construction-time validation — direct
    # ``run_consensus`` callers get the same fail-fast instead of an
    # always-empty vote.
    if min_votes > len(engines):
        raise ValueError(
            f"run_consensus: min_votes={min_votes} exceeds the number of "
            f"voting engines ({len(engines)}: {tuple(engines)}). No cluster "
            f"can ever reach this threshold, so the vote would silently "
            f"return empty."
        )
    if not (0.0 <= iou_threshold <= 1.0):
        raise ValueError(f"run_consensus: iou_threshold must be in [0, 1]; got {iou_threshold}")
    allow = _coerce_symbology_allowlist(symbologies)

    # Stage 1: per-engine scan.
    #
    # Code-review fix (2026-05-17): a 1-engine consensus is a legal
    # call shape (``Scanner(consensus=2, engines=("arbez",))`` with a
    # single installed engine still runs through the merge path)
    # plus edge cases where only one engine is available.
    # Pre-fix, we always spun up a ``ThreadPoolExecutor`` even for
    # one engine, paying the pool-creation + thread-spawn + executor-
    # shutdown cost on every scan. Short-circuit synchronously when
    # ``len(engines) == 1``: ~50 ms savings per scan on Apple Silicon
    # (measured), correctness identical because there's no parallel
    # dispatch to coordinate.
    #
    # The output Detections still get retagged as ``engine="consensus"``
    # downstream — preserving the consensus-shape contract so callers
    # branching on ``det.engine == "consensus"`` don't break when the
    # voter set degrades to 1.
    per_engine: dict[str, tuple[Detection, ...]] = {}
    if len(engines) == 1:
        only_name, only_eng = next(iter(engines.items()))
        try:
            per_engine[only_name] = only_eng.detect_and_decode(pil_image)
        except Exception as e:
            _log.warning(
                "consensus: engine %s raised %r; treating as empty",
                only_name,
                e,
            )
            per_engine[only_name] = ()
    else:
        # Multi-engine: parallel dispatch via ThreadPoolExecutor.
        # One thread per engine (S-018 dispatch model). Named futures
        # so engine failures don't kill the whole vote — we just
        # record that engine produced no detections.
        with ThreadPoolExecutor(max_workers=len(engines)) as ex:
            future_to_name = {
                ex.submit(eng.detect_and_decode, pil_image): name for name, eng in engines.items()
            }
            for fut in as_completed(future_to_name):
                name = future_to_name[fut]
                try:
                    per_engine[name] = fut.result()
                except Exception as e:
                    # S-039: ``name`` is a stable plain-string identifier
                    # like ``"zxing"``; ``%s`` reads more cleanly than the
                    # ``%r`` that printed extra quotes.
                    _log.warning(
                        "consensus: engine %s raised %r; treating as empty",
                        name,
                        e,
                    )
                    per_engine[name] = ()

    # Freeze the per-engine breakdown (each engine's own raw detections,
    # before clustering/voting) for the ConsensusResult.
    breakdown: dict[str, tuple[Detection, ...]] = {
        name: tuple(dets) for name, dets in per_engine.items()
    }

    # Stage 2: tag each detection with its source engine + flatten.
    tagged: list[tuple[str, Detection]] = [
        (name, d) for name, dets in per_engine.items() for d in dets
    ]
    if not tagged:
        return ConsensusResult(detections=(), per_engine=breakdown)
    # Sort by descending score so the highest-confidence detection
    # seeds each cluster.
    tagged.sort(key=lambda nd: nd[1].score, reverse=True)

    # Stage 3: greedy IoU-based clustering. Same shape as NMS but we
    # KEEP overlapping detections (grouping them) instead of dropping.
    groups: list[list[tuple[str, Detection]]] = []
    used = [False] * len(tagged)
    for i in range(len(tagged)):
        if used[i]:
            continue
        seed_name, seed_det = tagged[i]
        group: list[tuple[str, Detection]] = [(seed_name, seed_det)]
        used[i] = True
        for j in range(i + 1, len(tagged)):
            if used[j]:
                continue
            other_name, other_det = tagged[j]
            if _iou(seed_det.bbox_xyxy, other_det.bbox_xyxy) >= iou_threshold:
                group.append((other_name, other_det))
                used[j] = True
        groups.append(group)

    # Stage 4: vote. Keep groups with >= min_votes UNIQUE engines.
    accepted: list[Detection] = []
    for group in groups:
        unique_engines = {name for name, _ in group}
        if len(unique_engines) >= min_votes:
            # S-099: the voted label can be an undecoded detector box
            # while the payload came from another symbology. Filter on
            # the decoder. per_engine above stays raw; agreed_payloads
            # inside the aggregate drops disallowed symbologies.
            if allow is not None and _decoding_symbology(group) not in allow:
                continue
            accepted.append(_aggregate_group(group, allow=allow))

    # Engine Protocol contract: sort by descending score.
    accepted.sort(key=lambda d: d.score, reverse=True)
    return ConsensusResult(detections=tuple(accepted), per_engine=breakdown)


def run_consensus(
    pil_image: PILImage,
    engines: dict[str, Engine],
    *,
    min_votes: int = 2,
    iou_threshold: float = 0.5,
    symbologies: Iterable[Symbology | str] | None = None,
) -> tuple[Detection, ...]:
    """Vote across multiple engines on one image (merged result only).

    Thin wrapper over :func:`run_consensus_detailed` that returns just the
    merged consensus ``detections`` tuple — the historical S-032 signature.
    Use :func:`run_consensus_detailed` when you also need the per-engine
    breakdown. See that function for the full voting-policy docs.
    """
    return run_consensus_detailed(
        pil_image,
        engines,
        min_votes=min_votes,
        iou_threshold=iou_threshold,
        symbologies=symbologies,
    ).detections


# ── Internal helpers ──────────────────────────────────────────────────────


def _iou(
    bbox_a: tuple[float, float, float, float],
    bbox_b: tuple[float, float, float, float],
) -> float:
    """Intersection-over-union of two (x1, y1, x2, y2) bboxes.

    Returns 0.0 for degenerate boxes (zero or negative area) — robust against engines occasionally
    emitting bad geometry.
    """
    ax1, ay1, ax2, ay2 = bbox_a
    bx1, by1, bx2, by2 = bbox_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    if inter == 0.0:
        return 0.0

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _voted_symbology(group: list[tuple[str, Detection]]) -> Symbology:
    """Most-common symbology, with a score-then-value tiebreak.

    Among members whose symbology is at the top count, pick the
    highest score, then the symbology value. A ``best_det`` fast path
    would re-inherit ``as_completed`` order when two members tie at
    the top score with different top-count symbologies (S-077).
    """
    sym_counts: Counter[Symbology] = Counter(d.symbology for _, d in group)
    _sym_top, sym_top_n = sym_counts.most_common(1)[0]
    return max(
        (d for _, d in group if sym_counts[d.symbology] == sym_top_n),
        key=lambda d: (d.score, d.symbology.value),
    ).symbology


def _winning_decoder(group: list[tuple[str, Detection]]) -> Detection | None:
    """Detection that supplied the winning payload, or None if none decoded.

    Most common canonical payload, then highest score, then symbology
    value, then the canonical string. Score ties must not follow
    ``as_completed`` order: ZXing and WeChat both report 1.0, and a
    QR-only allow-list would otherwise keep or drop the cluster
    depending on which thread finished first (S-099).
    """
    decoded: list[tuple[Detection, str]] = []
    for _, det in group:
        if det.payload is None:
            continue
        canon = canonicalize_payload(det.payload, det.symbology)
        if canon is None:
            continue
        decoded.append((det, canon))
    if not decoded:
        return None
    counts: Counter[str] = Counter(canon for _, canon in decoded)
    top_n = max(counts.values())
    candidates = [det for det, canon in decoded if counts[canon] == top_n]
    return max(
        candidates,
        key=lambda det: (
            det.score,
            det.symbology.value,
            canonicalize_payload(det.payload, det.symbology) or "",
        ),
    )


def _decoding_symbology(group: list[tuple[str, Detection]]) -> Symbology:
    """Allow-list symbology: the winning decoder, else the voted label.

    An undecoded detector box can outvote the engine that actually
    decoded the code (S-099). Filtering on the voted label keeps that
    foreign payload.
    """
    winner = _winning_decoder(group)
    if winner is not None:
        return winner.symbology
    return _voted_symbology(group)


def _aggregate_group(
    group: list[tuple[str, Detection]],
    *,
    allow: frozenset[Symbology] | None = None,
) -> Detection:
    """Merge a group of overlapping detections (from different engines) into a single consensus
    Detection.

    Field-by-field rules (full deterministic spec in
    ``docs/consensus-rules.md``):

    * **bbox**: per-coord median across cluster members (robust to
      one engine reporting a slightly off bbox).
    * **score**: arithmetic mean of cluster members' scores.
    * **symbology**: the winning decoder's symbology when a payload
      is present, so a Data Matrix payload is not published as QR.
      Undecoded clusters keep the voted label (S-099).
    * **payload**: most common canonical payload; tiebreak by score,
      then symbology value, then canonical string (S-077 / S-099).
      Equal scores do not follow ``as_completed`` order.
    * **polygon**: highest-scored member's polygon (medians don't
      generalize cleanly to rotated quads).
    * **engine**: hardcoded ``"consensus"``.
    * **extras**: ``voted_by`` (sorted tuple of unique engine names),
      ``vote_count`` (= len(voted_by)), ``agreed_payloads`` (sorted
      tuple with one representative raw payload per canonical form
      (issue #12) -- deduped by canonical vote key, so two engines that
      reported the same physical code in different framing collapse to
      one entry, but each entry stays a raw engine string),
      ``source_count`` (total members in cluster; may exceed
      ``vote_count`` when one engine contributed multiple detections to
      the same IoU cluster).
    """
    # bbox: per-corner median (robust to one engine's bbox being off)
    medians = [statistics.median(d.bbox_xyxy[i] for _, d in group) for i in range(4)]
    bbox = (float(medians[0]), float(medians[1]), float(medians[2]), float(medians[3]))

    # Polygon stays with the first highest-scored member. Payload and
    # the published symbology use the winning decoder, which tie-breaks
    # equal scores by symbology value (S-099).
    _best_name, best_det = max(group, key=lambda nd: nd[1].score)

    symbology = _decoding_symbology(group)
    winner = _winning_decoder(group)
    payload = winner.payload if winner is not None else None

    # issue #12: agreed_payloads dedups on the canonical form.
    # Members outside the allow-list are omitted so a kept QR
    # detection does not publish a Data Matrix string (S-099).
    # per_engine stays raw; this extras field does not.
    raw_and_canon: list[tuple[str, str]] = []
    for _, det in group:
        if det.payload is None:
            continue
        if allow is not None and det.symbology not in allow:
            continue
        canon = canonicalize_payload(det.payload, det.symbology)
        if canon is None:
            continue
        raw_and_canon.append((det.payload, canon))

    score = float(statistics.mean(d.score for _, d in group))

    polygon = best_det.polygon

    engines_voted: tuple[str, ...] = tuple(sorted({name for name, _ in group}))
    # issue #12: dedup by canonical form, keep one RAW representative per
    # canonical key (highest-scored detection first, since `group` is
    # score-sorted descending from Stage 2). Fewer entries than before when
    # engines' raw forms differed but the physical code was the same.
    canon_to_raw: dict[str, str] = {}
    for raw, c in raw_and_canon:
        if c not in canon_to_raw:
            canon_to_raw[c] = raw
    agreed_payloads: tuple[str, ...] = tuple(sorted(canon_to_raw.values()))
    extras: dict[str, object] = {
        "voted_by": engines_voted,
        "vote_count": len(engines_voted),
        "agreed_payloads": agreed_payloads,
        "source_count": len(group),  # may exceed vote_count if same engine emitted multiple
    }

    return Detection(
        bbox_xyxy=bbox,
        symbology=symbology,
        score=score,
        payload=payload,
        engine="consensus",
        polygon=polygon,
        extras=extras,
    )
