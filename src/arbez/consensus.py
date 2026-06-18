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
* **symbology** — most common Symbology in the group; ties go to the
  symbology of the highest-scored member among the top-count
  candidates (stable secondary tiebreak by symbology value).
* **payload** — most common non-None payload; ties go to the
  highest-scored detection's payload. ``None`` if no engine decoded
  the crop.
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
  min_votes, iou_threshold)``.
* Output ``engine`` field is always ``"consensus"`` (not the source
  engine's name).
* ``extras`` keys ``voted_by``, ``vote_count``, ``agreed_payloads``
  are locked semantics; new keys may be added.
"""

from __future__ import annotations

import logging
import statistics
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import TYPE_CHECKING

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


def run_consensus_detailed(
    pil_image: PILImage,
    engines: dict[str, Engine],
    *,
    min_votes: int = 2,
    iou_threshold: float = 0.5,
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
        raise ValueError(
            f"run_consensus: iou_threshold must be in [0, 1]; got {iou_threshold}"
        )

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
                only_name, e,
            )
            per_engine[only_name] = ()
    else:
        # Multi-engine: parallel dispatch via ThreadPoolExecutor.
        # One thread per engine (S-018 dispatch model). Named futures
        # so engine failures don't kill the whole vote — we just
        # record that engine produced no detections.
        with ThreadPoolExecutor(max_workers=len(engines)) as ex:
            future_to_name = {
                ex.submit(eng.detect_and_decode, pil_image): name
                for name, eng in engines.items()
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
                        name, e,
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
            accepted.append(_aggregate_group(group))

    # Engine Protocol contract: sort by descending score.
    accepted.sort(key=lambda d: d.score, reverse=True)
    return ConsensusResult(detections=tuple(accepted), per_engine=breakdown)


def run_consensus(
    pil_image: PILImage,
    engines: dict[str, Engine],
    *,
    min_votes: int = 2,
    iou_threshold: float = 0.5,
) -> tuple[Detection, ...]:
    """Vote across multiple engines on one image (merged result only).

    Thin wrapper over :func:`run_consensus_detailed` that returns just the
    merged consensus ``detections`` tuple — the historical S-032 signature.
    Use :func:`run_consensus_detailed` when you also need the per-engine
    breakdown. See that function for the full voting-policy docs.
    """
    return run_consensus_detailed(
        pil_image, engines, min_votes=min_votes, iou_threshold=iou_threshold
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


def _aggregate_group(group: list[tuple[str, Detection]]) -> Detection:
    """Merge a group of overlapping detections (from different engines) into a single consensus
    Detection.

    Field-by-field rules (full deterministic spec in
    ``docs/consensus-rules.md``):

    * **bbox**: per-coord median across cluster members (robust to
      one engine reporting a slightly off bbox).
    * **score**: arithmetic mean of cluster members' scores.
    * **symbology**: most-common-value voting; tiebreak to the
      symbology of the highest-scored member among top-count
      candidates, with a stable secondary tiebreak by symbology
      value (same S-077 determinism pattern as payload).
    * **payload**: most-common-value voting over non-None payloads;
      tiebreak to the highest-scored member's payload IF non-None,
      otherwise to the highest-scored member whose payload IS
      non-None (S-077 fix: this deterministic tiebreak replaced
      a Counter-first-encountered fallback that inherited Stage 1
      ``as_completed`` non-determinism when ``best_det.payload``
      happened to be ``None``).
    * **polygon**: highest-scored member's polygon (medians don't
      generalize cleanly to rotated quads).
    * **engine**: hardcoded ``"consensus"``.
    * **extras**: ``voted_by`` (sorted tuple of unique engine names),
      ``vote_count`` (= len(voted_by)), ``agreed_payloads`` (sorted
      tuple of distinct non-None payloads), ``source_count`` (total
      members in cluster; may exceed ``vote_count`` when one engine
      contributed multiple detections to the same IoU cluster).
    """
    # bbox: per-corner median (robust to one engine's bbox being off)
    medians = [
        statistics.median(d.bbox_xyxy[i] for _, d in group)
        for i in range(4)
    ]
    bbox = (float(medians[0]), float(medians[1]),
            float(medians[2]), float(medians[3]))

    # Highest-scored member — drives tiebreaks for symbology/payload
    # and provides the polygon (median polygons doesn't generalize
    # cleanly to rotated quads).
    _best_name, best_det = max(group, key=lambda nd: nd[1].score)

    # symbology: most common; tie -> best_det's symbology
    #
    # Code-review fix (2026-06): same S-077 determinism pattern as the
    # payload tiebreak below. Pre-fix, when ``best_det``'s symbology
    # was NOT at the top count, we fell back to
    # ``Counter.most_common(1)[0]``'s first-encountered entry — which
    # inherits Stage 1 ``as_completed`` ordering through the
    # score-sorted tagged list (stable sort preserves insertion order
    # for tied scores), so the output could vary run to run. Fix:
    # among members whose symbology IS at the top count, pick the
    # highest-scored member's symbology, breaking exact score ties by
    # symbology value for a stable total order. Identical output in
    # the common no-tie case (a unique top-count symbology wins either
    # way). Rule documented in docs/consensus-rules.md.
    sym_counts: Counter[Symbology] = Counter(d.symbology for _, d in group)
    _sym_top, sym_top_n = sym_counts.most_common(1)[0]
    # No ``best_det`` fast path: ``best_det`` is the FIRST maximal-score
    # member, which would re-inherit ``as_completed`` ordering whenever
    # two members tie at the top score with different top-count
    # symbologies. The single expression below is deterministic for
    # every input (score, then symbology value) and identical to the
    # fast path in the common no-tie case.
    symbology = max(
        (d for _, d in group if sym_counts[d.symbology] == sym_top_n),
        key=lambda d: (d.score, d.symbology.value),
    ).symbology

    # payload: most common non-None; tie -> highest-scored DECODER's payload
    #
    # Code-review fix (2026-05-17): pre-fix, the tiebreak preferred
    # ``best_det.payload`` only if ``best_det.payload is not None``.
    # If the highest-scored detection had ``payload=None`` (detector
    # found the box but couldn't decode the crop), the tiebreak fell
    # through to ``Counter.most_common(1)[0]``'s first-encountered,
    # which is order-dependent on the post-sort tagged list — which
    # inherits from ``as_completed`` order (Stage 1 non-determinism).
    # Concrete bug: arbez fires score=0.9 with payload=None; zxing
    # fires score=0.7 with payload="X"; second zxing-like engine
    # fires score=0.65 with payload="Y". No count tie among payloads
    # (X: 1, Y: 1), but Counter ties at top → result depends on
    # whichever non-None payload was inserted first → non-det.
    #
    # Fix: when best_det has no payload, fall back to the highest-
    # scored member whose payload IS non-None (call it
    # ``decoder_best``). That detection wins the tiebreak. Strictly
    # deterministic given score values; only N1 (tied scores) can
    # still produce non-determinism, which is documented in
    # docs/consensus-rules.md.
    payloads = [d.payload for _, d in group if d.payload is not None]
    payload: str | None
    if payloads:
        payload_counts: Counter[str] = Counter(payloads)
        top_payload, top_payload_n = payload_counts.most_common(1)[0]
        if (
            best_det.payload is not None
            and payload_counts[best_det.payload] == top_payload_n
        ):
            payload = best_det.payload
        else:
            # best_det's payload is None OR not at the top count.
            # Find the highest-scored member with a non-None payload
            # whose payload IS at the top count; if there is one,
            # prefer it. This makes the tiebreak deterministic in the
            # face of payload-count ties.
            decoder_candidates = [
                d for _, d in group
                if d.payload is not None
                and payload_counts[d.payload] == top_payload_n
            ]
            if decoder_candidates:
                decoder_best = max(decoder_candidates, key=lambda d: d.score)
                payload = decoder_best.payload
            else:
                payload = top_payload
    else:
        payload = None

    score = float(statistics.mean(d.score for _, d in group))

    polygon = best_det.polygon

    engines_voted: tuple[str, ...] = tuple(sorted({name for name, _ in group}))
    agreed_payloads: tuple[str, ...] = tuple(sorted(set(payloads)))
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
