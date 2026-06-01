"""Decode-aware bench metrics for ``arbez_benchmark3.py`` (S-087).

Sibling helper. Computes the metrics that surface DECODED outputs
(not just raw detections) as the headline numbers for engine
comparison. Without these, an engine that emits 10 boxes per real
code looks 10x better by volume than an engine that emits 1 — even
though the second engine is doing equal or better real-world work.

Four families of metric live here, each scoped to a single function
for unit-testability:

1. :func:`per_engine_decode_metrics` — given one engine's records,
   augments its summary with ``n_decoded`` / ``n_unique_payloads`` /
   ``n_decoded_images`` / ``decode_rate``. Decode rate is bench3's
   long-standing measure but until S-087 it lived only in the
   summary.json; this also surfaces it in the headline tables.

2. :func:`effective_payload_recall` — for each engine, how much of
   the UNION of all engines' decoded ``(image, symbology, payload)``
   tuples did this engine also decode. Without ground truth, the
   union is the best proxy for "all codes that exist in the corpus
   and are decodable by at least one engine we tested". A high
   ``R_eff`` means the engine catches most of what's catchable; a
   low one means it leaves work on the table.

3. :func:`unique_engine_decodes` — per engine, count of
   ``(image, symbology, payload)`` tuples ONLY this engine decoded
   (count of 1 in the multi-set across all engines). This is the
   "what this engine uniquely contributes" metric — the argument
   for running consensus at all rather than picking a single
   engine and going.

4. :func:`beat_wechat_qr_scoreboard` — restricted to ``symbology=
   "qr"``: for each non-wechat engine, count of ``(image, payload)``
   tuples this engine decoded that wechat did not. Quantifies
   per-engine QR decode coverage relative to WeChat with corpus-
   level evidence.

Plus :func:`decoded_consensus_clusters` — the existing
``consensus_clusters`` filter to records with ``payload``, so the
"engines agreed on a thing" stat counts only agreements with an
actual decoded payload behind them.

All functions operate on plain :class:`DetRecord` lists; nothing
imports the benchmark module so the helpers are testable in
isolation with synthetic records.
"""
from __future__ import annotations

import re
from collections import Counter
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    # Only for type hints — the helpers operate on duck-typed records
    # so tests can pass simple dataclasses without importing bench3.
    from collections.abc import Iterable, Mapping


_QR_SYMBOLOGY = "qr"
_WECHAT_ENGINE = "wechat"


def per_engine_decode_metrics(
    records: list[Any],
) -> dict[str, Any]:
    """Augment a single engine's record list with decode-aware counts.

    Parameters
    ----------
    records:
        Iterable of records, each duck-typed to have ``payload``
        (``str | None``), ``image`` (``str``), and ``symbology``
        (``str``). bench3's :class:`DetRecord` fits; tests can pass
        a simple dataclass.

    Returns
    -------
    dict
        ``{"n_detected": int, "n_decoded": int, "n_unique_payloads":
        int, "n_decoded_images": int, "decode_rate": float}``.
        ``decode_rate`` is in [0, 1] (``n_decoded / n_detected``); 0
        if ``records`` is empty.
    """
    n_detected = len(records)
    decoded = [r for r in records if r.payload]
    n_decoded = len(decoded)
    n_unique_payloads = len(
        {(r.image, r.symbology, r.payload) for r in decoded}
    )
    n_decoded_images = len({r.image for r in decoded})
    decode_rate = (n_decoded / n_detected) if n_detected else 0.0
    return {
        "n_detected": n_detected,
        "n_decoded": n_decoded,
        "n_unique_payloads": n_unique_payloads,
        "n_decoded_images": n_decoded_images,
        "decode_rate": decode_rate,
    }


def _decoded_key_set(records: Iterable[Any]) -> set[tuple[str, str, str]]:
    """Set of ``(image, symbology, payload)`` for records with a
    non-empty payload. Used as the join key across engines."""
    return {
        (r.image, r.symbology, r.payload)
        for r in records
        if r.payload
    }


def effective_payload_recall(
    per_engine: Mapping[str, Any],
) -> dict[str, float]:
    """Per-engine: ``|engine_decodes ∩ union_decodes| / |union_decodes|``
    in [0, 1].

    The ``union_decodes`` set is the union across all engines of
    decoded ``(image, symbology, payload)`` tuples. The metric tells
    you what fraction of "all decodable codes any engine in this
    corpus run found" each individual engine recovered. It's a
    poor-man's recall when ground truth isn't available — biased
    upward (engines miss whatever every engine missed), but
    apples-to-apples within a run.

    Parameters
    ----------
    per_engine:
        Mapping from engine name to an object with a ``.records``
        attribute holding the engine's :class:`DetRecord`-like list.

    Returns
    -------
    dict
        ``{engine_name: recall_in_zero_to_one}``. All zeros when the
        run produced no decoded results across any engine.
    """
    engine_keys = {
        name: _decoded_key_set(result.records)
        for name, result in per_engine.items()
    }
    universe: set[tuple[str, str, str]] = set()
    for keys in engine_keys.values():
        universe |= keys
    if not universe:
        return {name: 0.0 for name in per_engine}
    return {
        name: len(keys & universe) / len(universe)
        for name, keys in engine_keys.items()
    }


def unique_engine_decodes(
    per_engine: Mapping[str, Any],
) -> dict[str, int]:
    """Per-engine: count of ``(image, symbology, payload)`` tuples
    that ONLY this engine decoded.

    Tuples decoded by 2+ engines are excluded — those represent
    redundant agreement, not unique contribution. The chart this
    metric drives is the justification for running consensus at
    all: if every engine's unique count is zero, you can pick the
    one with the best ``effective_payload_recall`` and skip
    consensus.

    Returns
    -------
    dict
        ``{engine_name: n_unique_decodes}``.
    """
    counts: Counter[tuple[str, str, str]] = Counter()
    engine_keys: dict[str, set[tuple[str, str, str]]] = {}
    for name, result in per_engine.items():
        keys = _decoded_key_set(result.records)
        engine_keys[name] = keys
        for k in keys:
            counts[k] += 1
    unique_keys = {k for k, c in counts.items() if c == 1}
    return {
        name: len(keys & unique_keys)
        for name, keys in engine_keys.items()
    }


def beat_wechat_qr_scoreboard(
    per_engine: Mapping[str, Any],
) -> dict[str, int]:
    """Per non-wechat engine: count of QR ``(image, payload)`` tuples
    this engine decoded that ``wechat`` did NOT decode.

    Symbology is restricted to ``"qr"`` because WeChat only decodes
    QR. For non-QR codes WeChat returns nothing, so the comparison
    would be a foregone conclusion + uninteresting.

    Returns
    -------
    dict
        ``{engine_name: n_qr_wins_over_wechat}``. Empty dict if
        ``wechat`` isn't in ``per_engine`` (run was invoked with
        ``--skip-wechat``).
    """
    if _WECHAT_ENGINE not in per_engine:
        return {}
    wechat_qr_keys = {
        (r.image, r.payload)
        for r in per_engine[_WECHAT_ENGINE].records
        if r.payload and r.symbology == _QR_SYMBOLOGY
    }
    out: dict[str, int] = {}
    for name, result in per_engine.items():
        if name == _WECHAT_ENGINE:
            continue
        engine_qr_keys = {
            (r.image, r.payload)
            for r in result.records
            if r.payload and r.symbology == _QR_SYMBOLOGY
        }
        out[name] = len(engine_qr_keys - wechat_qr_keys)
    return out


def decoded_consensus_clusters(
    all_records: list[Any],
    iou_threshold: float,
    cluster_fn: Any,
) -> Any:
    """Run an arbitrary clustering function over the decoded subset
    of records.

    Wired this way (taking ``cluster_fn`` as an argument) so the
    bench3-specific :func:`arbez_benchmark3.consensus_clusters`
    stays the single implementation of the IoU clustering algorithm;
    this helper just narrows the input to decoded records first.

    Parameters
    ----------
    all_records:
        Concatenation of every engine's records for the whole run.
    iou_threshold:
        Passed through to ``cluster_fn``.
    cluster_fn:
        Callable matching bench3's ``consensus_clusters`` signature
        ``(records, iou_threshold) -> dict[image, list[cluster]]``.
    """
    decoded = [r for r in all_records if r.payload]
    return cluster_fn(decoded, iou_threshold)


def greedy_decode_coverage_curve(
    per_engine: Mapping[str, Any],
) -> list[tuple[str, int, float, int]]:
    """S-088: at each step add the engine that maximises *marginal*
    decoded-payload coverage. Returns the greedy order + cumulative
    stats so a reader can answer "if I could run K engines, which K?"
    without manually enumerating ``C(n, k)`` subsets.

    Each item in the returned list is
    ``(engine_name, cumulative_decoded, cumulative_coverage_pct,
    marginal_decoded)``:

    * ``engine_name`` -- the engine selected at this step
    * ``cumulative_decoded`` -- size of the union of decoded
      ``(image, symbology, payload)`` keys across engines selected
      so far
    * ``cumulative_coverage_pct`` -- ``cumulative_decoded /
      |universe|`` in [0, 100] where universe = union across ALL
      engines (the same denominator
      :func:`effective_payload_recall` uses)
    * ``marginal_decoded`` -- how many new keys this engine adds
      vs the prior step (always positive on the first pick;
      monotonically non-increasing as the curve fills the union)

    Empty input or all-engines-zero-decodes returns ``[]``.

    Greedy is optimal here because the union-size function is
    submodular -- adding an engine to a smaller set never adds
    fewer keys than adding it to a larger one. So the greedy
    order matches the maximum-coverage order step-by-step.

    Parameters
    ----------
    per_engine:
        Mapping engine_name -> object with ``.records`` attribute.
        Same shape as :func:`effective_payload_recall`.
    """
    engine_keys = {
        name: _decoded_key_set(result.records)
        for name, result in per_engine.items()
    }
    universe: set[tuple[str, str, str]] = set()
    for keys in engine_keys.values():
        universe |= keys
    if not universe or not engine_keys:
        return []

    n_total = len(universe)
    selected: set[tuple[str, str, str]] = set()
    remaining = dict(engine_keys)
    out: list[tuple[str, int, float, int]] = []

    while remaining:
        # Pick the engine whose unselected-key contribution is
        # largest. Ties broken alphabetically by engine name for
        # determinism across runs.
        best_name: str | None = None
        best_marginal = -1
        for name in sorted(remaining):
            marginal = len(remaining[name] - selected)
            if marginal > best_marginal:
                best_marginal = marginal
                best_name = name
        assert best_name is not None  # remaining is non-empty
        selected |= remaining.pop(best_name)
        out.append((
            best_name,
            len(selected),
            100.0 * len(selected) / n_total,
            best_marginal,
        ))
    return out


def latency_recall_quadrants(
    per_engine: Mapping[str, Any],
) -> dict[str, tuple[float, float, str]]:
    """S-088: classify each engine by (mean latency, effective payload
    recall) into one of four labeled quadrants.

    The thresholds are the **median** of each axis across all engines
    in the run, so the chart self-calibrates -- a fast all-engine
    set still divides cleanly into fast/slow halves rather than
    everyone landing in "fast" because the absolute cutoff is too
    high.

    Returns
    -------
    dict
        ``{engine_name: (mean_ms, R_eff_pct, quadrant_label)}``
        where ``quadrant_label`` is one of
        ``"fast & accurate"``, ``"fast & lossy"``,
        ``"slow & accurate"``, ``"slow & lossy"``.

    "Fast" = mean latency below the median across engines in the
    run; "accurate" = R_eff above the median. The labels are intended
    for chart annotation; they're descriptive not prescriptive
    ("lossy" means low R_eff in this specific run + corpus, not a
    blanket judgment).
    """
    recall = effective_payload_recall(per_engine)
    latencies: dict[str, float] = {}
    for name, result in per_engine.items():
        walls = getattr(result, "wall_ms_per_image", None)
        if walls is None or len(walls) == 0:
            # Fall back to 0; the caller's chart will still place
            # this engine on the y axis even if its x is collapsed.
            latencies[name] = 0.0
        else:
            latencies[name] = sum(walls) / len(walls)
    if not latencies:
        return {}

    # Medians, with stable handling of even counts.
    lat_sorted = sorted(latencies.values())
    rec_sorted = sorted(recall.values())
    mid = len(lat_sorted) // 2
    if len(lat_sorted) % 2:
        lat_median = lat_sorted[mid]
        rec_median = rec_sorted[mid]
    else:
        lat_median = (lat_sorted[mid - 1] + lat_sorted[mid]) / 2
        rec_median = (rec_sorted[mid - 1] + rec_sorted[mid]) / 2

    out: dict[str, tuple[float, float, str]] = {}
    for name in per_engine:
        mean_ms = latencies[name]
        r_eff = 100.0 * recall.get(name, 0.0)
        is_fast = mean_ms <= lat_median
        is_accurate = r_eff >= 100.0 * rec_median
        if is_fast and is_accurate:
            q = "fast & accurate"
        elif is_fast and not is_accurate:
            q = "fast & lossy"
        elif not is_fast and is_accurate:
            q = "slow & accurate"
        else:
            q = "slow & lossy"
        out[name] = (mean_ms, r_eff, q)
    return out


# ── Payload normaliser (S-090 follow-up investigation) ────────────────
#
# zxing-cpp renders control characters as visible text tokens for
# human-readability (``<DC2>``, ``<U+87>``, ``<DEL>``, ``␝``, ``␠``).
# Apple Vision returns the raw bytes. Both are CORRECT decodes of
# the same payload bytes -- but pure string equality treats them as
# disagreements. The S-089 corpus run had 198 apparent Apple Vision
# disagreements; 195+ were this rendering difference + only 3 were
# genuine misreads. The normaliser below maps both styles to raw
# bytes so peer-validated correctness reflects actual decode
# correctness, not display-string differences.

_CONTROL_NAMES = (
    "NUL", "SOH", "STX", "ETX", "EOT", "ENQ", "ACK", "BEL",
    "BS",  "HT",  "LF",  "VT",  "FF",  "CR",  "SO",  "SI",
    "DLE", "DC1", "DC2", "DC3", "DC4", "NAK", "SYN", "ETB",
    "CAN", "EM",  "SUB", "ESC", "FS",  "GS",  "RS",  "US",
)
"""C0 control character mnemonics (0x00..0x1F)."""

_CARET_NAME_RE = re.compile(
    r"<(NUL|SOH|STX|ETX|EOT|ENQ|ACK|BEL|BS|HT|LF|VT|FF|CR|SO|SI|"
    r"DLE|DC1|DC2|DC3|DC4|NAK|SYN|ETB|CAN|EM|SUB|ESC|FS|GS|RS|US|DEL)>",
)
"""Pattern matching zxing's caret-notation rendering of C0 controls
plus DEL (0x7F)."""

_UPLUS_RE = re.compile(r"<U\+([0-9A-Fa-f]{2,6})>")
"""Pattern matching ``<U+XX>`` through ``<U+XXXXXX>`` Unicode
escape rendering used by zxing for code points outside C0 (e.g.
C1 controls 0x80-0x9F)."""

# Unicode "Symbols for Control Pictures" block (U+2400..U+241F).
# ``␝`` (U+241D) shown for raw 0x1D (GS / FNC1), ``␠`` (U+2420)
# for ASCII space when zxing wants to surface a space inside a
# structured payload, etc.
_SYMBOLS_FOR_CONTROL = {
    chr(0x2400 + i): chr(i) for i in range(0x20)
}
_SYMBOLS_FOR_CONTROL[chr(0x2420)] = " "  # SYMBOL FOR SPACE -> space
_SYMBOLS_FOR_CONTROL[chr(0x2421)] = chr(0x7F)  # SYMBOL FOR DELETE


def _resolve_caret(match: re.Match[str]) -> str:
    name = match.group(1)
    if name == "DEL":
        return chr(0x7F)
    return chr(_CONTROL_NAMES.index(name))


def _resolve_uplus(match: re.Match[str]) -> str:
    return chr(int(match.group(1), 16))


def normalize_payload(s: str) -> str:
    """Map equivalent renderings of the same byte sequence to a
    canonical raw-byte form.

    Designed to neutralise zxing-cpp's "human-readable" rendering of
    control characters (caret names like ``<DC2>``, ``<U+XX>``
    Unicode escapes, the Symbols-for-Control-Pictures block
    ``␝``/``␠``/etc.) so peer-validated correctness reflects
    actual decode correctness, not display-string differences.

    Idempotent: calling twice yields the same string. Safe on
    ``None``: returns ``None``.

    >>> normalize_payload("DEA<DC2><EM>")
    'DEA\\x12\\x19'
    >>> normalize_payload("DEA\\x12\\x19")
    'DEA\\x12\\x19'
    >>> normalize_payload("<U+87>")
    '\\x87'
    >>> normalize_payload("\\u241dGS1\\u241dpayload")  # ␝ = U+241D
    '\\x1dGS1\\x1dpayload'
    """
    if not s:
        return s
    s = _CARET_NAME_RE.sub(_resolve_caret, s)
    s = _UPLUS_RE.sub(_resolve_uplus, s)
    s = "".join(_SYMBOLS_FOR_CONTROL.get(c, c) for c in s)
    return s


def consensus_validated_recall(
    per_engine: Mapping[str, Any],
    clusters_by_image: Mapping[str, list[list[Any]]],
    min_votes: int = 2,
    normalise: bool = True,
) -> dict[str, dict[str, Any]]:
    """S-089 practical correctness: how often each engine produces the **consensus-verified** payload value.

    Premise. ``effective_payload_recall`` uses the UNION of all
    decodes as the gold standard, so any singleton decode (one
    engine alone) counts toward the universe. That over-rewards
    engines that fire many singleton payloads even if some are
    wrong -- the metric can't tell signal from noise without a
    second opinion.

    This metric defines a *consensus-verified* payload as one where
    **at least ``min_votes`` engines decoded the SAME value** for a
    given (image, bbox-cluster). Singleton decodes are explicitly
    excluded from the gold standard. Each engine is then scored
    against this peer-validated set:

    * ``verified_universe`` -- the count of (image, cluster_id,
      consensus_payload) tuples that ``>= min_votes`` engines
      agreed on.
    * ``correct`` -- of those, how many did this engine also
      decode with the matching payload value.
    * ``disagreed`` -- the engine decoded this cluster but with a
      DIFFERENT payload than consensus (a wrong reading).
    * ``missed`` -- the engine did not decode this cluster at all.
    * ``correctness_pct`` -- ``correct / verified_universe`` in
      [0, 100]. The "practical correctness" metric:
      this rewards being RIGHT on peer-validated codes, not
      catching lone singletons.
    * ``disagreement_pct`` -- ``disagreed / (correct + disagreed)``
      in [0, 100]. Of the verified codes this engine decoded,
      what fraction did it get WRONG. Surfaces engines that
      decode aggressively but with errors.

    Parameters
    ----------
    per_engine:
        Mapping engine_name -> object with a ``.records`` attribute.
    clusters_by_image:
        Output of bench3's ``consensus_clusters``: image ->
        list of clusters, each cluster a list of DetRecords sharing
        a bbox.
    min_votes:
        Minimum agreeing engines for a payload to count as
        consensus-verified. Default 2 (any peer corroboration is
        sufficient; raises the bar above lone-engine singletons).
    normalise:
        S-090: when True (default), apply :func:`normalize_payload`
        to each engine's decoded payload before comparison. Maps
        zxing-cpp's rendering of control characters (``<DC2>``,
        ``<U+87>``, ``␝``, ``␠``, ``<DEL>``) to the raw bytes Apple
        Vision and other engines return. Set to False to compare
        raw payload strings exactly (legacy behaviour pre-S-090).

    Returns
    -------
    dict
        ``{engine_name: {correct: int, disagreed: int, missed: int,
        verified_universe: int, correctness_pct: float,
        disagreement_pct: float}}``. Empty dict if no clusters
        reached the ``min_votes`` threshold (e.g. 1-engine bench).
    """
    if min_votes < 1:
        raise ValueError(f"min_votes must be >= 1, got {min_votes}")

    # S-090: optional payload normalisation neutralises zxing-style
    # control-char rendering (``<DC2>``, ``<U+87>``, ``␝``, ``␠``,
    # ``<DEL>``) so peer comparison reflects actual decode
    # correctness rather than display-string formatting.
    norm = normalize_payload if normalise else (lambda s: s)

    # Build the verified-payload set. Each entry is
    # (image, cluster_idx, consensus_payload) plus the set of
    # engines that voted for that payload.
    verified: list[tuple[str, int, str, set[str]]] = []
    for image, clusters in clusters_by_image.items():
        for cluster_idx, cluster in enumerate(clusters):
            # Per-payload engine vote tally within the cluster.
            votes: dict[str, set[str]] = {}
            for r in cluster:
                if not r.payload:
                    continue
                votes.setdefault(norm(r.payload), set()).add(r.engine)
            for payload, engines_voting in votes.items():
                if len(engines_voting) >= min_votes:
                    verified.append(
                        (image, cluster_idx, payload, engines_voting),
                    )

    out: dict[str, dict[str, Any]] = {}
    verified_universe = len(verified)
    for name in per_engine:
        correct = 0
        disagreed = 0
        for image, cluster_idx, payload, engines_voting in verified:
            if name in engines_voting:
                correct += 1
                continue
            # Did this engine decode this cluster with a different
            # payload? Look in the cluster.
            cluster = clusters_by_image.get(image, [])
            if cluster_idx >= len(cluster):
                continue
            this_engine_payloads = {
                norm(r.payload) for r in cluster[cluster_idx]
                if r.engine == name and r.payload
            }
            if this_engine_payloads and payload not in this_engine_payloads:
                disagreed += 1
        missed = verified_universe - correct - disagreed
        correctness_pct = (
            0.0 if verified_universe == 0
            else 100.0 * correct / verified_universe
        )
        disagreement_pct = (
            0.0 if (correct + disagreed) == 0
            else 100.0 * disagreed / (correct + disagreed)
        )
        out[name] = {
            "correct": correct,
            "disagreed": disagreed,
            "missed": missed,
            "verified_universe": verified_universe,
            "correctness_pct": correctness_pct,
            "disagreement_pct": disagreement_pct,
        }
    return out


def payload_agreement_distribution(
    clusters_by_image: Mapping[str, list[list[Any]]],
) -> dict[int, int]:
    """For each cluster, count how many engines agreed on the SAME
    decoded payload (not just the IoU-matched bounding box). Returns
    a histogram ``{n_engines_agreeing: cluster_count}``.

    Sharper signal than bbox-only consensus: 4 engines that disagree
    on the payload are 4 separate readings, not consensus. Captures
    cases where engines see "a code is here" but disagree on what
    the code SAYS — those should be flagged for review, not counted
    as confident detections.

    Clusters with zero decoded records are dropped (they contribute
    no payload to agree on).
    """
    hist: Counter[int] = Counter()
    for clusters in clusters_by_image.values():
        for cluster in clusters:
            payloads = [r.payload for r in cluster if r.payload]
            if not payloads:
                continue
            most_common_payload, _ = Counter(payloads).most_common(1)[0]
            engines_agreeing = {
                r.engine for r in cluster
                if r.payload == most_common_payload
            }
            hist[len(engines_agreeing)] += 1
    return dict(hist)
