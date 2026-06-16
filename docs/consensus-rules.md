# Consensus rules — what the multi-engine `Scanner()` actually does

This page is the deterministic field-by-field spec for the consensus
merge on the multi-engine `Scanner()` path. The implementation lives in
`src/arbez/consensus.py`; this is the user-facing companion that
explains the rules without reading the code.

Every multi-engine path uses the same machinery; only the engine set
and the agreement threshold differ (`Scanner`'s integer `consensus=`
maps onto the `min_votes` below):

| Path | Engines | threshold |
|---|---|---|
| Bare `Scanner()` | all installed | 1 (union) |
| `Scanner(consensus=N)` | all installed | N |
| `Scanner(consensus=N, engines=...)` | as passed | N |

## The pipeline (5 stages)

```
INPUTS:
  image           : a PIL RGB image (already coerced by Scanner)
  engines         : dict[engine_name -> Engine instance]
  min_votes       : int >= 1
  iou_threshold   : float in [0, 1]

S1  RUN
    Each engine in parallel: one thread per engine via ThreadPoolExecutor.
    A single engine raising does NOT kill the vote — its contribution
    becomes the empty tuple; the failure is logged at WARNING. The other
    engines proceed unchanged.

S2  TAG + SORT
    Every detection becomes (engine_name, Detection). The flat list is
    sorted by score DESCENDING (Python's stable sort).

S3  CLUSTER (greedy IoU)
    For each unclaimed detection i (taken in S2 order):
      seed = i; claim i.
      for each unclaimed j > i:
        if IoU(seed.bbox, j.bbox) >= iou_threshold:
          add j to seed's cluster; claim j.
    A claimed detection cannot seed another cluster or join a different one.
    IoU is computed against the SEED, not against any other cluster member.

S4  FILTER
    Keep a cluster iff |{engine_name in cluster}| >= min_votes.
    Counts UNIQUE engines — two detections from the same engine in one
    cluster contribute 1 vote, not 2.

S5  MERGE
    Each surviving cluster becomes one consensus Detection. See the
    field-by-field rules below.

OUTPUT
    Tuple of consensus Detections sorted by score DESCENDING. Each carries
    engine="consensus" and extras["voted_by"] listing the contributing
    engine names.
```

## Field-by-field merge rules

For each surviving cluster, exactly these rules produce the output
Detection. Every rule is deterministic given the cluster contents.

| Output field | Rule |
|---|---|
| `bbox_xyxy` | per-coordinate **median** across cluster members. `(median(x1), median(y1), median(x2), median(y2))`. Robust against one engine reporting a slightly off bbox. |
| `score` | arithmetic **mean** of cluster members' scores. |
| `symbology` | Most common symbology across cluster members. **Tie at the top count → among the members whose symbology sits at the top count, the highest-scored member's symbology wins; any remaining tie breaks stably on the symbology's string value.** |
| `payload` | If any non-None payloads exist in the cluster: most common non-None payload. **Tie at the top count → the highest-scored member's payload wins if it sits at the top count; otherwise the highest-scored member whose payload sits at the top count wins (the S-077 deterministic pick).** If no non-None payloads: `None`. |
| `polygon` | The highest-scored member's polygon. Medians don't generalize cleanly to rotated quads. |
| `engine` | Hardcoded literal string `"consensus"`. |
| `extras.voted_by` | Sorted tuple of unique engine names that contributed. |
| `extras.vote_count` | Count of unique engine names (= `len(voted_by)`). |
| `extras.agreed_payloads` | Sorted tuple of distinct non-None payloads observed in the cluster. Lets callers see when engines decoded the same crop differently. |
| `extras.source_count` | Total detections in the cluster. May exceed `vote_count` if an engine emitted multiple overlapping detections that all landed in the same cluster. |

## Worked examples

### Example 1 — image with 4 distinct codes, 2 engines, full agreement

Image contains a QR, a Code 128, a DataMatrix, and an EAN-13, occupying
distinct regions. Both `arbez` and `zxing` detect all 4 and decode the
same payloads.

```
After S2 (sorted desc by score):
  (arbez, QR_0.97), (arbez, DM_0.94), (zxing, QR_0.93),
  (arbez, C128_0.92), (zxing, DM_0.91), (zxing, EAN13_0.90),
  (arbez, EAN13_0.88), (zxing, C128_0.86)

After S3 (greedy clustering by IoU):
  cluster 1: seed=(arbez, QR_0.97);  absorbs (zxing, QR_0.93).
  cluster 2: seed=(arbez, DM_0.94);  absorbs (zxing, DM_0.91).
  cluster 3: seed=(arbez, C128_0.92); absorbs (zxing, C128_0.86).
  cluster 4: seed=(zxing, EAN13_0.90); absorbs (arbez, EAN13_0.88).
  (Distinct codes have IoU ~0 between their bboxes, so they don't
  cross-contaminate.)

After S4 (vote_count >= min_votes=2): all 4 clusters survive.

After S5 (per-cluster merge):
  Each output Detection has engine="consensus", voted_by=("arbez", "zxing"),
  vote_count=2, source_count=2, payload from the only-non-None pool,
  bbox = per-coord median of the two members' bboxes.

Output: 4 consensus Detections, sorted by their consensus scores.
```

### Example 2 — same scene, but the two engines decode DIFFERENT payloads

Both engines find all 4 codes, but `arbez` decodes payloads `A1, A2, A3, A4`
while `zxing` decodes `B1, B2, B3, B4`. Clustering is the same as
Example 1 — the bboxes still overlap. The interesting part is per-cluster
payload tiebreak.

Take cluster 1 (the QR):

```
cluster 1 members:
  (arbez, Detection(payload="A1", score=0.97))
  (zxing, Detection(payload="B1", score=0.93))

Payload tiebreak:
  payloads = ["A1", "B1"]      # both tied at count=1
  top_n = 1

  best_det = max(members, key=score)
           = (arbez, Detection(payload="A1", score=0.97))

  if best_det.payload is not None and payload_counts["A1"] == top_n:
      # "A1" count is 1, equal to top_n → take best_det's payload
      payload = "A1"

extras.agreed_payloads = ("A1", "B1")   # disagreement is preserved + visible
```

The highest-score detection wins the payload tiebreak. The other
engine's payload is preserved in `agreed_payloads` so downstream
code can audit the disagreement.

**Edge case (rare):** if both engines' scores are exactly equal
(say both 0.93), `best_det` resolves to whichever detection
appeared first in the post-S2 tagged list — which depends on the
order `as_completed` returned the engines' futures (Stage 1) →
**non-deterministic across runs**. See "Non-determinism sources"
below.

### Example 3 — `min_votes=2`, but only one engine detected this code

A faint Code 128 in the image; `arbez` saw it (one detection), `zxing`
didn't.

```
cluster N: [(arbez, C128_0.74)]    # only one member, one unique engine

After S4: vote_count = 1 < min_votes = 2 → cluster DROPPED.
```

The detection doesn't appear in the consensus output. In the S-075
default path (`min_votes=1`) it would survive.

### Example 4 — same engine emits 2 overlapping detections in one cluster

`arbez` sometimes emits multiple overlapping detections for the same
physical code (YOLOX-s artifact). Suppose `arbez` returns two QR
detections with overlap, and `zxing` returns one.

```
cluster 1 members:
  (arbez, QR_0.97)
  (arbez, QR_0.93)
  (zxing, QR_0.91)

After S4: unique engine count = |{arbez, zxing}| = 2, >= min_votes=2 → survives.

extras:
  voted_by      = ("arbez", "zxing")    # set of unique names
  vote_count    = 2                      # |voted_by|
  source_count  = 3                      # total members
```

`source_count > vote_count` is the signal that one engine "double-voted"
into the same cluster. The vote count still tracks the unique-engine
agreement.

## Non-determinism sources

The consensus result is deterministic **except** in one named
tied-value corner (N1). It is rare in practice and observable only
when explicitly hit.

### N1 — equal scores from different engines

```
Where: Stage 2 sort is stable. Ties in score preserve the input
       relative order. The input order is per_engine.items() = dict
       insertion order = the order as_completed returned the engine
       futures = non-deterministic across runs.

Effect: which detection becomes a cluster's seed can flip across
        runs. The cluster MEMBERS are identical (greedy clustering
        is symmetric over overlapping bboxes), but bbox medians +
        the score-driven payload tiebreak can flip.

When does this bite: only when two detections from different engines
        have BIT-IDENTICAL scores. Real model confidences are
        float-valued, so this is rare. zxing and wechat return
        constant proxy scores (score=1.0 for every decode), so two
        constant-score engines disagreeing on the same crop will
        always hit N1.
```

### Former N2 — Counter ties in symbology / payload aggregation (resolved)

Earlier releases resolved symbology / payload count ties via
`Counter.most_common`'s first-encountered pick, which inherited
N1's ordering. Both rules now resolve count ties by score instead:

- **payload** — the S-077 deterministic pick: the highest-scored
  member whose payload sits at the top count wins.
- **symbology** — among the members whose symbology sits at the top
  count, the highest-scored member's symbology wins, with a final
  stable tiebreak on the symbology's string value.

Given distinct scores, both picks are fully deterministic. The
symbology rule's value-based final tiebreak is deterministic even
under tied scores; the payload pick under bit-identical scores
reduces to N1.

## Determinism guarantees (independent of N1)

- Set of surviving cluster regions: deterministic (greedy IoU is geometry-only).
- `vote_count`, `source_count`, `voted_by`, `agreed_payloads`: deterministic.
- `bbox_xyxy` medians: deterministic (median is order-independent).
- `score` mean: deterministic.
- `symbology`: deterministic — count ties resolve by score, then
  stably by the symbology's string value.

The non-determinism only affects:

- `bbox_xyxy` when N1 swaps which detections seed clusters (and so the median's input set changes).
- `payload` when N1 produces bit-identical scores among the members whose payload sits at the top count (the S-077 pick is score-driven).

## See also

- `src/arbez/consensus.py` — the implementation. Same shape as this spec.
- [`concepts.md`](concepts.md) — how Scanner orchestrates consensus.
- [`api-reference.md`](api-reference.md) — Scanner constructor parameters.
- [`DECISIONS.md`](../DECISIONS.md) — S-032 (consensus voting locked
  from v0.0.18), S-075 (bare `Scanner()` default consensus).
