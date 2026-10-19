# Yield snapshot

The README benchmark copies [yield-snapshot.json](yield-snapshot.json).
`tests/test_readme_benchmark.py` fails when the two disagree. Change
the snapshot first, then the README, in the same commit.

This page is the counting contract. It is not a second set of results.

## One pass

Each photograph is opened once with `Scanner()`. The engines loaded on
the measured Mac were `arbez`, `apple_vision`, `zxing`, and `wechat`.
`Result.detections` is the union. `Result.per_engine` is each engine's
own detections. `consensus=2` and `consensus=3` are not extra scans.
They count payloads whose `vote_count` on that same union was at least
2 or at least 3.

## What a count is

A distinct code is one payload string. An image counts when that
configuration had at least one payload. Empty payloads are ignored.
The same string under two symbology labels is counted in both
symbology rows, so a column does not have to add up to the distinct
total.

The single-engine figures in the snapshot are payloads that appeared
with `vote_count == 1` and exactly one voter, and never with
`vote_count >= 2`. On this snapshot those payloads are Apple Vision
410, ZXing 31, arbez 11, and WeChat 3 (455). Payloads seen with
`vote_count >= 2` are 4,265. The union is 5,026. 455 + 4,265 is 306
short of 5,026, so those two figures are not a partition of the union.
A payload with no vote record is in neither figure. The README does
not print them as a split.

## What this snapshot is not

The June 2026 README counted 4,290 images and 5,014 codes. That count
included 14 synthesized format probes. Those 14 files were not in the
directory this snapshot scanned, so they are not in these numbers.
Do not put them back without measuring them.

Do not paste the BoofCV box score or the labeled-text field score into
this table. Those runs call `Scanner(engines=["arbez", "zxing", "wechat"])`
and they score different things.

## Replacing the snapshot

1. Scan the directory with one bare `Scanner()` per file.
2. Replace `yield-snapshot.json` with that summary and the provenance
   keys already in the file: commit, date, Python, package version,
   platform, photo count, absent probes, timeouts, errors, formats,
   and the engines that loaded.
3. Update the README benchmark so the test passes.
4. Leave a format out of the table when the run did not see it. Do not
   estimate it.
