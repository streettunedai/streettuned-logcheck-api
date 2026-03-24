from __future__ import annotations

from typing import Dict, List


def finalize_trust_buckets(
    trust_buckets: Dict[str, List[str]],
    invalid_reasons: Dict[str, str],
    uncertain: List[str],
    suspect: List[str],
) -> Dict[str, List[str]]:
    confirmed = set(trust_buckets["confirmed_channels"])
    missing = set(trust_buckets["missing_channels"])
    invalid = set(trust_buckets["invalid_channels"])
    suspect_set = set(trust_buckets["suspect_channels"])
    uncertain_set = set(trust_buckets["uncertain_channels"])

    for ch in invalid_reasons:
        if ch in confirmed:
            confirmed.remove(ch)
        invalid.add(ch)

    for ch in uncertain:
        if ch in confirmed:
            confirmed.remove(ch)
        uncertain_set.add(ch)

    for ch in suspect:
        if ch in confirmed:
            confirmed.remove(ch)
        suspect_set.add(ch)

    trust_buckets["confirmed_channels"] = sorted(confirmed)
    trust_buckets["missing_channels"] = sorted(missing)
    trust_buckets["invalid_channels"] = sorted(invalid)
    trust_buckets["suspect_channels"] = sorted(suspect_set)
    trust_buckets["uncertain_channels"] = sorted(uncertain_set)
    return trust_buckets
