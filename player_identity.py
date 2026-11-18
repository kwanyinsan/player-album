"""Deterministic identity continuity across collection revisions."""
from collections import defaultdict
import re


def reconcile_player_ids(labels, records, previous_groups, next_number=1):
    members = defaultdict(set)
    for label, record in zip(labels, records):
        members[int(label)].add(record.local_player_id)
    old = {g["player_id"]: set(g["local_tracks"]) for g in previous_groups}
    if len(old) != len(previous_groups) or any(not re.fullmatch(r"P[0-9]+", p) for p in old):
        raise ValueError("Previous player IDs must be unique P-prefixed numbers")
    next_number = max([next_number, *[int(p[1:]) + 1 for p in old]])
    candidates = []
    for label, tracks in members.items():
        for player, prior in old.items():
            overlap = len(tracks & prior)
            if overlap:
                candidates.append((-overlap, player, min(tracks), label))
    assigned, used = {}, set()
    for _, player, _, label in sorted(candidates):
        if label not in assigned and player not in used:
            assigned[label] = player
            used.add(player)
    for label in sorted(members, key=lambda item: min(members[item])):
        if label not in assigned:
            assigned[label] = f"P{next_number:04d}"
            next_number += 1
    lineage = [{"player_id": assigned[label], "previous_player_ids": sorted(
        player for player, tracks in old.items() if tracks & members[label]
    )} for label in sorted(members)]
    return assigned, {"next_player_number": next_number, "lineage": lineage}
