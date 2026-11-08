"""Module F: descriptive conversation analytics.

Scope limits, enforced in code:
  * only the user's own exported records are accepted;
  * participant names are replaced by "me"/"them" before storage;
  * no trait, emotion, personality or intent inference is performed — the outputs
    are counts, times, lengths and reciprocity ratios;
  * no message is generated for sending and no automated outreach exists.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime

import numpy as np

SESSION_GAP_MINUTES = 45.0     # a silence longer than this starts a new session


def _parse_ts(value) -> datetime:
    if isinstance(value, datetime):
        return value
    text = str(value).strip().replace("Z", "+00:00")
    return datetime.fromisoformat(text)


def analyse(messages: list[dict], me: str = "me", mine_label: str = "me") -> dict:
    """messages: [{sender, timestamp, text}]. Sender identity is anonymised here."""
    if not messages:
        raise ValueError("no messages supplied")
    rows = []
    for m in messages:
        ts = _parse_ts(m["timestamp"])
        sender = mine_label if str(m.get("sender", "")).lower() in {me.lower(), "me"} else "them"
        text = str(m.get("text", ""))
        rows.append({"sender": sender, "ts": ts, "text": text,
                     "length": len(text.split())})
    rows.sort(key=lambda r: r["ts"])

    gaps = [(rows[i]["ts"] - rows[i - 1]["ts"]).total_seconds() for i in range(1, len(rows))]
    gaps = [max(0.0, g) for g in gaps]

    # sessions
    sessions = [[rows[0]]]
    for i, r in enumerate(rows[1:], start=1):
        if gaps[i - 1] > SESSION_GAP_MINUTES * 60:
            sessions.append([r])
        else:
            sessions[-1].append(r)

    per_sender: dict[str, list[dict]] = {"me": [], "them": []}
    for r in rows:
        per_sender[r["sender"]].append(r)

    reply_latencies = {"me": [], "them": []}
    for i in range(1, len(rows)):
        prev, cur = rows[i - 1], rows[i]
        if cur["sender"] != prev["sender"]:
            reply_latencies[cur["sender"]].append(
                (cur["ts"] - prev["ts"]).total_seconds())
    # the first unanswered outgoing message counts as an initiation
    initiations = {"me": 0, "them": 0}
    for s in sessions:
        initiations[s[0]["sender"]] += 1

    def pct(vals, p):
        return round(float(np.percentile(vals, p)), 1) if vals else None

    def stats(vals):
        if not vals:
            return {"n": 0, "median": None, "p90": None, "mean": None}
        return {"n": len(vals), "median": round(float(np.median(vals)), 1),
                "p90": pct(vals, 90), "mean": round(float(np.mean(vals)), 1)}

    total_len = {k: sum(r["length"] for r in v) for k, v in per_sender.items()}
    counts = {k: len(v) for k, v in per_sender.items()}
    reciprocity = (deficit(counts, total_len, reply_latencies))

    duration_h = (rows[-1]["ts"] - rows[0]["ts"]).total_seconds() / 3600.0

    # conversation continuity: how often a session is followed within 24h
    continuations = 0
    for i in range(1, len(sessions)):
        gap_h = (sessions[i][0]["ts"] - sessions[i - 1][-1]["ts"]).total_seconds() / 3600.0
        if gap_h <= 24:
            continuations += 1
    continuity = continuations / max(1, len(sessions) - 1)

    # topic transitions: Jaccard distance between consecutive message token sets
    def toks(text):
        return {t for t in text.lower().split() if len(t) > 3}
    jaccards = []
    for i in range(1, len(rows)):
        a, b = toks(rows[i - 1]["text"]), toks(rows[i]["text"])
        if a and b:
            jaccards.append(1.0 - len(a & b) / len(a | b))
    topic_shift = float(np.mean(jaccards)) if jaccards else None

    ret = {
        "messages": len(rows),
        "duration_hours": round(duration_h, 2),
        "sessions": len(sessions),
        "session_gap_minutes": SESSION_GAP_MINUTES,
        "initiations": initiations,
        "initiation_balance": round(
            initiations["me"] / max(1, initiations["me"] + initiations["them"]), 4),
        "message_counts": counts,
        "message_share": {k: round(v / max(1, len(rows)), 4) for k, v in counts.items()},
        "word_counts": total_len,
        "mean_words_per_message": {k: round(total_len[k] / max(1, counts[k]), 2)
                                   for k in total_len},
        "reply_latency_s": {k: stats(v) for k, v in reply_latencies.items()},
        "message_length_distribution": {
            k: {
                "p10": pct([r["length"] for r in v], 10),
                "median": pct([r["length"] for r in v], 50),
                "p90": pct([r["length"] for r in v], 90),
                "max": max((r["length"] for r in v), default=0),
            } for k, v in per_sender.items()
        },
        "continuity": round(float(continuity), 4),
        "mean_topic_shift_jaccard": None if topic_shift is None else round(topic_shift, 4),
        "gap_distribution_s": stats(gaps),
        "reciprocity": reciprocity,
        "engagement": engagement_patterns(rows, sessions),
        "limitations": [
            "Descriptive statistics only; no inference about the other person's traits, "
            "feelings or intent is made.",
            "Reply latency reflects device and app behaviour as much as interest.",
            "Session boundaries use a fixed 45-minute gap, a modelling choice.",
            "With fewer than ~40 messages the distributions are too sparse to interpret.",
        ],
        "sequence_model": sequence_model_note(len(rows)),
    }
    return ret


def deficit(counts: dict, words: dict, latency: dict) -> dict:
    """Reciprocity index in [0,1]: 1 means perfectly balanced participation."""
    def ratio_balance(a: float, b: float) -> float:
        hi, lo = max(a, b), min(a, b)
        return 0.0 if hi == 0 else lo / hi

    msg_balance = ratio_balance(counts["me"], counts["them"])
    word_balance = ratio_balance(words["me"], words["them"])
    lat_med = {k: (float(np.median(v)) if v else 0.0) for k, v in latency.items()}
    lat_balance = ratio_balance(lat_med["me"], lat_med["them"]) if min(lat_med.values()) > 0 else 0.0
    index = 0.45 * msg_balance + 0.35 * word_balance + 0.20 * lat_balance
    return {
        "index": round(float(index), 4),
        "message_balance": round(msg_balance, 4),
        "word_balance": round(word_balance, 4),
        "latency_balance": round(lat_balance, 4),
        "note": "Balance is symmetric: being far ahead or far behind both lower the index.",
    }


def engagement_patterns(rows: list[dict], sessions: list[list[dict]]) -> dict:
    hours = Counter(r["ts"].hour for r in rows)
    longest = max(sessions, key=len) if sessions else []
    return {
        "messages_per_session": round(len(rows) / max(1, len(sessions)), 2),
        "longest_session_messages": len(longest),
        "busiest_hours_utc": [h for h, _ in hours.most_common(3)],
        "double_texts": _double_texts(rows),
    }


def _double_texts(rows: list[dict]) -> int:
    """Consecutive messages from the same sender before any reply (descriptive)."""
    n = 0
    for i in range(1, len(rows)):
        if rows[i]["sender"] == rows[i - 1]["sender"]:
            n += 1
    return n


def sequence_model_note(n: int) -> dict:
    return {
        "available": False,
        "reason": "A sequence model needs several hundred conversations from the same user. "
                  f"This import has {n} messages in a single conversation, so the product "
                  "reports descriptive statistics only.",
        "threshold_messages": 500,
        "threshold_conversations": 30,
    }
