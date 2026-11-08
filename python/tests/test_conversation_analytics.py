"""Module F: conversation analytics.

Invariants only: the module must compute behavioural aggregates from timestamps and
senders, must never echo raw text, and must refuse psychological inference.
"""
from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from attraction_engine.analytics import conversation


def msgs(pairs, start=datetime(2026, 1, 1, 9, 0, 0), gap_min=30):
    out = []
    for i, (sender, text) in enumerate(pairs):
        out.append({"sender": sender, "timestamp": start + timedelta(minutes=gap_min * i),
                    "text": text})
    return out


def test_balanced_conversation_metrics():
    data = msgs([("me", "hello there"), ("them", "hi, how was the climb"),
                 ("me", "long, the ridge was windy"), ("them", "sounds worth it")])
    out = conversation.analyse(data, me="me")
    assert out["messages"] == 4
    # Only "me" opened the conversation, so the initiation balance is 1.0 by the
    # documented definition (share opened by me); message balance is even.
    assert out["initiation_balance"] == pytest.approx(1.0)
    assert out["message_share"]["me"] == pytest.approx(0.5)
    assert 0.0 <= out["reciprocity"]["index"] <= 1.0
    assert out["limitations"], "the module must disclose its limits"


def test_one_sided_conversation_is_flagged():
    data = msgs([("me", "hello"), ("me", "still there?"), ("them", "yes"),
                 ("me", "great"), ("me", "so about Saturday")])
    out = conversation.analyse(data, me="me")
    assert out["initiation_balance"] == pytest.approx(1.0)
    assert out["engagement"]["double_texts"] >= 1
    assert out["message_share"]["me"] > 0.6


def test_sessions_split_on_long_gaps():
    start = datetime(2026, 1, 1, 9, 0, 0)
    data = [
        {"sender": "me", "timestamp": start, "text": "morning"},
        {"sender": "them", "timestamp": start + timedelta(minutes=2), "text": "morning"},
        # > 6 h gap => new session
        {"sender": "me", "timestamp": start + timedelta(hours=9), "text": "evening"},
        {"sender": "them", "timestamp": start + timedelta(hours=9, minutes=5),
         "text": "evening to you"},
    ]
    out = conversation.analyse(data, me="me")
    assert out["sessions"] == 2


def test_latency_is_measured_from_the_other_party_turn():
    start = datetime(2026, 1, 1, 9, 0, 0)
    data = [
        {"sender": "me", "timestamp": start, "text": "hi"},
        {"sender": "them", "timestamp": start + timedelta(minutes=10), "text": "hi"},
        {"sender": "me", "timestamp": start + timedelta(minutes=20), "text": "how are you"},
        {"sender": "them", "timestamp": start + timedelta(minutes=50), "text": "good"},
    ]
    out = conversation.analyse(data, me="me")
    lat = out["reply_latency_s"]
    # me replied 10 minutes after them -> one sample of 600 s. "them" has two
    # replies: 10 min after my first message and 30 min after my second, so the
    # median of {600, 1800} is 1200 s.
    assert lat["me"]["n"] == 1
    assert lat["me"]["median"] == pytest.approx(600, rel=0.01)
    assert lat["them"]["n"] == 2
    assert lat["them"]["median"] == pytest.approx(1200, rel=0.01)


def test_raw_text_is_never_returned():
    secret = "my exact street address is 14 Mall Road and my phone is 9999999999"
    out = conversation.analyse(msgs([("me", secret), ("them", "ok")]), me="me")
    assert secret not in str(out)
    assert "Mall Road" not in str(out)
    assert "9999999999" not in str(out)


def test_psychological_inference_is_refused():
    out = conversation.analyse(msgs([("me", "hi"), ("them", "hi")]), me="me")
    blob = str(out).lower()
    for banned in ("attachment style", "personality type", "narciss", "anxious-avoidant",
                   "emotional intelligence score", "toxicity"):
        assert banned not in blob


def test_sequence_model_is_reported_unavailable_when_untrained():
    out = conversation.analyse(msgs([("me", "hi"), ("them", "hi")]), me="me")
    # The product must not imply a trained language model where there is none.
    seq = out.get("sequence_model")
    if seq is not None:
        assert seq["available"] is False
        assert seq.get("reason")


def test_empty_or_single_message_input_is_handled():
    out = conversation.analyse(msgs([("me", "only one")]), me="me")
    assert out["messages"] == 1
    assert out["sessions"] == 1


def test_ordering_of_input_does_not_change_the_metrics():
    data = msgs([("me", "a"), ("them", "b"), ("me", "c"), ("them", "d")])
    forward = conversation.analyse(data, me="me")
    shuffled = conversation.analyse(list(reversed(data)), me="me")
    assert shuffled["messages"] == forward["messages"]
    assert shuffled["sessions"] == forward["sessions"]
    assert shuffled["message_share"] == forward["message_share"]
    # The analysis must sort by timestamp itself rather than trusting input order.
    assert shuffled["reply_latency_s"]["me"]["median"] == pytest.approx(
        forward["reply_latency_s"]["me"]["median"])
