import pytest

from llm_diagnose.utils.throughput import TokenThroughputTracker, count_tokens_from_batch


def test_count_tokens_prefers_attention_mask() -> None:
    batch = {"attention_mask": [[1, 1, 0], [1, 1, 1]]}
    assert count_tokens_from_batch(batch) == 5


def test_count_tokens_falls_back_to_input_ids_shape() -> None:
    batch = {"input_ids": [[1, 2, 3, 4, 5]]}
    assert count_tokens_from_batch(batch) == 5


def test_tracker_emits_to_sink() -> None:
    class _Sink:
        def __init__(self) -> None:
            self.payloads = []

        def on_throughput(self, payload):
            self.payloads.append(payload)

    sink = _Sink()
    tracker = TokenThroughputTracker(sink=sink, min_interval_seconds=0.0, min_tokens_delta=0)
    tracker.add_tokens(10)

    assert sink.payloads, "tracker should emit immediately when thresholds are zero"
    last = sink.payloads[-1]
    assert last["tokens"] == 10
    assert last["tokens_per_second"] > 0


def test_tracker_can_emit_zero_rate() -> None:
    class _Sink:
        def __init__(self) -> None:
            self.payloads = []

        def on_throughput(self, payload):
            self.payloads.append(payload)

    sink = _Sink()
    tracker = TokenThroughputTracker(sink=sink, min_interval_seconds=0.0, min_tokens_delta=0)
    tracker.add_tokens(5)
    tracker.emit_zero_rate(status="running")

    assert sink.payloads, "tracker should emit"
    last = sink.payloads[-1]
    assert last["tokens_per_second"] == 0.0
    assert last["tokens_per_second_avg"] == 0.0
    assert last["_status"] == "running"


def test_count_tokens_from_tensor_shape() -> None:
    torch = pytest.importorskip("torch")
    batch = torch.ones((2, 3, 4))
    # Shape product should be used when no mask/ids are available.
    assert count_tokens_from_batch(batch) == 24
