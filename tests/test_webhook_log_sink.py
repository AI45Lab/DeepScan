from llm_diagnose.run import _WebhookLogSink


def test_webhook_log_sink_appends_job_id_and_payload() -> None:
    captured = {}
    sink = _WebhookLogSink("http://example.com/api/internal/log", "123")

    def _fake_send(url, payload):  # type: ignore[no-untyped-def]
        captured["url"] = url
        captured["payload"] = payload

    sink._send_json = _fake_send  # type: ignore[assignment]
    sink.log("abc")

    assert "jobId=123" in captured["url"]
    assert captured["payload"] == {"jobId": "123", "log": "abc"}
