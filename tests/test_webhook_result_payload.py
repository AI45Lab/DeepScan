from llm_diagnose.run import _WebhookSink


def _sample_run_result():
    return {
        "run_id": "job-123",
        "evaluators": [
            {"id": "x-boundary", "type": "xboundary"},
            {"id": "tellme", "type": "tellme"},
            {"id": "spin", "type": "spin"},
        ],
        "models": [
            {
                "evaluations": [
                    {
                        "evaluator": {"id": "x-boundary", "type": "xboundary"},
                        "results": {
                            "metrics": {
                                "per_layer": {
                                    "9": {
                                        "separation_score": 21.0,
                                        "boundary_ratio": 1.0,
                                        "details": {"dist_bound_safe": 12.0, "dist_bound_harmful": 11.5},
                                    }
                                }
                            }
                        },
                    },
                    {
                        "evaluator": {"id": "tellme", "type": "tellme"},
                        "results": {
                            "metrics": {
                                "R_diff": 0.42,
                                "R_same": 0.18,
                                "R_gap": 0.24,
                                "erank": 512.3,
                                "cos_sim": 0.12,
                                "pcc": 0.34,
                                "L1": 5.6,
                                "L2": 4.2,
                                "hausdorff": 0.78,
                            }
                        },
                    },
                    {
                        "evaluator": {"id": "spin", "type": "spin"},
                        "results": {
                            "totals": {
                                "fairness_privacy_neurons_coupling_ratio": 0.22,
                            },
                            "layers": [
                                {
                                    "layer_idx": 0,
                                    "fairness_privacy_neurons_coupling_ratio": 0.1,
                                    "totals": {"coupled": 1, "total_neurons": 10},
                                },
                                {
                                    "layer_idx": 1,
                                    "fairness_privacy_neurons_coupling_ratio": 0.2,
                                    "totals": {"coupled": 2, "total_neurons": 10},
                                },
                            ],
                        },
                    },
                ]
            }
        ],
    }


def test_webhook_sink_formats_diagnosis_report() -> None:
    sink = _WebhookSink("http://example.com/progress", "job-123")
    payload = sink._format_diagnosis_report(_sample_run_result())

    assert payload is not None
    assert payload["jobId"] == "job-123"
    assert payload["type"] == "diagnosis"

    entries = {item["name"]: item for item in payload["results"]}
    assert set(entries.keys()) == {"x-boundary", "tellme", "spin", "mi-peaks"}

    xb_metrics = entries["x-boundary"]["metrics"]
    xb_table = xb_metrics["table"]
    assert xb_table[0]["layer"] == 9
    assert xb_table[0]["boundary_ratio"] == 1.0
    assert xb_table[0]["dist_bound_safe"] == 12.0
    assert xb_metrics["overall"]["separation_score_avg"] == 21.0

    tellme_table = entries["tellme"]["metrics"]["table"]
    assert tellme_table[0]["r_diff"] == 0.42
    assert tellme_table[0]["r_same"] == 0.18
    assert tellme_table[0]["r_gap"] == 0.24
    assert tellme_table[0]["l2"] == 4.2
    assert tellme_table[0]["hausdorff"] == 0.78

    spin_metrics = entries["spin"]["metrics"]
    spin_table = spin_metrics["table"]
    assert spin_table[0]["fairness_privacy_neurons_coupling_ratio"] == 0.22
    assert spin_metrics["overall"]["fairness_privacy_neurons_coupling_ratio"] == 0.22
    charts = spin_metrics["charts"]
    assert charts[0]["type"] == "bar"
    assert charts[0]["data"] == [0.1, 0.2]

    charts = entries["mi-peaks"]["metrics"]["charts"]
    assert isinstance(charts, list)
    assert charts[0]["type"] == "line"
    line_chart = charts[0]["data"]
    assert isinstance(line_chart, list)
    assert len(line_chart) == 80
    assert line_chart[0] == 0.08
    assert line_chart[-1] == 0.07


def test_webhook_sink_posts_status_message() -> None:
    captured = {}
    sink = _WebhookSink("http://example.com/progress", "job-777")

    def _fake_send(url, payload, method="post"):  # type: ignore[no-untyped-def]
        captured["url"] = url
        captured["payload"] = payload
        captured["method"] = method

    sink._send_json = _fake_send  # type: ignore[assignment]
    sink.post_status_message(
        "all done",
        status="complete",
        progress=100.0,
        extra={"throughput": {"tokens_per_second": 70}},
    )

    assert "jobId=job-777" in captured["url"]
    assert captured["method"] == "post"
    assert captured["payload"] == {"status": "complete", "progress": 100.0, "perf": 70}


def test_webhook_sink_prefers_real_mi_peaks_metrics_when_present() -> None:
    sink = _WebhookSink("http://example.com/progress", "job-999")
    result = _sample_run_result()
    # Inject MI-Peaks evaluator output (lightweight; no heavy compute in unit tests).
    result["models"][0]["evaluations"].append(
        {
            "evaluator": {"id": "mi-peaks", "type": "mi-peaks"},
            "results": {
                "metrics": {
                    "mi_mean_trajectory": [0.1, 0.2, 0.3],
                    "charts": [{"type": "line", "data": [0.1, 0.2, 0.3]}],
                }
            },
        }
    )

    payload = sink._format_diagnosis_report(result)
    assert payload is not None
    entries = {item["name"]: item for item in payload["results"]}
    mi_charts = entries["mi-peaks"]["metrics"]["charts"]
    assert mi_charts[0]["type"] == "line"
    assert mi_charts[0]["data"] == [0.1, 0.2, 0.3]
