import io
import json
import urllib.error

import pytest
from pydantic import ValidationError

from video_analyzer import LLMConfig, llm, polish_analysis
from video_analyzer.errors import AnalysisError
from video_analyzer.schema import (
    AnalysisConfig,
    AnalysisResult,
    DiarizationTurn,
    MediaInfo,
    Observation,
    Scene,
    Speaker,
    TranscriptUnit,
)


@pytest.fixture
def source(tmp_path):
    units = [
        TranscriptUnit(
            id=f"t{i}",
            start=i,
            end=i + 0.8,
            text="こんにちは " + "word " * (8 + i % 4),
            language="ja",
            speaker_id="speaker_01",
            scene_id="scene1",
        )
        for i in range(20)
    ]
    return AnalysisResult(
        application_version="0.1.0",
        report_language="ja",
        media=MediaInfo(
            path=str(tmp_path / "missing.mp4"),
            duration=20,
            width=160,
            height=120,
            video_stream_index=0,
            video_start=0,
        ),
        provenance={"configuration": {"diarization": True}, "elapsed_seconds": 5},
        speakers=[Speaker(id="speaker_01"), Speaker(id="speaker_02")],
        diarization=[DiarizationTurn(start=0, end=20, speaker_id="speaker_01")],
        exclusive_diarization=[DiarizationTurn(start=0, end=20, speaker_id="speaker_01")],
        transcript=units,
        scenes=[Scene(id="scene1", start=0, end=20, transcript_ids=[unit.id for unit in units])],
    )


@pytest.fixture
def config():
    return LLMConfig(
        base_url="http://localhost:8000/v1/",
        model="test-model",
        max_input_chars=2200,
        transcript_overlap_units=2,
    )


@pytest.fixture
def completions(monkeypatch):
    calls = []

    def complete(self, request_messages):
        payload = json.loads(request_messages[1]["content"])
        calls.append((request_messages, payload))
        if "core" in payload:
            return {
                "corrections": [
                    {
                        "id": unit["id"],
                        "text": "修正 " + unit["text"],
                        "speaker_id": "speaker_02" if payload["diarization_enabled"] else None,
                    }
                    for unit in payload["core"]
                ]
            }
        summary = payload["previous_summary"] or ""
        return {
            "description": summary
            + "/"
            + ",".join(str(item["start"]) for item in payload["new_observations"])
        }

    monkeypatch.setattr(llm.LLMClient, "complete", complete)
    monkeypatch.setattr(llm.time, "monotonic", lambda: 1.0)
    return calls


def test_overlapping_batches_have_single_ownership(source, config, completions, tmp_path):
    original = source.model_dump()
    result = polish_analysis(source, config, work_dir=tmp_path / "work")
    assert len(completions) > 2
    owned = []
    for index, (request_messages, payload) in enumerate(completions):
        assert len(llm.serialized(request_messages)) <= config.max_input_chars
        core = payload["core"]
        first, end = int(core[0]["id"][1:]), int(core[-1]["id"][1:]) + 1
        assert [unit["id"] for unit in payload["context_before"]] == [
            f"t{i}" for i in range(max(0, first - 2), first)
        ]
        assert [unit["id"] for unit in payload["context_after"]] == [
            f"t{i}" for i in range(end, min(20, end + 2))
        ]
        assert all(unit["text"].startswith("修正 ") for unit in payload["context_before"])
        assert all(not unit["text"].startswith("修正 ") for unit in payload["context_after"])
        if index == 0:
            assert payload["context_before"] == []
        owned.extend(unit["id"] for unit in core)
    assert completions[-1][1]["context_after"] == []
    assert owned == [unit.id for unit in source.transcript]
    assert source.model_dump() == original
    assert result.diarization == source.diarization
    assert result.exclusive_diarization == source.exclusive_diarization
    for before, after in zip(source.transcript, result.transcript, strict=True):
        assert after.model_dump(exclude={"text", "speaker_id"}) == before.model_dump(
            exclude={"text", "speaker_id"}
        )
        assert after.speaker_id == "speaker_02"
        assert after.text == "修正 " + before.text


def test_zero_overlap_and_budget_change(source, config, completions, tmp_path):
    zero = config.model_copy(update={"transcript_overlap_units": 0})
    polish_analysis(source, zero, work_dir=tmp_path / "work")
    count = len(completions)
    assert all(
        not payload["context_before"] and not payload["context_after"] for _, payload in completions
    )
    completions.clear()
    polish_analysis(
        source, zero.model_copy(update={"max_input_chars": 8000}), work_dir=tmp_path / "work"
    )
    assert len(completions) < count


def test_budget_cannot_fit_overlap_or_item(source, config, completions, tmp_path):
    small = config.model_copy(update={"max_input_chars": 10})
    with pytest.raises(AnalysisError, match="increase llm.max_input_chars.*overlap"):
        polish_analysis(source, small, work_dir=tmp_path / "work")
    assert not completions


def test_scene_continuation_covers_all_observations(source, config, completions, tmp_path):
    source.transcript = []
    source.scenes[0].observations = [
        Observation(
            start=i, end=i + 1, description=f"Visual event {i}. " + "x" * 600, frame_timestamps=[i]
        )
        for i in range(8)
    ]
    original = source.model_dump()
    result = polish_analysis(source, config, work_dir=tmp_path / "work")
    assert len(completions) > 2
    new_starts = []
    for index, (request_messages, payload) in enumerate(completions):
        assert len(llm.serialized(request_messages)) <= config.max_input_chars
        assert payload["report_language"] == "ja"
        if index:
            assert payload["previous_summary"]
            assert len(payload["context_observations"]) == 1
            assert payload["context_observations"][0]["start"] == new_starts[-1]
        else:
            assert payload["previous_summary"] is None
            assert payload["context_observations"] == []
        new_starts.extend(item["start"] for item in payload["new_observations"])
    assert new_starts == list(range(8))
    assert all(str(i) in result.scenes[0].description for i in range(8))
    assert result.scenes[0].observations == source.scenes[0].observations
    assert source.model_dump() == original


def test_scene_zero_overlap_and_empty_scene(source, config, completions, tmp_path):
    source.transcript = []
    result = polish_analysis(source, config, work_dir=tmp_path / "work")
    assert result.scenes[0].description is None
    assert not completions
    source.scenes[0].observations = [
        Observation(start=i, end=i + 1, description="x" * 700, frame_timestamps=[i])
        for i in range(5)
    ]
    polish_analysis(
        source,
        config.model_copy(update={"scene_overlap_observations": 0}),
        work_dir=tmp_path / "work",
    )
    assert len(completions) > 1
    assert all(not payload["context_observations"] for _, payload in completions)


@pytest.mark.parametrize(
    "invalid",
    [
        {"corrections": []},
        {"corrections": [{"id": "invented", "text": "test", "speaker_id": None}]},
        {"corrections": [{"id": "t0", "text": "test", "speaker_id": "invented"}]},
        {"corrections": [{"id": "t0", "text": "test", "speaker_id": 123}]},
        {"corrections": [{"id": "t0", "text": " ", "speaker_id": None}]},
        {"corrections": [{"id": "t0", "text": "test", "speaker_id": None, "start": 2}]},
        {"corrections": [{"id": "t0", "text": "test", "speaker_id": None}], "extra": True},
        {"corrections": [{"id": "t0", "text": "test", "speaker_id": None}] * 2},
    ],
)
def test_invalid_corrections_not_applied_or_checkpointed(
    source, config, monkeypatch, tmp_path, invalid
):
    source.transcript = source.transcript[:1]
    original = source.model_dump()
    monkeypatch.setattr(llm.LLMClient, "complete", lambda *_: invalid)
    with pytest.raises(AnalysisError, match="transcript batch 1.*invalid corrections"):
        polish_analysis(source, config, work_dir=tmp_path / "work")
    assert source.model_dump() == original
    assert not list((tmp_path / "work").rglob("transcript/*.json"))


def test_response_cannot_edit_overlap(source, config, completions, monkeypatch, tmp_path):
    valid_complete = llm.LLMClient.complete

    def bad(self, request_messages):
        response = valid_complete(self, request_messages)
        payload = completions[-1][1]
        if payload["context_after"]:
            unit = payload["context_after"][0]
            response["corrections"].append({"id": unit["id"], "text": "oops", "speaker_id": None})
        return response

    monkeypatch.setattr(llm.LLMClient, "complete", bad)
    with pytest.raises(AnalysisError, match="invalid corrections"):
        polish_analysis(source, config, work_dir=tmp_path / "work")


@pytest.mark.parametrize(
    "invalid", [{"description": " "}, {"description": 5}, {"description": "ok", "extra": 1}]
)
def test_invalid_scene_response(source, config, monkeypatch, tmp_path, invalid):
    source.transcript = []
    source.scenes[0].observations = [
        Observation(start=0, end=1, description="scene", frame_timestamps=[0])
    ]
    monkeypatch.setattr(llm.LLMClient, "complete", lambda *_: invalid)
    with pytest.raises(AnalysisError, match="scene scene1, batch 1.*invalid"):
        polish_analysis(source, config, work_dir=tmp_path / "work")


def test_diarization_disabled(source, config, completions, tmp_path):
    source.provenance["configuration"]["diarization"] = False
    for unit in source.transcript:
        unit.speaker_id = None
    result = polish_analysis(source, config, work_dir=tmp_path / "work")
    assert all(unit.speaker_id is None for unit in result.transcript)
    assert all(not payload["allowed_speakers"] for _, payload in completions)


def test_resume_replays_polished_overlap_and_scene_summary(
    source, config, completions, monkeypatch, tmp_path
):
    source.scenes[0].observations = [
        Observation(start=i, end=i + 1, description="x" * 600, frame_timestamps=[i])
        for i in range(8)
    ]
    expected = polish_analysis(source, config, work_dir=tmp_path / "uninterrupted")
    expected_requests = len(completions)
    completions.clear()
    complete = llm.LLMClient.complete

    def fail(self, request_messages):
        # Fail after both transcript and scene work have been accepted.
        if (
            "previous_summary" in request_messages[1]["content"]
            and json.loads(request_messages[1]["content"])["previous_summary"]
        ):
            raise AnalysisError("endpoint offline")
        return complete(self, request_messages)

    with monkeypatch.context() as patch:
        patch.setattr(llm.LLMClient, "complete", fail)
        with pytest.raises(AnalysisError, match="scene scene1, batch 2.*endpoint offline"):
            polish_analysis(source, config, work_dir=tmp_path / "resumed")
    accepted = len(completions)
    resumed = polish_analysis(source, config, work_dir=tmp_path / "resumed", resume=True)
    assert resumed == expected
    assert len(completions) == expected_requests
    assert accepted > 1
    count = len(completions)
    source.provenance["elapsed_seconds"] = 999
    polish_analysis(source, config, work_dir=tmp_path / "resumed", resume=True)
    assert len(completions) == count


@pytest.mark.parametrize(
    "change",
    [
        {"model": "other"},
        {"base_url": "http://other/v1"},
        {"transcript_overlap_units": 0},
        {"max_input_chars": 8000},
    ],
)
def test_changed_settings_repeat_polishing(source, config, completions, tmp_path, change):
    polish_analysis(source, config, work_dir=tmp_path / "work")
    count = len(completions)
    polish_analysis(
        source, config.model_copy(update=change), work_dir=tmp_path / "work", resume=True
    )
    assert len(completions) > count


def test_timeout_and_key_rotation_reuse_cache(source, config, completions, tmp_path):
    polish_analysis(source, config, work_dir=tmp_path / "work")
    count = len(completions)
    polish_analysis(
        source,
        config.model_copy(update={"timeout": 999, "api_key_env": "ROTATED"}),
        work_dir=tmp_path / "work",
        resume=True,
    )
    assert len(completions) == count


@pytest.mark.parametrize(
    "field,value",
    [
        ("timeout", 0),
        ("max_input_chars", 0),
        ("transcript_overlap_units", -1),
        ("scene_overlap_observations", True),
        ("model", " "),
        ("base_url", "file:///tmp/test"),
        ("base_url", "http://user:secret@host/v1"),
        ("base_url", "https://host/v1?token=secret"),
    ],
)
def test_invalid_configuration(field, value):
    with pytest.raises(ValidationError):
        LLMConfig(**{field: value})


def test_explicit_endpoint_required(source, tmp_path):
    assert AnalysisConfig().llm_enabled is False
    with pytest.raises(ValidationError, match="explicit llm.base_url and llm.model"):
        AnalysisConfig(llm_enabled=True)
    with pytest.raises(AnalysisError, match="explicit llm.base_url and llm.model"):
        polish_analysis(source, LLMConfig(), work_dir=tmp_path)


def test_old_json_loads_without_scene_description(source):
    data = source.model_dump()
    del data["scenes"][0]["description"]
    assert AnalysisResult.model_validate(data).scenes[0].description is None


class Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def envelope(content='{"description":"ok"}', finish_reason="stop"):
    return json.dumps(
        {"choices": [{"finish_reason": finish_reason, "message": {"content": content}}]}
    ).encode()


@pytest.mark.parametrize("authenticated", [False, True])
def test_client_request_endpoint_and_optional_auth(config, monkeypatch, authenticated):
    monkeypatch.delenv(config.api_key_env, raising=False)
    if authenticated:
        monkeypatch.setenv(config.api_key_env, "secret-key")
    client = llm.LLMClient(config)
    requests = []

    def open_request(request, timeout):
        requests.append(request)
        assert timeout == 120
        return Response(envelope())

    monkeypatch.setattr(client.opener, "open", open_request)
    request_messages = llm.messages("JSON only", {"test": "日本語"})
    assert client.complete(request_messages) == {"description": "ok"}
    request = requests[0]
    assert request.full_url == "http://localhost:8000/v1/chat/completions"
    assert request.get_method() == "POST"
    assert json.loads(request.data) == {
        "model": "test-model",
        "messages": request_messages,
        "stream": False,
    }
    assert request.get_header("Authorization") == ("Bearer secret-key" if authenticated else None)


@pytest.mark.parametrize(
    "status,retries", [(401, 0), (403, 0), (400, 0), (302, 0), (429, 2), (503, 2)]
)
def test_http_failures_and_retries_hide_provider_body(config, monkeypatch, status, retries):
    client = llm.LLMClient(config)
    calls, waits = [], []

    def fail(*args, **kwargs):
        calls.append(1)
        raise urllib.error.HTTPError(
            "http://host", status, "secret-key", {}, io.BytesIO(b"secret-key")
        )

    monkeypatch.setattr(client.opener, "open", fail)
    monkeypatch.setattr(llm.time, "sleep", waits.append)
    with pytest.raises(AnalysisError, match=f"HTTP {status}") as error:
        client.complete([])
    assert "secret-key" not in str(error.value)
    assert len(calls) == retries + 1
    assert waits == [1, 2][:retries]


def test_connection_retries_then_success(config, monkeypatch):
    client = llm.LLMClient(config)
    calls = []

    def open_request(*args, **kwargs):
        calls.append(1)
        if len(calls) <= 2:
            raise urllib.error.URLError("secret-key")
        return Response(envelope())

    monkeypatch.setattr(client.opener, "open", open_request)
    monkeypatch.setattr(llm.time, "sleep", lambda _: None)
    assert client.complete([]) == {"description": "ok"}
    assert len(calls) == 3


@pytest.mark.parametrize(
    "body",
    [
        b"not JSON secret-key",
        b"{}",
        b'{"choices":["invalid choice"]}',
        envelope("not JSON secret-key"),
        envelope(finish_reason="length"),
        envelope('{"description":"a","description":"b"}'),
        envelope("NaN"),
        envelope(None),
    ],
)
def test_malformed_endpoint_responses_hide_body(config, monkeypatch, body):
    client = llm.LLMClient(config)
    monkeypatch.setattr(client.opener, "open", lambda *args, **kwargs: Response(body))
    with pytest.raises(AnalysisError, match="invalid or incomplete JSON") as error:
        client.complete([])
    assert "secret-key" not in str(error.value)


def test_redirects_are_not_followed():
    assert llm.NoRedirect().redirect_request(None, None, 302, "", {}, "http://elsewhere") is None
