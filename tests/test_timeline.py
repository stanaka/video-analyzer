from video_analyzer.schema import DiarizationTurn, Scene, TranscriptUnit
from video_analyzer.timeline import (
    assign_units,
    audio_chunks,
    join_text,
    normalize_speakers,
    owned_units,
    speech_chunks,
)


def test_speaker_scene_boundaries_and_unknowns():
    turns = [
        DiarizationTurn(start=0, end=1.4, speaker_id="A"),
        DiarizationTurn(start=1.4, end=3, speaker_id="B"),
    ]
    scenes = [Scene(id="first", start=0, end=2), Scene(id="second", start=2, end=5)]
    units = [
        TranscriptUnit(start=1.3, end=1.7, text="hello"),
        TranscriptUnit(start=1.9, end=2.5, text="world"),
        TranscriptUnit(start=4, end=4.5, text="unassigned"),
    ]
    result = list(assign_units(units, turns, scenes))
    assert [u.speaker_id for u in result] == ["B", "B", None]
    assert [u.scene_id for u in result] == ["first", "second", "second"]
    assert sum(len(s.transcript_ids) for s in scenes) == 3


def test_context_boundary_ownership_and_repetitions():
    raw = [
        {"start": 19.6, "end": 20.2, "text": "once"},
        {"start": 20.4, "end": 20.8, "text": "again"},
    ]
    first = owned_units(raw, 0, 0, 20, 40)
    second = owned_units(
        [{**u, "start": u["start"] - 15, "end": u["end"] - 15} for u in raw], 15, 20, 40, 40
    )
    assert [u.text for u in first + second] == ["once", "again"]
    scenes = [Scene(id="s", start=0, end=40)]
    repeated = [
        TranscriptUnit(start=1, end=1.3, text="yes"),
        TranscriptUnit(start=1.1, end=1.3, text="yes"),
        TranscriptUnit(start=1.5, end=1.8, text="yes"),
    ]
    assert len(list(assign_units(repeated, [], scenes))) == 2


def test_chunks_cover_long_recordings_with_bounded_context():
    chunks = list(audio_chunks(3600.25))
    assert chunks[0][1] == 0
    assert chunks[-1][2] == 3600.25
    assert all(end - start <= 30 for _, _, _, start, end in chunks)
    assert all(left[2] == right[1] for left, right in zip(chunks, chunks[1:], strict=False))


def test_japanese_and_english_text():
    def units(*text):
        return [TranscriptUnit(start=0, end=1, text=t) for t in text]

    assert join_text(units("今日", "は", "晴れ", "です。")) == "今日は晴れです。"
    assert join_text(units("Hello", " world", ".")) == "Hello world."
    assert join_text(units("Good", "morning")) == "Good morning"


def test_global_speaker_order_preserves_overlaps():
    regular = [{"start": 2, "end": 4, "speaker_id": "Z"}, {"start": 1, "end": 3, "speaker_id": "Y"}]
    speakers, turns, exclusive = normalize_speakers(regular, regular[:1])
    assert [s.id for s in speakers] == ["speaker_01", "speaker_02"]
    assert turns[0].speaker_id == "speaker_01"
    assert exclusive[0].speaker_id == "speaker_02"
    assert turns[0].end > turns[1].start


def test_speech_regions_split_at_silence_and_keep_context_bounded():
    turns = [
        DiarizationTurn(start=1, end=5, speaker_id="A"),
        DiarizationTurn(start=7, end=50, speaker_id="B"),
        DiarizationTurn(start=30, end=35, speaker_id="A"),
    ]
    chunks = list(speech_chunks(60, turns))
    assert chunks[0][1:3] == (0.7, 5.3)
    assert chunks[1][1] == 6.7
    assert len(chunks) == 4
    assert all(end - start <= 30.000001 for _, _, _, start, end in chunks)
    assert chunks[-1][2] == 50.3


def test_zero_duration_tokens_are_retained_and_assigned_as_points():
    raw = [
        {"start": 2, "end": 2, "text": "は"},
        {"start": 2, "end": 2.2, "text": "晴れ"},
        {"start": 4, "end": 4, "text": "。"},
    ]
    units = owned_units(raw, 0, 0, 4, 4)
    scenes = [Scene(id="first", start=0, end=2), Scene(id="second", start=2, end=4)]
    turns = [DiarizationTurn(start=1, end=3, speaker_id="A")]
    result = list(assign_units(units, turns, scenes))
    assert join_text(result) == "は晴れ。"
    assert [u.scene_id for u in result] == ["second"] * 3
    assert [u.speaker_id for u in result] == ["A", "A", None]
