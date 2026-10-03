import re

from .schema import DiarizationTurn, Scene, Speaker, TranscriptUnit


def overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def normalize_speakers(regular: list[dict], exclusive: list[dict]):
    ids = {}
    for turn in sorted(regular + exclusive, key=lambda t: (t["start"], t["end"], t["speaker_id"])):
        ids.setdefault(turn["speaker_id"], f"speaker_{len(ids) + 1:02d}")

    def convert(turns):
        return [
            DiarizationTurn(**{**turn, "speaker_id": ids[turn["speaker_id"]]})
            for turn in sorted(turns, key=lambda t: (t["start"], t["end"]))
        ]

    return [Speaker(id=i) for i in ids.values()], convert(regular), convert(exclusive)


def audio_chunks(duration: float, core_seconds=20.0, context_seconds=5.0):
    index, start = 0, 0.0
    while start < duration - 1e-7:
        end = min(start + core_seconds, duration)
        yield (
            index,
            start,
            end,
            max(0.0, start - context_seconds),
            min(duration, end + context_seconds),
        )
        index += 1
        start = end


def speech_chunks(duration: float, turns: list[DiarizationTurn]):
    """Keep language detection/context within speech regions separated by silence."""
    regions = []
    for turn in sorted(turns, key=lambda turn: turn.start):
        start, end = max(0.0, turn.start - 0.3), min(duration, turn.end + 0.3)
        if regions and start <= regions[-1][1]:
            regions[-1][1] = max(regions[-1][1], end)
        else:
            regions.append([start, end])
    index = 0
    for region_start, region_end in regions:
        for _, core_start, core_end, clip_start, clip_end in audio_chunks(
            region_end - region_start
        ):
            yield (
                index,
                region_start + core_start,
                region_start + core_end,
                region_start + clip_start,
                region_start + clip_end,
            )
            index += 1


def owned_units(
    raw: list[dict], clip_start: float, core_start: float, core_end: float, duration: float
):
    result = []
    for unit in raw:
        start = max(0.0, min(duration, float(unit["start"]) + clip_start))
        end = max(start, min(duration, float(unit["end"]) + clip_start))
        midpoint = (start + end) / 2
        owned = core_start <= midpoint < core_end or midpoint == core_end == duration
        if owned and unit["text"].strip():
            result.append(
                TranscriptUnit(
                    start=start, end=end, text=unit["text"], language=unit.get("language")
                )
            )
    return result


def deduplicate_units(units: list[TranscriptUnit]) -> list[TranscriptUnit]:
    """Drop boundary duplicates only when identical text actually overlaps in time."""
    result = []
    for unit in sorted(units, key=lambda u: (u.start, u.end)):
        duplicate = any(
            other.text.strip() == unit.text.strip()
            and overlap(other.start, other.end, unit.start, unit.end)
            > 0.5 * min(other.end - other.start, unit.end - unit.start)
            for other in result[-8:]
        )
        if not duplicate:
            result.append(unit)
    return result


def assign_units(units: list[TranscriptUnit], turns: list[DiarizationTurn], scenes: list[Scene]):
    def score(unit, span):
        # Aligners can give short tokens identical start/end times. Keep their
        # text and treat these point timestamps as belonging to the containing span.
        if unit.start == unit.end:
            return float(span.start <= unit.start < span.end)
        return overlap(unit.start, unit.end, span.start, span.end)

    for scene in scenes:
        scene.transcript_ids = []
    for i, unit in enumerate(deduplicate_units(units)):
        unit.id = f"text_{i + 1:06d}"
        best = max(turns, key=lambda t: score(unit, t), default=None)
        unit.speaker_id = best.speaker_id if best and score(unit, best) > 0 else None
        scene = (
            scenes[-1]
            if unit.start == scenes[-1].end
            else max(scenes, key=lambda s: score(unit, s))
        )
        unit.scene_id = scene.id
        scene.transcript_ids.append(unit.id)
        yield unit


_CJK = re.compile(r"[\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af]")


def join_text(units: list[TranscriptUnit]) -> str:
    text = ""
    for unit in units:
        piece = unit.text
        if not text:
            text = piece.lstrip()
        elif text[-1:].isspace() or piece[:1].isspace():
            text += piece
        elif _CJK.search(text[-1:] + piece[:1]) or piece[:1] in ",.!?;:、。！？)]}」』":
            text += piece
        else:
            text += " " + piece
    return text.strip()
