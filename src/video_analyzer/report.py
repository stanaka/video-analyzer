import itertools
import re

from .schema import AnalysisResult
from .timeline import join_text


def timestamp(seconds: float) -> str:
    milliseconds = round(seconds * 1000)
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, milliseconds = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"


def escape(text: str) -> str:
    # Model output is report content, never raw HTML or Markdown instructions.
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return re.sub(r"([\\`*_{}\[\]()#+.!|>~-])", r"\\\1", text).replace("\n", " ")


def render_json(result: AnalysisResult) -> str:
    return result.model_dump_json(indent=2) + "\n"


def render_markdown(result: AnalysisResult) -> str:
    ja = result.report_language == "ja"
    lines = ["# 動画解析レポート" if ja else "# Video analysis report", ""]
    lines += [
        f"- {'動画' if ja else 'Video'}: {escape(result.media.path)}",
        f"- {'長さ' if ja else 'Duration'}: {timestamp(result.media.duration)}",
        f"- {'話者数' if ja else 'Speakers'}: {len(result.speakers)}",
        "",
    ]
    lines += ["## モデルと実行環境" if ja else "## Models and runtime", ""]
    for name, entry in result.provenance.get("models", {}).items():
        quant = f" ({entry['quantization']})" if entry.get("quantization") else ""
        lines.append(
            f"- {escape(name)}: {escape(entry['repo_id'])}{escape(quant)}; `{entry['revision']}`"
        )
    for stage, runtime in result.provenance.get("runtimes", {}).items():
        lines.append(f"- {escape(stage)}: {escape(runtime.get('device', 'unknown'))}")
    lines += [
        "",
        "モデルによる文字起こし・話者区分・映像説明には誤りが含まれる場合があります。"
        if ja
        else "Transcription, speaker assignments, and visual descriptions are model estimates.",
        "",
    ]
    if result.warnings:
        lines += ["## 注意事項" if ja else "## Warnings", ""]
        lines += [f"- {escape(warning)}" for warning in result.warnings]
        lines.append("")
    by_id = {unit.id: unit for unit in result.transcript}
    for number, scene in enumerate(result.scenes, 1):
        lines += [
            f"## {'シーン' if ja else 'Scene'} {number}: {timestamp(scene.start)} – {timestamp(scene.end)}",
            "",
            "### 映像の説明" if ja else "### Visual description",
            "",
        ]
        for observation in scene.observations:
            lines.append(
                f"- **{timestamp(observation.start)} – {timestamp(observation.end)}**: {escape(observation.description)}"
            )
        lines += ["", "### 発話" if ja else "### Speech", ""]
        units = [by_id[unit_id] for unit_id in scene.transcript_ids]
        if not units:
            lines.append("検出された発話はありません。" if ja else "No speech detected.")
        # Split at speaker changes or long pauses, not merely sentence punctuation.
        groups = []
        for _, speaker_units in itertools.groupby(units, key=lambda unit: unit.speaker_id):
            current = []
            for unit in speaker_units:
                if current and unit.start - current[-1].end > 1.5:
                    groups.append(current)
                    current = []
                current.append(unit)
            if current:
                groups.append(current)
        for group in groups:
            speaker = group[0].speaker_id or ("話者不明" if ja else "Unknown speaker")
            lines.append(
                f"- **{timestamp(group[0].start)} – {timestamp(group[-1].end)} · {escape(speaker)}**: {escape(join_text(group))}"
            )
        lines.append("")
    return "\n".join(lines)
