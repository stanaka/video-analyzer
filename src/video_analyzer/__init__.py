"""Offline video analysis; importing the API never imports a model runtime."""

__version__ = "0.1.0"

from .report import render_json, render_markdown
from .schema import AnalysisConfig, AnalysisResult, LLMConfig


def analyze_video(path, config=None):
    """Analyze a local MP4 and return an AnalysisResult without writing a report."""
    from .pipeline import analyze_video as run

    return run(path, config or AnalysisConfig())


def write_html(result, output, *, video_path=None):
    """Export an offline HTML viewer and audio sidecars without rerunning analysis."""
    from .html_report import write_html as write

    return write(result, output, video_path=video_path)


def polish_analysis(result, llm_config, *, work_dir=None, resume=False):
    """Polish saved analysis text through a configured OpenAI-compatible endpoint."""
    from .llm import polish_analysis as polish

    return polish(result, llm_config, work_dir=work_dir, resume=resume)


__all__ = [
    "AnalysisConfig",
    "AnalysisResult",
    "LLMConfig",
    "analyze_video",
    "polish_analysis",
    "render_json",
    "render_markdown",
    "write_html",
]
