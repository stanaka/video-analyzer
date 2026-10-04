"""Offline video analysis; importing the API never imports a model runtime."""

__version__ = "0.1.0"

from .report import render_json, render_markdown
from .schema import AnalysisConfig, AnalysisResult


def analyze_video(path, config=None):
    """Analyze a local MP4 and return an AnalysisResult without writing a report."""
    from .pipeline import analyze_video as run

    return run(path, config or AnalysisConfig())


def write_html(result, output, *, video_path=None):
    """Export an offline HTML viewer and audio sidecars without rerunning analysis."""
    from .html_report import write_html as write

    return write(result, output, video_path=video_path)


__all__ = [
    "AnalysisConfig",
    "AnalysisResult",
    "analyze_video",
    "render_json",
    "render_markdown",
    "write_html",
]
