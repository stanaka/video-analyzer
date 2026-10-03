"""Offline video analysis; importing the API never imports a model runtime."""

__version__ = "0.1.0"

from .report import render_json, render_markdown
from .schema import AnalysisConfig, AnalysisResult


def analyze_video(path, config=None):
    """Analyze a local MP4 and return an AnalysisResult without writing a report."""
    from .pipeline import analyze_video as run

    return run(path, config or AnalysisConfig())


__all__ = ["AnalysisConfig", "AnalysisResult", "analyze_video", "render_json", "render_markdown"]
