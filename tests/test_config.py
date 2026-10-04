from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from video_analyzer import AnalysisConfig, cli
from video_analyzer.config import load_settings
from video_analyzer.errors import AnalysisError


@pytest.fixture
def captured_config(monkeypatch):
    captured = []

    def capture(path, config):
        captured.append((path, config))
        raise AnalysisError("Stop after capturing configuration")

    monkeypatch.setattr(cli, "analyze_video", capture)
    return captured


@pytest.mark.parametrize("document", ["", "# empty settings\n", "---\n", "{}"])
def test_empty_yaml_uses_defaults(tmp_path, captured_config, document):
    path = tmp_path / "settings.yaml"
    path.write_text(document)
    assert cli.main(["analyze", "input.mp4", "--config", str(path)]) == 1
    assert captured_config == [(Path("input.mp4"), AnalysisConfig())]


def test_all_analysis_fields_load_from_yaml(tmp_path, captured_config):
    settings = {
        "start_time": 10,
        "end_time": 90.5,
        "transcription": False,
        "diarization": True,
        "scene_description": False,
        "asr": "qwen",
        "diarization_backend": "pyannote",
        "profile": "cpu",
        "model_dir": "models",
        "runtime_dir": "runtime",
        "work_dir": "work",
        "resume": True,
        "language": "ja",
        "report_language": "en",
        "num_speakers": 3,
        "min_speakers": 2,
        "max_speakers": 4,
        "pyannote": {
            "clustering": {"threshold": 0.65, "Fa": 0.1, "Fb": 1.2},
            "segmentation": {"min_duration_off": 0.1},
        },
        "audio_stream": 1,
        "audio_normalization": "peak",
        "noise_reduction": True,
        "scene_threshold": 4.0,
        "min_scene_seconds": 1.0,
        "audio_device": "cpu",
        "diarization_device": "cpu",
        "vision_device": "cpu",
        "window_seconds": 10,
        "frames_per_window": 4,
        "frame_long_edge": 512,
        "max_new_tokens": 128,
        "worker_timeout": 120,
    }
    assert settings.keys() == AnalysisConfig.model_fields.keys()
    path = tmp_path / "settings.yaml"
    path.write_text(yaml.safe_dump(settings))
    assert cli.main(["analyze", "input.mp4", "--config", str(path)]) == 1
    assert captured_config[0][1] == AnalysisConfig(**settings)


@pytest.mark.parametrize(
    "value,seconds",
    [(90.5, 90.5), ("90.5", 90.5), ("01:30.5", 90.5), ("00:01:30.5", 90.5), ("90:00", 5400)],
)
def test_analysis_time_formats(value, seconds):
    config = AnalysisConfig(start_time=value, end_time=6000)
    assert config.start_time == seconds
    assert config.model_dump(mode="json")["start_time"] == seconds


@pytest.mark.parametrize(
    "value",
    [
        -1,
        "-00:01",
        True,
        False,
        "NaN",
        "inf",
        float("inf"),
        "01:60",
        "01:60:00",
        "1:2:3:4",
        "1::2",
        "1:2.3.4",
        "",
        None,
    ],
)
def test_invalid_analysis_start_time(value):
    with pytest.raises(ValidationError):
        AnalysisConfig(start_time=value)


@pytest.mark.parametrize("end", [0, 2, -1, True, "00:60", "NaN"])
def test_invalid_analysis_end_time(end):
    with pytest.raises(ValidationError):
        AnalysisConfig(start_time=2, end_time=end)


def test_range_cli_overrides_yaml_and_show_settings(tmp_path, captured_config, capsys):
    path = tmp_path / "settings.yaml"
    path.write_text('start_time: "01:30"\nend_time: "00:03:00.5"\n')
    assert (
        cli.main(
            ["analyze", "input.mp4", "--config", str(path), "--start-time", "0", "--show-settings"]
        )
        == 1
    )
    assert captured_config[0][1] == AnalysisConfig(start_time=0, end_time=180.5)
    assert "end_time: 180.5" in capsys.readouterr().err
    assert cli.main(["analyze", "input.mp4", "--config", str(path), "--end-time", "200"]) == 1
    assert captured_config[1][1] == AnalysisConfig(start_time=90, end_time=200)


def test_yaml_boolean_resolution_preserves_off_and_does_not_change_pyyaml(tmp_path):
    path = tmp_path / "settings.yaml"
    path.write_text(
        "audio_normalization: off\nnoise_reduction: false\nresume: TRUE\nlanguage: no\n"
    )
    settings = load_settings(path)
    assert settings == {
        "audio_normalization": "off",
        "noise_reduction": False,
        "resume": True,
        "language": "no",
    }
    assert AnalysisConfig(**settings).audio_normalization == "off"
    assert yaml.safe_load("off") is False


def test_relative_paths_use_working_directory(tmp_path, monkeypatch, captured_config):
    directory = tmp_path / "configs"
    directory.mkdir()
    config_path = directory / "settings.yaml"
    config_path.write_text("model_dir: models\nruntime_dir: runtime\nwork_dir: work\n")
    monkeypatch.chdir(tmp_path)
    assert cli.main(["analyze", "input.mp4", "--config", "configs/settings.yaml"]) == 1
    config = captured_config[0][1]
    for field, name in (("model_dir", "models"), ("runtime_dir", "runtime"), ("work_dir", "work")):
        assert getattr(config, field).resolve() == tmp_path / name


def test_explicit_defaults_and_boolean_flags_override_yaml(tmp_path, captured_config):
    path = tmp_path / "settings.yaml"
    path.write_text(
        "transcription: false\ndiarization: true\nscene_description: false\n"
        "audio_normalization: dynamic\nnoise_reduction: true\nresume: true\n"
        "asr: qwen\nprofile: cpu\n"
    )
    assert (
        cli.main(
            [
                "analyze",
                "input.mp4",
                "--config",
                str(path),
                "--transcription",
                "--no-diarization",
                "--scene-description",
                "--audio-normalization",
                "off",
                "--no-noise-reduction",
                "--no-resume",
                "--asr",
                "whisper",
                "--profile",
                "auto",
            ]
        )
        == 1
    )
    assert captured_config[0][1] == AnalysisConfig(diarization=False)


def test_partial_cli_overrides_preserve_other_nested_yaml_settings(tmp_path, captured_config):
    path = tmp_path / "settings.yaml"
    path.write_text(
        "min_speakers: 3\nmax_speakers: 1\npyannote:\n"
        "  clustering: {threshold: 0.65, Fa: 0.1, Fb: 1.2}\n"
        "  segmentation: {min_duration_off: 0.2}\n"
    )
    assert (
        cli.main(
            [
                "analyze",
                "input.mp4",
                "--config",
                str(path),
                "--pyannote-clustering-threshold",
                "0.6",
                "--max-speakers",
                "4",
            ]
        )
        == 1
    )
    config = captured_config[0][1]
    assert (config.min_speakers, config.max_speakers) == (3, 4)
    assert config.pyannote.overrides() == {
        "clustering": {"threshold": 0.6, "Fa": 0.1, "Fb": 1.2},
        "segmentation": {"min_duration_off": 0.2},
    }


@pytest.mark.parametrize(
    "flag,group,name,value",
    [
        ("--pyannote-clustering-threshold", "clustering", "threshold", 0.65),
        ("--pyannote-clustering-fa", "clustering", "Fa", 0.07),
        ("--pyannote-clustering-fb", "clustering", "Fb", 0.8),
        ("--pyannote-segmentation-min-duration-off", "segmentation", "min_duration_off", 0.0),
    ],
)
def test_tuning_cli_only(captured_config, flag, group, name, value):
    assert cli.main(["analyze", "input.mp4", flag, str(value)]) == 1
    assert captured_config[0][1].pyannote.overrides() == {group: {name: value}}


def test_exact_count_overrides_yaml_bounds(tmp_path, captured_config):
    path = tmp_path / "settings.yaml"
    path.write_text("min_speakers: 5\nmax_speakers: 1\n")
    assert cli.main(["analyze", "input.mp4", "--config", str(path), "--num-speakers", "2"]) == 1
    assert captured_config[0][1].num_speakers == 2


@pytest.mark.parametrize("show_settings", [False, True])
def test_show_settings_is_opt_in_and_prints_merged_yaml_before_analysis(
    tmp_path, monkeypatch, capsys, show_settings
):
    path = tmp_path / "settings.yaml"
    path.write_text(
        "work_dir: 作業\nnoise_reduction: true\n"
        "pyannote:\n  clustering: {threshold: 0.65, Fa: 0.1}\n",
        encoding="utf-8",
    )
    captured = []

    def analyze(_, config):
        output = capsys.readouterr()
        assert output.out == ""
        if show_settings:
            assert output.err.count("# Analysis settings\n") == 1
            assert "Null pyannote tuning values use the loaded model's defaults." in output.err
            assert "作業" in output.err
            settings = yaml.safe_load(output.err)
            assert settings == config.model_dump(mode="json")
            assert list(settings) == list(AnalysisConfig.model_fields)
            assert settings["profile"] == "auto"
            assert settings["noise_reduction"] is False
            assert settings["work_dir"] == "作業"
            assert settings["pyannote"] == {
                "clustering": {"threshold": 0.7, "Fa": 0.1, "Fb": None},
                "segmentation": {"min_duration_off": None},
            }
            assert "show_settings" not in settings
        else:
            assert output.err == ""
        captured.append(config)
        raise AnalysisError("Stop after checking startup output")

    monkeypatch.setattr(cli, "analyze_video", analyze)
    args = [
        "analyze",
        "input.mp4",
        "--config",
        str(path),
        "--no-noise-reduction",
        "--pyannote-clustering-threshold",
        "0.7",
    ]
    if show_settings:
        args.append("--show-settings")
    assert cli.main(args) == 1
    assert len(captured) == 1
    output = capsys.readouterr()
    assert output.out == ""
    assert output.err == "Error: Stop after checking startup output\n"
    assert cli.parser().parse_args(["analyze", "input.mp4"]).show_settings is False


@pytest.mark.parametrize("options", [["--num-speakers", "0"], ["--output", "input.mp4"]])
def test_show_settings_waits_for_validation(options, monkeypatch, capsys):
    monkeypatch.setattr(cli, "analyze_video", lambda *_: pytest.fail("Analysis must not start"))
    assert cli.main(["analyze", "input.mp4", "--show-settings", *options]) == 1
    output = capsys.readouterr()
    assert output.out == ""
    assert output.err.startswith("Error:")
    assert "# Analysis settings" not in output.err


@pytest.mark.parametrize(
    "document,message",
    [
        ("settings: [", "Cannot load configuration"),
        ("- item\n", "must contain a mapping"),
        ("42", "must contain a mapping"),
        ("null", "must contain a mapping"),
        ("false", "must contain a mapping"),
        ("---\n{}\n---\n{}", "Cannot load configuration"),
        ("resume: true\nresume: false\n", "duplicate key"),
        ("pyannote:\n  clustering: {Fa: 0.1, Fa: 0.2}", "duplicate key"),
        ("1: value", "configuration keys must be strings"),
        ("[one, two]: value", "configuration keys must be strings"),
        ("!unknown {}", "could not determine a constructor"),
        ("input: input.mp4", "Extra inputs are not permitted"),
        ("output: report.md", "Extra inputs are not permitted"),
        ("format: json", "Extra inputs are not permitted"),
        ("show_settings: true", "Extra inputs are not permitted"),
        ("transcripton: true", "Extra inputs are not permitted"),
        ("pyannote: {clustering: {fa: 0.1}}", "Extra inputs are not permitted"),
        ("pyannote: {segmentation: {min_duration_off: .nan}}", "finite number"),
        ("pyannote: {clustering: {threshold: .inf}}", "finite number"),
        ("min_speakers: 3\nmax_speakers: 2", "min_speakers"),
        (
            "diarization_backend: sortformer\npyannote: {clustering: {Fa: 0.1}}",
            "select the pyannote",
        ),
    ],
)
def test_bad_configuration_fails_before_analysis(
    tmp_path, captured_config, capsys, document, message
):
    path = tmp_path / "settings.yaml"
    path.write_text(document)
    assert cli.main(["analyze", "input.mp4", "--config", str(path)]) == 1
    assert captured_config == []
    error = capsys.readouterr().err
    assert message in error and str(path) in error


@pytest.mark.parametrize("kind", ["missing", "directory", "invalid_utf8"])
def test_unreadable_yaml_is_reported(tmp_path, captured_config, capsys, kind):
    path = tmp_path / "settings.yaml"
    if kind == "directory":
        path.mkdir()
    elif kind == "invalid_utf8":
        path.write_bytes(b"\xff")
    assert cli.main(["analyze", "input.mp4", "--config", str(path)]) == 1
    assert captured_config == []
    assert "Cannot load configuration" in capsys.readouterr().err


@pytest.mark.parametrize("field", ["num_speakers", "min_speakers", "max_speakers"])
@pytest.mark.parametrize("value", [0, -1, 1.5, True, "2"])
def test_counts_require_positive_integers(field, value):
    with pytest.raises(ValidationError):
        AnalysisConfig(**{field: value})


@pytest.mark.parametrize(
    "group,name",
    [
        ("clustering", "threshold"),
        ("clustering", "Fa"),
        ("clustering", "Fb"),
        ("segmentation", "min_duration_off"),
    ],
)
@pytest.mark.parametrize("value", [-0.1, float("nan"), float("inf"), True, "0.1"])
def test_tuning_requires_finite_numeric_values(group, name, value):
    with pytest.raises(ValidationError):
        AnalysisConfig(pyannote={group: {name: value}})


@pytest.mark.parametrize("name", ["Fa", "Fb"])
def test_vbx_weights_must_be_positive(name):
    with pytest.raises(ValidationError):
        AnalysisConfig(pyannote={"clustering": {name: 0.0}})


def test_tuning_preserves_defaults_and_does_not_enforce_search_ranges():
    assert AnalysisConfig().pyannote.overrides() == {}
    assert AnalysisConfig(pyannote={"clustering": {"Fa": None}}).pyannote.overrides() == {}
    config = AnalysisConfig(
        pyannote={
            "clustering": {"threshold": 0.0, "Fa": 0.9, "Fb": 20},
            "segmentation": {"min_duration_off": 2.0},
        }
    )
    assert config.pyannote.clustering.Fb == 20.0
    assert config.pyannote.segmentation.min_duration_off == 2.0


@pytest.mark.parametrize(
    "options",
    [
        {"min_speakers": 1},
        {"max_speakers": 4},
        {"pyannote": {"clustering": {"threshold": 0.0}}},
        {"pyannote": {"segmentation": {"min_duration_off": 0.0}}},
    ],
)
@pytest.mark.parametrize("backend", ["sortformer", "nemotron"])
def test_nemo_rejects_overrides_only_when_enabled(options, backend):
    with pytest.raises(ValidationError, match="select the pyannote"):
        AnalysisConfig(diarization_backend=backend, **options)
    assert AnalysisConfig(diarization_backend=backend, diarization=False, **options)
    assert AnalysisConfig(diarization_backend=backend, pyannote={})


def test_help_documents_configuration_and_overrides(capsys):
    with pytest.raises(SystemExit) as exited:
        cli.main(["analyze", "--help"])
    assert exited.value.code == 0
    help_text = capsys.readouterr().out
    for flag in ("--config", "--show-settings", "--min-speakers", "--max-speakers", "--no-resume"):
        assert flag in help_text
    for name in cli.PYANNOTE_PARAMETERS:
        assert f"--{name.replace('_', '-')}" in help_text


@pytest.mark.parametrize("command", [["doctor"], ["prepare-models", "--profile", "mac"]])
def test_yaml_is_only_available_for_analyze(command):
    with pytest.raises(SystemExit) as exited:
        cli.parser().parse_args([*command, "--config", "settings.yaml"])
    assert exited.value.code == 2
