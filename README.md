# Offline Video Analyzer

A Python CLI and API for MP4 transcription, speaker diarization, scene detection,
and visual descriptions. After explicit model preparation, analysis runs locally
without downloads or hosted inference services.

- **Speech:** Whisper large-v3 (default), or Qwen3-ASR-1.7B with Qwen3-ForcedAligner-0.6B.
- **Speakers:** pyannote `speaker-diarization-community-1` (default), NVIDIA Sortformer v2.1, or NVIDIA Nemotron 3 Diarization, retaining overlapping and exclusive turns.
- **Scenes:** PySceneDetect's adaptive detector, using presentation timestamps for variable frame rates.
- **Descriptions:** Molmo2-8B, using a 4-bit MLX conversion on Apple Silicon and original weights on CUDA.
- **Reports:** Markdown, JSON, or an interactive offline HTML viewer. Japanese headings/descriptions and original-language speech by default.

## Install

Use [mise](https://mise.jdx.dev/getting-started.html) and FFmpeg including `ffprobe`.
The project's `mise.toml` pins Python 3.11.15 (default), Python 3.12.15 for the
optional Sortformer and Nemotron workers, and uv 0.12.11; `mise.lock` records
download URLs and checksums for Apple Silicon and Linux x86_64. Use mise
2026.9.16 or newer, and run these commands from the repository.

Apple Silicon macOS 15 or newer:

```bash
brew install mise ffmpeg
mise trust
mise install --locked
mise run setup
```

Linux x86_64 with an NVIDIA GPU:

```bash
sudo apt-get install ffmpeg
# Install mise first using its official installation instructions.
mise trust
mise install --locked
mise run setup
```

`mise run setup` chooses the Mac or CUDA installation for the current platform
and checks FFmpeg before installing dependencies. Explicit tasks are also available:
`mise run setup:mac` and `mise run setup:cuda`. For CLI development and tests without
large model workers, use `mise run setup:dev`.

mise manages Python and uv; uv installs the locked CLI and worker environments.
FFmpeg remains a system package. Models are downloaded separately and explicitly
with `prepare-models`. All tasks work without activating a shell environment.
With mise shell activation enabled, the project's existing `.venv` is activated
automatically; `source .venv/bin/activate` also permits direct `video-analyzer` commands.

The CUDA lock uses PyTorch 2.8 and CUDA 12.8 packages; install a compatible NVIDIA
driver separately. CUDA inference needs room for Molmo2 weights and activations;
24 GB VRAM is a starting target, not a guarantee for every configuration. The Mac
backend targets the 24 GB M2 using quantized weights. Models run sequentially.

The CLI, PyTorch worker, and MLX worker have separate dependency locks. Qwen
requires Transformers 4.57.6; MLX uses 5.14.0. Do not combine their environments.
The hash-locked MLX-VLM 0.7.4 release includes Molmo2 multi-image processing and
image-feature overflow fixes. MLX also includes PyTorch for Molmo2's upstream image
processor. Setup uses mise's selected Python and disables implicit Python downloads
by uv.

## Prepare models once while online

Accept access on [pyannote's model page](https://huggingface.co/pyannote/speaker-diarization-community-1),
then authenticate with a Hugging Face read token:

```bash
mise exec -- hf auth login
mise run prepare-models --profile mac --asr both
mise run doctor --asr both --verify-hashes
```

An existing `hf auth login` is reused; there is no need to log in again.

Use `--profile cuda` on Linux. Omit `--asr both` to prepare Whisper, diarization,
and Molmo2 only; `--asr qwen` selects Qwen and its aligner instead.

Preparation uses `hf` and its saved login (or `HF_TOKEN`). It downloads tokenizers,
processors, custom model code, weight shards, and pyannote's nested assets.
`.models/manifest.json` records immutable revisions and file checksums. Redundant
FP32 Whisper variants are excluded. Reserve approximately 20–25 GB of disk for
the Mac installation with both ASR backends, plus input videos and working audio.

`doctor` is offline. It checks FFmpeg, worker environments, accelerators, and model
completeness; `--verify-hashes` also reads weights to verify SHA-256 checksums. It
does not run inference. macOS processes need access to Metal for MLX inference.

### Optional Sortformer v2.1 diarization

After the regular setup, install the separate NVIDIA reference runtime and
explicitly prepare its checkpoint:

```bash
mise install --locked
mise run setup:sortformer
mise run prepare-models --profile mac --diarization-backend sortformer
mise run doctor --profile mac --diarization-backend sortformer --verify-hashes
mise run analyze input.mp4 --diarization-backend sortformer
```

Use `--profile cuda` for Linux CUDA. For Linux CPU inference, prepare with
`--profile cuda`, then use `--profile cpu` with `doctor` and `analyze`. The optional
`.runtime/sortformer` environment uses Python **3.12.15**, `nemo-toolkit[asr]==2.7.3`,
and PyTorch/torchaudio **2.8.0** with separate hash locks for Apple Silicon and
Linux x86_64. Existing setup tasks and default pyannote diagnostics do not install
or require NeMo. Python 3.11 remains the default for the CLI and existing workers.

The checkpoint is NVIDIA's pretrained
[`diar_streaming_sortformer_4spk-v2.1`](https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2.1),
with a **four-speaker maximum**. It is not a Japanese fine-tuned variant, and better
Japanese accuracy is not assumed. It uses CPU on Mac and CPU/CUDA on Linux;
Apple GPU acceleration and live microphone input are not supported. Omit
`--num-speakers` when Sortformer diarization is enabled: an explicit count is
rejected, with guidance to select pyannote if a known count is needed.

Preparation downloads the exact `.nemo` file (approximately 450 MiB) and records
its resolved revision, size, and SHA-256 under the separate `sortformer` manifest
entry. Existing pyannote (`diarization`) and ASR assets are retained. Use
`--diarization-backend both` on `prepare-models` or `doctor` to prepare or check
both diarizers; `analyze` accepts one backend. Sortformer itself does not require
Hugging Face authentication. Pyannote's access agreement is needed only when
preparing pyannote. Preparation and doctor still include the chosen ASR and
profile's vision models.

Sortformer consumes the same processed 16 kHz mono audio as transcription. A
single recording retains streaming speaker state throughout inference, with
batch size one and no data-loader subprocesses. NVIDIA's recording preset uses
chunk length 340, right context 40, FIFO length 40, speaker cache length 188, and
cache update period 300 (80 ms frames). NeMo reports an effective cache update
period of 340 for this preset because it updates at completed chunk boundaries.
The configured and effective settings are recorded in provenance.

The regular timeline uses NeMo's default postprocessing and retains overlap. The
exclusive timeline selects the highest-probability **active** speaker in each
80 ms frame, using the lower model speaker index for ties. It is restricted to
regular speech regions, preserves silence, merges adjacent identical labels,
and clips timestamps to the recording duration. Reports retain the existing
schema, canonical speaker IDs, and transcript assignment behavior. Backend,
checkpoint revision, runtime versions, inference settings, and exclusive-timeline
derivation are recorded in JSON provenance. Switching backends or checkpoints
uses a separate analysis cache, including transcription results.

`--no-diarization` and videos without audio require neither diarizer's assets nor
its worker environment, regardless of the selected backend.

### Optional Nemotron 3 diarization

Install the separate NVIDIA runtime and explicitly prepare its checkpoint:

```bash
mise install --locked
mise run setup:nemotron
mise run prepare-models --profile mac --diarization-backend nemotron
mise run doctor --profile mac --diarization-backend nemotron --verify-hashes
mise run analyze input.mp4 --diarization-backend nemotron
# Or load the dedicated configuration:
mise run analyze input.mp4 --config examples/analysis_nemotron.yaml
```

Use `--profile cuda` on Linux CUDA. For Linux CPU inference, prepare with
`--profile cuda`, then select `--profile cpu` for `doctor` and `analyze`.
Nemotron uses CPU on Apple Silicon and CPU/CUDA on Linux, with a maximum of
**eight speakers**. Omit speaker-count constraints and pyannote tuning when it is
enabled. Python 3.11 remains the CLI default; this optional environment uses
Python **3.12.15**, PyTorch/torchaudio **2.8.0**, and a NeMo source build reporting
version **3.1.0**. It has its own hash locks and does not change the existing workers.

The published NeMo 3.0.0 package predates the required high-resolution model
support. The runtime pins NVIDIA's source archive at revision
`1688cc3d6a9ade854f544987810c53f605dc86fc`, with its archive hash recorded in
both platform locks. Offline diagnostics verify the source revision and required
NeMo APIs. Normal setup and pyannote/Sortformer diagnostics do not require this
optional environment.

Preparation downloads NVIDIA's original
[`Nemotron-3-Diarization.nemo`](https://huggingface.co/nvidia/Nemotron-3-Diarization)
checkpoint and README into a separate `nemotron` manifest entry, with the resolved
model revision, size, and SHA-256. Preparing Nemotron retains previously prepared
models. The existing `--diarization-backend both` continues to select pyannote and
Sortformer v2.1; prepare and check Nemotron separately using `nemotron`.

One complete recording uses NeMo's streaming state with batch size one and no
data-loader subprocesses. Its recording preset uses chunk length 340, right
context 40, FIFO length 40, speaker cache length **264**, and cache update period
300, measured in **80 ms encoder frames**. Returned speaker probabilities use
**10 ms output frames**, including progress timing and exclusive-timeline
derivation. Configured/effective cache settings, output stride, runtime versions,
and the pinned source revision are recorded in provenance. Changes to that source
revision or adapter settings invalidate Nemotron analysis checkpoints.

The regular timeline preserves overlap using NeMo's default postprocessing. The
exclusive timeline chooses the highest-probability active speaker in each 10 ms
frame, breaking ties by lower model speaker index, preserving silence and partial
frames, and clipping to the original duration. Reports and transcript assignment
use the same schema as the existing backends. Disabled diarization and videos
without audio require neither the Nemotron checkpoint nor its runtime.

## Analyze

```bash
mise run analyze input.mp4
mise run analyze input.mp4 --config examples/analysis.yaml
mise run analyze input.mp4 --config examples/analysis.yaml --show-settings
mise run analyze input.mp4 --asr qwen --format json -o report.json
mise run analyze input.mp4 --format html --audio-normalization dynamic --noise-reduction
mise run analyze input.mp4 --resume
mise run analyze input.mp4 --language ja --num-speakers 2
mise run analyze input.mp4 --report-language en --audio-stream 1
mise run analyze input.mp4 --audio-normalization dynamic --noise-reduction
mise run analyze input.mp4 --no-diarization --no-scene-description
mise run analyze input.mp4 --no-transcription --no-diarization
mise run analyze --help
```

Each CLI command accepts `-h` or `--help` to show its options, defaults, and examples
without running the command:

```bash
mise run prepare-models --help
mise run doctor --help
mise run analyze --help
```

With the virtual environment activated, use `video-analyzer --help` for the
command list, or `video-analyzer COMMAND --help` for the same command-specific help.

Analysis always saves a JSON result. The default produces `input.analysis.json`
and `input.analysis.md` beside the video. `--format html` produces
`input.analysis.json` and `input.analysis.html`; `--format json` writes only
`input.analysis.json`.

`-o` sets the selected report's path. For Markdown or HTML, the JSON file is saved
beside that report with its extension replaced by `.json`: for example,
`--format html -o reports/review.html` also writes `reports/review.json`. A
Markdown/HTML output path must not end in `.json` or refer to its JSON companion.
Progress and the companion JSON path go to stderr; stdout contains only the
completed selected report path.

Files are written atomically after analysis succeeds. An analysis failure
preserves existing outputs. JSON is saved before Markdown/HTML generation, so a
later export failure preserves the completed analysis for reuse while leaving an
existing selected report intact.

| Option | Default | Behavior |
| --- | --- | --- |
| `--format` | `markdown` | `markdown`, `json`, or an interactive `html` report; JSON is always saved |
| `--config` | none | Load YAML analysis settings; explicit CLI options override individual settings |
| `--show-settings` | disabled | Print all merged analysis settings as YAML to stderr before starting analysis |
| `--resume` / `--no-resume` | disabled | Reuse matching checkpoints, or disable reuse even when enabled in YAML |
| `--transcription` / `--no-transcription` | enabled | Enable or disable speech transcription |
| `--diarization` / `--no-diarization` | enabled | Enable or disable speaker identification |
| `--scene-description` / `--no-scene-description` | enabled | Enable or disable visual descriptions; scene detection always runs |
| `--asr` | `whisper` | Whisper or Qwen |
| `--diarization-backend` | `pyannote` | Community-1, pretrained Sortformer v2.1 (maximum four speakers), or Nemotron 3 (maximum eight speakers) |
| `--profile` | `auto` | MLX on Apple Silicon; CUDA elsewhere. Explicit `cpu` is available for large-memory hosts. |
| `--language` | automatic | Spoken language hint, such as `ja` or `en`; no translation |
| `--report-language` | `ja` | Japanese or English headings/descriptions |
| `--num-speakers` | automatic | Optional known speaker count for pyannote; overrides min/max bounds; rejected with enabled Sortformer or Nemotron diarization |
| `--min-speakers` / `--max-speakers` | automatic | Optional positive lower/upper speaker-count bounds for pyannote |
| `--pyannote-clustering-threshold` | model default | Nonnegative threshold for initial clustering before VBx refinement |
| `--pyannote-clustering-fa` | model default | Positive VBx acoustic-evidence scale (`Fa`) |
| `--pyannote-clustering-fb` | model default | Positive VBx speaker-regularization coefficient (`Fb`) |
| `--pyannote-segmentation-min-duration-off` | model default | Nonnegative pause duration in seconds; fills shorter gaps for the same speaker |
| `--audio-stream` | `0` | Zero-based audio stream ordinal, not container stream index |
| `--audio-normalization` | `off` | `off`, `peak` for one gain adjustment across the recording, or `dynamic` to balance quiet and loud passages |
| `--noise-reduction` / `--no-noise-reduction` | disabled | Reduce rumble and background noise before normalization and audio analysis |
| `--scene-threshold` | `3.0` | Adaptive cut detection sensitivity |
| `--min-scene-seconds` | `0.5` | Minimum spacing between cuts |
| `--window-seconds` | `30` | Description window length, maximum 30 seconds |
| `--frames-per-window` | `8` | Maximum eight chronological frames per window |
| `--audio-device` | `auto` | CPU, MPS, or CUDA; automatic MPS retries unsupported operations and out-of-memory errors on CPU |
| `--diarization-device` | `auto` | CPU on Mac, CUDA on the CUDA profile |
| `--vision-device` | `auto` | MLX/Metal on Mac; CPU/CUDA overrides for PyTorch vision |
| `--model-dir` | `.models` | Prepared models |
| `--runtime-dir` | `.runtime` | Parent of `torch`, `mlx`, and optional `sortformer`/`nemotron` environments |
| `--work-dir` | `.video-analyzer` beside input | Checkpoints and temporary audio |
| `--worker-timeout` | `3600` | Seconds allowed without worker progress; diarization batch progress resets the timer |

Paths are relative to the current directory. Outside this repository, supply
absolute model and runtime directory paths.

The three features are independent. Use `--no-diarization --no-scene-description`
for transcription without speaker labels, or `--no-transcription --no-diarization`
for visual-only analysis. Disabled features do not require their model assets or
worker environments. Disabling all three produces media metadata and detected
scene boundaries without requiring any prepared models or model workers.
`prepare-models` and `doctor` prepare/check the selected diarization and ASR
backends plus the profile's vision model.
Markdown reports list the enabled features and omit disabled content sections;
JSON retains its existing fields, with empty lists for disabled outputs.

### YAML configuration and pyannote tuning

Use `--config FILE` on `analyze` to load any `AnalysisConfig` settings by their
snake_case field names. The input video, `--output`, and `--format` remain CLI
arguments. `prepare-models` and `doctor` do not load analysis configuration files.
Start with [examples/analysis.yaml](examples/analysis.yaml):

```yaml
language: ja
diarization_backend: pyannote
min_speakers: 2
max_speakers: 4
audio_normalization: off
pyannote:
  clustering:
    threshold: 0.65
    Fa: 0.07
    Fb: 0.8
  segmentation:
    min_duration_off: 0.1
```

Precedence is **explicit CLI options, then YAML settings, then application
defaults**. Nested settings merge by individual field: overriding the threshold
retains `Fa`, `Fb`, and the pause duration from YAML. Explicit flags override YAML
even when they select an application default, such as `--audio-normalization off`
or `--no-noise-reduction`. Use `--no-resume` to disable YAML-enabled checkpoint
reuse.

Add `--show-settings` to print the complete merged configuration as YAML to stderr
after validation and before analysis starts. It is **disabled by default** and is
a CLI-only option. The output includes all analysis defaults, nested pyannote
settings, and paths as strings. Configured `auto` and `null` values remain intact;
a comment explains that null pyannote tuning values use the loaded model's defaults.
Analysis then continues normally, stdout still contains only the completed report
path, and the flag does not affect provenance or checkpoint reuse.

```bash
mise run analyze input.mp4 --config examples/analysis.yaml --pyannote-clustering-threshold 0.6
mise run analyze input.mp4 --config examples/analysis.yaml --num-speakers 2 --no-resume
mise run analyze input.mp4 --min-speakers 2 --max-speakers 4 --pyannote-clustering-fa 0.1
```

Relative paths in YAML, including `model_dir`, `runtime_dir`, and `work_dir`, are
relative to the **current working directory**. Use `true` and `false` for boolean
values; unquoted `off` remains a string. Empty files use application defaults.
Unreadable files, malformed YAML, duplicate or unknown keys, invalid value types,
and invalid settings are reported before analysis starts.

Speaker counts must be positive integers. When `num_speakers` is supplied it
takes precedence over both bounds; otherwise `min_speakers` must not exceed
`max_speakers`. All tuning values must be finite. `Fa` and `Fb` must be positive;
the threshold and pause duration may be zero. Validation does not restrict
values to pyannote's suggested optimization search ranges. Count constraints and
pyannote tuning are rejected when Sortformer or Nemotron diarization is enabled, but can
remain in a configuration when diarization is disabled.

Tuning values in the example are starting points for experiments, not a preset
proven to improve accuracy. Omitted or `null` tuning values preserve the loaded
model's defaults. Overrides are applied in memory without editing downloaded
models. The configured values appear in JSON `provenance.configuration` and are
included in checkpoint identity. Changing a setting reruns analysis; equivalent
YAML and CLI settings reuse the same checkpoints with `--resume`, regardless of
the configuration filename.

### Audio preprocessing

Normalization and noise reduction are optional and independent. Both default to
off. They run locally through FFmpeg before speaker identification and
transcription, using the same processed audio for Whisper or Qwen and diarization.
They preserve the video-relative timing and produce mono 16 kHz, 16-bit PCM.
No additional models or dependencies are needed. When there is no audio track,
or both transcription and diarization are disabled, preprocessing is skipped.

```bash
# One gain adjustment for the entire recording, including recordings lasting hours.
mise run analyze input.mp4 --audio-normalization peak

# Gradually balance quieter and louder passages, optionally reducing noise first.
mise run analyze input.mp4 --audio-normalization dynamic
mise run analyze input.mp4 --audio-normalization dynamic --noise-reduction

# Reduce noise without normalizing, or explicitly disable both features.
mise run analyze input.mp4 --audio-normalization off --noise-reduction
mise run analyze input.mp4 --audio-normalization off --no-noise-reduction
```

`peak` makes two passes: it measures the highest sample peak after any noise
reduction, then applies one constant gain to reach -3 dBFS. Quiet and loud
passages retain their relative levels, and a single loud moment limits the gain
for the whole recording. Digital silence remains silent.

`dynamic` uses FFmpeg's
[`dynaudnorm`](https://ffmpeg.org/ffmpeg-filters.html#dynaudnorm) with 500 ms frames
and a 31-frame smoothing window (about 15.5 seconds). It gradually balances
longer quiet and loud passages, targeting a -3 dBFS peak with amplification
limited to 20 dB. A -60 dBFS threshold limits amplification of very quiet
background noise; additional compression and RMS targeting are disabled.
Smoothing, the gain limit, and the threshold mean every passage will not reach
the target. Processing uses one streaming pass with bounded buffering, retaining
filter state throughout the recording instead of restarting at ASR chunk boundaries.

Noise reduction applies an 80 Hz high-pass filter followed by conservative FFT
denoising (`afftdn=nr=10:nf=-50:tn=1:gs=5`), with a 10 dB reduction setting,
automatic noise-floor tracking, and gain smoothing. It runs before normalization;
its delay is compensated and the audio tail is preserved. All processing
parameters are fixed in this version.

As the [noisy-audio guide](https://vocova.app/blog/transcribe-noisy-audio)
recommends, compare transcription of a representative sample with and without
denoising: cleaner-sounding audio does not always improve recognition. The
selected mode and noise-reduction flag are recorded in JSON
`provenance.configuration`. Markdown reports show the configured normalization
mode and noise-reduction enabled/disabled status in the "Models and runtime"
section, localized to the report language. If enabled preprocessing was skipped,
the report explains whether the input has no audio track or both transcription
and diarization are disabled. Changing either uses a separate checkpoint directory;
`--resume` reuses work only for matching settings.

### Interactive HTML viewer

Create an HTML report and its JSON result during analysis, or export a saved JSON
result without loading models or rerunning inference:

```bash
mise run analyze input.mp4 --format html --audio-normalization dynamic --noise-reduction
mise run export-html input.analysis.json
mise run export-html report.json --video /new/location/input.mp4 -o reports/viewer.html
# With the environment activated:
video-analyzer export-html report.json --help
```

Open the HTML directly in a desktop browser. It embeds its scripts, styles, and
analysis data, with no server or internet connection required. The source MP4 is
referenced through a relative link; it is not copied or embedded. Use a
browser-decodable MP4 (typically H.264/AAC). If the video moves, use **Locate
original video** in the viewer, or re-export with `--video`. Export verifies the
recorded input hash when available and otherwise checks media metadata.

The timeline shows one colored lane per speaker and retains overlapping speech.
Hover over a speech block for that speaker's transcript within five seconds of
the pointer time, clipped to the block. Click the video itself to play or pause.
Click anywhere in a timeline lane or its ruler to seek, play, and scroll the
transcript pane to the matching passage. Overlapping speech prefers the clicked
speaker's passage; a click in a silent gap shows the next passage, or the last
passage after speech ends. Transcript and scene entries also jump to their timestamps.
Use **Zoom** and horizontal scrolling for detail, or **Fit** for the full recording.
Focused speech blocks support Enter, and the focused timeline supports Space to
play/pause and arrow keys to seek five seconds. The player controls are keyboard
accessible. Scene descriptions, the transcript, provenance, and warnings are
included, with Japanese or English labels matching the result.

**Original** is selected initially and uses the track chosen for analysis. A
single-track video's original audio plays directly from the MP4. For multi-track
MP4s, export creates an unprocessed PCM16 WAV retaining the selected track's sample
rate and channel count. This makes track selection independent of browser support.
**Processed** appears when normalization or denoising was configured and an audio
track exists; its label names the settings. Its aligned 16 kHz mono PCM WAV is
regenerated by FFmpeg using the recorded settings, without depending on cached
analysis files. Configured playback processing is available even when audio
analysis features were disabled. Older JSON without preprocessing settings uses
the original audio only.

Generated audio lives in a companion `<report-stem>.media/` folder. Keep this
folder beside the HTML, and retain the source video's relative location when
moving the report. Audio generation completes before the HTML is atomically
published; failures preserve an existing report and its audio. Repeated exports
retain prior audio generations for reports that are already open. To reclaim
space, close those reports, delete the companion folder, and re-export. A report
without any required audio sidecars does not create a media folder.

Switching audio preserves the playback position, speed, volume, mute state, and
whether playback is paused. This is a read-only viewer; it does not edit the video,
transcript, or speaker assignments, and does not transcode unsupported video codecs.

## Python API

```python
from pathlib import Path
from video_analyzer import AnalysisConfig, analyze_video, render_json, render_markdown, write_html

result = analyze_video(Path("input.mp4"), AnalysisConfig(resume=True))
Path("report.md").write_text(render_markdown(result), encoding="utf-8")
Path("report.json").write_text(render_json(result), encoding="utf-8")
write_html(result, "report.html")

# Transcription only, without speaker labels or visual descriptions.
result = analyze_video(
    Path("input.mp4"),
    AnalysisConfig(transcription=True, diarization=False, scene_description=False),
)

# Dynamically balance volume and reduce noise before audio analysis.
result = analyze_video(
    Path("input.mp4"),
    AnalysisConfig(audio_normalization="dynamic", noise_reduction=True),
)

# Select pretrained Sortformer v2.1 after preparing its optional runtime and model.
result = analyze_video(Path("input.mp4"), AnalysisConfig(diarization_backend="sortformer"))

# Set pyannote speaker bounds and override only selected model parameters.
result = analyze_video(
    Path("input.mp4"),
    AnalysisConfig(
        min_speakers=2,
        max_speakers=4,
        pyannote={
            "clustering": {"threshold": 0.65, "Fa": 0.07, "Fb": 0.8},
            "segmentation": {"min_duration_off": 0.1},
        },
    ),
)
```

The API returns a Pydantic `AnalysisResult` without writing a report. Importing it
does not load PyTorch or MLX. Model stages use isolated interpreters communicating
through versioned JSON Lines over stdin/stdout.

`write_html(result, output, *, video_path=None)` returns the absolute output
`Path`. It validates the local source, creates any audio sidecars, and publishes
the HTML. Supply `video_path` when the original MP4 has moved. It leaves the result
and the JSON schema unchanged.

JSON schema `1.0` includes media metadata, model/runtime provenance, anonymous
speakers, regular/exclusive turns, timed transcript units, scenes, observations,
and warnings. Times are seconds relative to video start. Scenes reference
canonical transcript IDs; a unit crossing a cut is represented once. Overlapping
diarization is retained even though text receives one speaker assignment.
`provenance.stage_metrics` records inference time and peak memory measurements
from completed work, including reused checkpoints. Runtime load measurements are
separate; `provenance.elapsed_seconds` measures the current API invocation. MLX
device allocation and process RSS overlap on unified memory and must not be added.

## Processing and recovery

Audio extraction preserves stream offsets, padding delayed audio and trimming
audio before video start. Full-recording diarization preserves speaker identities.
ASR separates speech regions at pauses, then uses 20-second core windows with up
to five seconds of context on each side within a region. Language detection runs
again for each region, and 0.3-second padding protects words near VAD boundaries;
timestamp ownership and overlap reconciliation remove boundary duplicates.
With diarization disabled, transcription instead covers the full recording in
20-second core windows with up to five seconds of context on each side, and
transcript speaker IDs remain unset. Scene detection still runs when visual
descriptions are disabled, retaining scene timestamps and transcript references.
No-audio videos produce visual reports. Silence produces an empty transcript.

Diarization reports its device, audio duration, and progress on stderr: pyannote
reports segmentation, speaker embeddings, and clustering; Sortformer reports
completed streaming steps. The worker timeout measures time
without progress, so a long recording can take more than one hour in total while
batches continue completing. Model loading, ASR chunks, visual windows, and
clustering must each finish within the timeout when they provide no intermediate
progress. A timeout reports the last progress message; a worker exit reports its
exit code or signal. After a timeout, retry with `--resume` and, if necessary,
`--worker-timeout 7200`.

On macOS, PyAV and OpenCV can emit duplicate `AVFFrameReceiver`/`AVFAudioReceiver`
warnings during scene detection. These libraries run in the parent process,
separately from the diarization worker. Pyannote can also emit a pooling `std()`
warning for very short speaker regions; this warning alone does not mean the
worker failed. Use the subsequent timeout or exit diagnostic to identify the
failure rather than removing bundled libraries.

With `--audio-device auto`, an unsupported MPS operation or MPS out-of-memory
error releases the accelerator model and retries on CPU. If this happens during
transcription, the failed chunk is retried and subsequent chunks stay on CPU;
the report includes a warning. CPU transcription can be slower. To use CPU from
the start, pass `--audio-device cpu`. An explicit `--audio-device mps` propagates
errors without switching devices. The MPS memory limit remains enabled.

Long scenes retain their detected boundaries and receive timestamped observations
for consecutive windows. Each window uses up to eight actual frames resized to a
maximum 768-pixel long edge. This is sampled multi-image description: fast actions
between samples may be missed. Visual descriptions are independent of the speech.

Completed stages, ASR chunks, and visual windows are checkpointed atomically.
`--resume` reuses matching input content, settings, model revisions, and application
version and implementation fingerprint. Different inputs/settings use separate
cache directories. A process lock protects each run. Failed runs preserve completed
checkpoints and exit nonzero.
Omitting `--resume` starts fresh and clears older checkpoints for that same run
identity before doing new work.

Qwen's aligner supports Chinese, English, Cantonese, French, German, Italian,
Japanese, Korean, Portuguese, Russian, and Spanish. Unsupported or mixed language
labels that cannot be aligned produce an error recommending Whisper. Model text,
timestamps, speaker labels, and visual descriptions can be inaccurate. Quantized
and original Molmo2 weights may produce different descriptions.

## Offline behavior

Only `prepare-models` contacts Hugging Face. Analysis checks local assets before
inference, enables library offline flags, disables telemetry, and blocks Python
network sockets in workers. Tokens are not passed to workers or written into
reports/manifests. Molmo2 custom code is loaded from its prepared snapshot.

Socket guards are defense in depth, not an OS sandbox for native code. Independent
offline verification should also disable networking or use an OS network sandbox.
An air-gapped installation bundle is outside this version's scope.

## Tests

```bash
mise run check             # lint, formatting, and normal tests
mise run test -q           # tests alone; extra arguments are forwarded
mise run build            # wheel and source distribution
mise tasks                # list available tasks
```

The normal suite generates MP4 fixtures and simulates model workers. It checks
cuts, variable frame rates, delayed audio, alignment, recovery, reports, missing
models, network guards, and worker failures without downloading weights.

HTML browser tests are separate from the normal suite. Explicitly install their
development-only browser binary, then run them against generated local files:

```bash
mise run setup:dev
mise run setup:browser-tests
mise run test:browser
# Optional WebKit check on macOS:
PLAYWRIGHT_BROWSERS_PATH=.cache/playwright .venv/bin/python -m playwright install webkit
VIDEO_ANALYZER_TEST_BROWSER=webkit mise run test:browser
```

On Linux, install the browser's system dependencies with
`PLAYWRIGHT_BROWSERS_PATH=.cache/playwright .venv/bin/python -m playwright install --with-deps chromium`.
CI runs Chromium tests on Linux. The tests exercise direct `file://` playback,
overlap, popups, zoomed seeking, track switching, synchronization, missing media,
and safe rendering without requesting remote resources. WebKit testing checks
the engine used by Safari; it does not replace checking the Safari application.

Real-model tests require a local video with speech and visible content:

```bash
VIDEO_ANALYZER_MODEL_TEST_VIDEO=/absolute/path/two-speakers.mp4 \
  mise run test:models
```

These opt-in tests run both speech backends, diarization, and Molmo2 with an empty
general Hugging Face cache, check resume without workers, and exercise eight
768-pixel frames. Set `VIDEO_ANALYZER_TEST_PROFILE=cuda` on Linux. Optional
`VIDEO_ANALYZER_TEST_LANGUAGES=japanese,english` asserts detected languages in a
multilingual fixture.
To exercise both prepared diarizers, set `VIDEO_ANALYZER_TEST_DIARIZATION=both`
(`sortformer` selects only that backend; the default is `pyannote`). Sortformer
acceptance also covers empty, very short, silent, and multiple-chunk recordings.
For independent offline acceptance, run the tests inside an OS network sandbox
as shown in the validation notes. Linux CUDA installation and inference still
need validation on an NVIDIA host.
See [validation notes](docs/validation.md) for hardware tests actually performed.

Upstream libraries/models retain their licenses. Community-1 is CC-BY-4.0; see
the [Molmo2](https://huggingface.co/allenai/Molmo2-8B) and
[Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR) repositories for their model licenses.
