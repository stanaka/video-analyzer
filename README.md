# Offline Video Analyzer

A Python CLI and API for MP4 transcription, speaker diarization, scene detection,
and visual descriptions. After explicit model preparation, analysis runs locally
without downloads or hosted inference services.

- **Speech:** Whisper large-v3 (default), or Qwen3-ASR-1.7B with Qwen3-ForcedAligner-0.6B.
- **Speakers:** pyannote `speaker-diarization-community-1`, retaining overlapping and exclusive turns.
- **Scenes:** PySceneDetect's adaptive detector, using presentation timestamps for variable frame rates.
- **Descriptions:** Molmo2-8B, using a 4-bit MLX conversion on Apple Silicon and original weights on CUDA.
- **Reports:** Markdown or JSON. Japanese headings/descriptions and original-language speech by default.

## Install

Use [mise](https://mise.jdx.dev/getting-started.html) and FFmpeg including `ffprobe`.
The project's `mise.toml` pins Python 3.11.15 and uv 0.12.11; `mise.lock` records
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

## Analyze

```bash
mise run analyze input.mp4
mise run analyze input.mp4 --asr qwen --format json -o report.json
mise run analyze input.mp4 --resume
mise run analyze input.mp4 --language ja --num-speakers 2
mise run analyze input.mp4 --report-language en --audio-stream 1
mise run analyze -- --help
```

The default output is `input.analysis.md` beside the video, or
`input.analysis.json` with `--format json`. Progress goes to stderr; stdout
contains the completed report path. Final reports are written atomically only
after all stages succeed. Existing reports are preserved on failure.

| Option | Default | Behavior |
| --- | --- | --- |
| `--asr` | `whisper` | Whisper or Qwen |
| `--profile` | `auto` | MLX on Apple Silicon; CUDA elsewhere. Explicit `cpu` is available for large-memory hosts. |
| `--language` | automatic | Spoken language hint, such as `ja` or `en`; no translation |
| `--report-language` | `ja` | Japanese or English headings/descriptions |
| `--num-speakers` | automatic | Optional known speaker count |
| `--audio-stream` | `0` | Zero-based audio stream ordinal, not container stream index |
| `--scene-threshold` | `3.0` | Adaptive cut detection sensitivity |
| `--min-scene-seconds` | `0.5` | Minimum spacing between cuts |
| `--window-seconds` | `30` | Description window length, maximum 30 seconds |
| `--frames-per-window` | `8` | Maximum eight chronological frames per window |
| `--audio-device` | `auto` | CPU, MPS, or CUDA; automatic MPS retries unsupported operations and out-of-memory errors on CPU |
| `--diarization-device` | `auto` | CPU on Mac, CUDA on the CUDA profile |
| `--vision-device` | `auto` | MLX/Metal on Mac; CPU/CUDA overrides for PyTorch vision |
| `--model-dir` | `.models` | Prepared models |
| `--runtime-dir` | `.runtime` | Parent of `torch` and `mlx` environments |
| `--work-dir` | `.video-analyzer` beside input | Checkpoints and temporary audio |
| `--worker-timeout` | `3600` | Seconds allowed without worker progress; diarization batch progress resets the timer |

Paths are relative to the current directory. Outside this repository, supply
absolute model and runtime directory paths.

## Python API

```python
from pathlib import Path
from video_analyzer import AnalysisConfig, analyze_video, render_json, render_markdown

result = analyze_video(Path("input.mp4"), AnalysisConfig(resume=True))
Path("report.md").write_text(render_markdown(result), encoding="utf-8")
Path("report.json").write_text(render_json(result), encoding="utf-8")
```

The API returns a Pydantic `AnalysisResult` without writing a report. Importing it
does not load PyTorch or MLX. Model stages use isolated interpreters communicating
through versioned JSON Lines over stdin/stdout.

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
No-audio videos produce visual reports. Silence produces an empty transcript.

Diarization reports its device, audio duration, and progress through segmentation,
speaker embeddings, and clustering on stderr. The worker timeout measures time
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
See [validation notes](docs/validation.md) for hardware tests actually performed.

Upstream libraries/models retain their licenses. Community-1 is CC-BY-4.0; see
the [Molmo2](https://huggingface.co/allenai/Molmo2-8B) and
[Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR) repositories for their model licenses.
