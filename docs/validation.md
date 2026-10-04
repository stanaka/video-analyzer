# Validation record

Executed on 2026-10-03 on an Apple M2 Mac with 24 GB unified memory, macOS
27.0.1, and Python 3.11.15. These are smoke tests and measured examples, not
accuracy benchmarks or throughput guarantees.

## Environment and model preparation

Setup was subsequently migrated to mise 2026.9.16 with Python 3.11.15 and uv
0.12.11. `mise run setup` completed on this Mac, and all three environments now
use the mise-managed Python interpreter. `mise run check` passed all 30 normal
tests plus lint and formatting. `mise run doctor --asr both --verify-hashes`
reported no errors and confirmed MPS/Metal acceleration. Tool download URLs and
checksums were locked for macOS arm64 and Linux x86_64; Linux installation was
not executed. These setup checks did not rerun the earlier real-model smoke tests.

Both worker environments were installed from their platform hash locks. The audio
worker used PyTorch 2.8.0, Transformers 4.57.6, pyannote.audio 4.0.4, and qwen-asr
0.0.6. The vision worker used MLX 0.32.3, MLX-VLM 0.7.4, and Transformers 5.14.0.
The installed MLX-VLM code includes multi-image handling and image-feature clipping
to prevent numerical overflow in Molmo2.

`prepare-models --profile mac --asr both` was executed successfully using the
user's existing `hf` login. It was rerun against the prepared snapshots to verify
the command's repeatability. No authentication token was included in a command,
report, checkpoint, or manifest. Prepared model assets occupy approximately 15 GiB.
The final `doctor --asr both --verify-hashes` check passed with OS networking
disabled, reporting MPS and Metal available and no missing or corrupt assets.

| Model | Immutable revision |
| --- | --- |
| MLX Molmo2-8B-4bit | `4fcbe926577641af12aa5086cf5c292db9dd9730` |
| pyannote community-1 | `3533c8cf8e369892e6b79ff1bf80f7b0286a54ee` |
| Whisper large-v3 | `06f233fe06e710322aca913c1bc4249a0d71fce1` |
| Qwen3-ASR-1.7B | `7278e1e70fe206f11671096ffdd38061171dd6e5` |
| Qwen3-ForcedAligner-0.6B | `c7cbfc2048c462b0d63a45797104fc9db3ad62b7` |

## Automated checks

**30 automated tests passed** in the normal suite. Separately, **three real-model
Mac tests passed**: the Whisper pipeline, the Qwen pipeline, and the maximum-frame
Molmo2 test. The two pipeline tests also exercised checkpoint-only resume.

The normal test suite covers generated MP4s with known cuts, no cuts, no audio,
silent audio, delayed audio, and variable frame rates. It also covers overlapping
speaker turns, unmatched words, zero-duration aligned tokens, Japanese spacing
and punctuation, scene-boundary ownership, and bounded chunking of a one-hour
timeline. The one-hour test exercises chunk scheduling, not real-model inference.

Recovery tests interrupt ASR chunks and visual windows, then verify that only
incomplete work is repeated. Additional checks cover fresh-run cache isolation,
input/model/configuration invalidation, atomic reports, corrupt or missing assets,
worker errors/timeouts, Python socket blocking, and automatic versus explicit MPS
fallback behavior. JSON round-trips preserve the canonical result.

Ruff lint and formatting checks passed. Both a wheel and a source distribution
were built offline. Installation scripts passed shell syntax checks. The included
GitHub Actions workflow has not been executed on remote runners in this session.

## Real offline pipeline tests

The fixture is a 16.3-second, 320×240 MP4 with a white background and a red square.
It includes Japanese speech synthesized with the macOS Kyoko voice, followed by
English speech synthesized with Daniel, separated by silence. Its SHA-256 is
`dd1dbdd45e6c0461120f2b8b74851d7e9a46fd8f5fc018f052f1b5cd1542d5f6`.

Each run used a new, empty `HF_HOME`. The OS sandbox denied all network activity,
in addition to the application's offline settings and Python socket guard. The
only files subsequently created under those Hugging Face homes were local custom
code modules; no general Hub weight cache was populated. Models came exclusively
from `.models`.

Both backends completed scene detection, full-recording diarization, transcription,
and Japanese Molmo2 descriptions. Both detected two speakers automatically and
retained Japanese and English. The tests then resumed with a worker factory that
raises an error if invoked: both resumes passed without loading a model.

| Measured operation | Whisper | Qwen |
| --- | ---: | ---: |
| Complete API invocation, including worker startup/load | 73.50 s | 80.42 s |
| Diarization inference, CPU | 5.78 s | 5.53 s |
| ASR inference, MPS FP16, summed speech chunks | 12.73 s | 5.28 s |
| Molmo2 inference, MLX, eight 320×240 frames | 36.53 s | 36.93 s |
| Molmo2 peak MLX device allocation | 7.89 GiB | 7.89 GiB |

No analysis checkpoints were reused during the initial runs. Filesystem and Metal
kernel caches could be warm. These timings exclude model downloads and environment
installation. Runtime loading, decoding, and orchestration explain the difference
between stage inference times and the complete invocation.

The initial Mac integration gate used two ordered frames showing a red square
moving across a white background. Molmo2 loaded offline, described the movement
in Japanese, and completed generation in 31.65 seconds with a 7.41 GiB MLX peak.
Its deliberately short 128-token limit truncated the final sentence; production
uses a 448-token budget and a shorter-description prompt.

The maximum-frame follow-up used eight ordered 768×576 images with a red square
moving from left to right, spanning a 30-second observation window. It passed
with OS networking disabled: **140.15 seconds inference, 146.78 seconds for the
test, and 9.51 GiB peak MLX device allocation** (10,208,660,664 bytes). It produced
Japanese text describing the square's movement without running out of memory.
The model incorrectly described the supplied four-second sample spacing as one
second; generated prose is not a substitute for the canonical numeric timestamps.

MLX device allocation is not total system memory. Process RSS omits some Metal
allocations, and the figures overlap on unified memory: do not add RSS and MLX
allocation. Other applications and input resolution affect available headroom.

## Quality observations and limits

Whisper correctly transcribed the Japanese sentence in this fixture. Qwen wrote
`動学解析` where the intended phrase was `動画解析`. Both interpreted the synthesized
English word “offline” as “a flying.” These outputs were retained without manual
correction. This test demonstrates functioning language preservation, alignment,
and speaker assignment; it does not establish recognition accuracy on natural
recordings. Automatic language detection restarts across separated speech regions;
switching languages inside uninterrupted speech can still be missed.

Molmo2 produced meaningful Japanese descriptions of the red square and white
background. Descriptions may include redundant wording or unsupported details.
Sparse frame sampling can miss brief actions. Reports identify the models and
quantization and state that generated results are estimates.

The pinned macOS PyAV/OpenCV wheels emit duplicate FFmpeg AVFoundation class
warnings when imported together. They did not prevent these file-decoding tests.
Pyannote emitted a pooling warning on the synthetic voices; Transformers emitted
a deprecation warning. The diagnostics remain on stderr.

Linux CUDA dependency resolution and a hash lock are provided. **No NVIDIA GPU
was available, so CUDA installation and real-model inference were not executed.**
The original Molmo2 CUDA backend follows the upstream multi-image interface but
still requires validation on the target GPU. No hour-long real-model run, dense
overlapping-speech accuracy test, or broad codec corpus was executed. CPU retry
behavior was fault-injected in tests; neither successful MPS smoke test required
a CPU retry.

## Reproduce real-model checks

After preparation, supply an MP4 containing speech and visible content. On macOS,
the following command adds OS-level network denial while allowing Metal access:

```bash
/usr/bin/sandbox-exec -p '(version 1) (allow default) (deny network*)' \
  /usr/bin/env VIDEO_ANALYZER_MODEL_TEST_VIDEO=/absolute/path/fixture.mp4 \
  .venv/bin/pytest -m models -v -s
```

For a Japanese/English fixture, also set
`VIDEO_ANALYZER_TEST_LANGUAGES=japanese,english`. The tests create independent empty
Hugging Face homes and write JSON/Markdown results under pytest's temporary
directory. On Linux, set `VIDEO_ANALYZER_TEST_PROFILE=cuda` and use a network-disabled
container or network namespace suitable for the installed NVIDIA runtime.

The ignored `.validation` directory in this working copy contains the generated
fixture, raw reports, logs, and checkpoint measurements from this session. Test
fixtures and model weights are not included in the package.

## Sortformer v2.1 setup and validation (2026-10-04)

The optional reference runtime was installed on the same Apple M2 Mac using
`mise run setup:sortformer`: Python 3.12.15, NeMo 2.7.3, and
PyTorch/torchaudio 2.8.0. The CLI and existing workers still use Python 3.11.15.
Separate dependency resolutions and SHA-256 locks were generated for macOS arm64
and Linux x86_64. Linux installation, CPU inference, and CUDA inference were not
executed; no Linux/NVIDIA host was available.

`prepare-models --profile mac --diarization-backend sortformer` downloaded the
original NVIDIA checkpoint while preserving the pyannote, Whisper, Qwen, aligner,
and Molmo2 manifest entries. The checkpoint is 471,367,680 bytes, from immutable
revision `cd03eee90fbec18297ac31b8c21546e596b7f71c`, with SHA-256
`8abd32832159c6ac1148c926b7276f35ba34582c444e559dce1f1253fea42ef8`.
It is the pretrained v2.1 model, not a Japanese fine-tuned checkpoint.

Offline `doctor --profile mac --asr both --diarization-backend both --verify-hashes`
passed with all three worker environments and all six model entries. The NeMo
class was imported inside the Sortformer worker with both the application's
Python socket guard and the macOS network sandbox active. NeMo warns that the
configured cache update period 300 has an effective period of 340 with the
recording preset's chunk length; both values are recorded in provenance.

`mise run check` passed **357 normal tests**, including lint and formatting.
Coverage includes both diarizers with both ASRs across Mac, CPU, and CUDA profile
routing; preparation coexistence and backend-specific errors; missing/ancillary-only
checkpoints; silence, overlap, ties, partial frames, and speech-union preservation;
streaming-state continuity and progress cleanup; and resume/cache isolation.
These profile-routing tests use model doubles and do not establish CUDA hardware
support. `UV_OFFLINE=1 mise run build` built the wheel and source distribution;
both contain the Sortformer worker and exclude local models, runtimes, and caches.
All setup scripts passed shell syntax checks.

Five real-model tests also passed under OS network denial: the four combinations
of pyannote/Sortformer and Whisper/Qwen, and a Sortformer edge-case test. Each
pipeline completed Japanese/English transcription, speaker assignment, and Molmo2
visual descriptions, detected two speakers, and then resumed with a worker factory
that raises if invoked. Each used a separate empty `HF_HOME`, without a token;
none populated a general Hub weight cache. The input was the same 16.3-second
Japanese/English fixture described above. This confirms integration, not an
accuracy improvement.

The Sortformer edge-case test reused one loaded worker for empty audio, 10 ms of
silence, one second of silence, and 65 seconds of repeated fixture audio. Empty
and silent inputs produced empty timelines. The long input completed three
streaming steps, emitted progress through the worker protocol, and produced
bounded, nonoverlapping exclusive turns. The tiny input was padded internally
for NeMo's STFT, with output timestamps clamped to its original duration.

To reproduce these five checks after preparing both diarizers and ASR backends:

```bash
/usr/bin/sandbox-exec -p '(version 1) (allow default) (deny network*)' \
  /usr/bin/env VIDEO_ANALYZER_MODEL_TEST_VIDEO=/absolute/path/fixture.mp4 \
  VIDEO_ANALYZER_TEST_DIARIZATION=both \
  .venv/bin/pytest -m models -k 'real_offline_pipeline or real_nemo' -v -s
```

On an NVIDIA host, also set `VIDEO_ANALYZER_TEST_PROFILE=cuda` and use the Linux
network-isolation procedure described above. The opt-in tests select that
profile's CUDA device, while Sortformer uses CPU for the Mac profile.

### Japanese recording comparison

At the user's request, this comparison used generated Japanese speech. macOS
`say` synthesized one passage with **Kyoko** and another with **Eddy (Japanese
(Japan))**, at rate 160. Each passage was repeated four times in alternating
order, with 0.8 seconds of digital silence inserted before every clip. The
result is **109.28075 seconds**, 16 kHz mono PCM, with no intentional overlap.
Its SHA-256 is
`59e2843310997bcc0728a18cba9ab6c3732dc48648c09aa550e919f56e755a55`.
Normalization and denoising were off. The passages were:

- Kyoko: 「今日は、動画の話者を区別する仕組みについて話します。音声の処理はすべてこのコンピューターで実行します。次の発表者に交代します。」
- Eddy: 「ありがとうございます。私は、長い録音の中でも同じ話者を識別できるか確認します。ネットワークに接続しなくても分析できることが大切です。」

Both diarizers consumed the identical audio on the Mac CPU, in separate fresh
workers with automatic speaker counts. They ran sequentially, without ASR or
vision inference alongside them, with empty Hugging Face homes, no token, and
OS networking blocked. Sortformer completed **five streaming steps** using the
recording preset and retained speaker state across them.

| Measurement | pyannote Community-1 | Sortformer v2.1 |
| --- | ---: | ---: |
| Diarization inference, excluding startup/load | 69.79 s | 2.86 s |
| Worker startup, load, inference, and shutdown | 75.21 s | 8.20 s |
| Peak process RSS | 4.95 GiB | 2.03 GiB |
| Detected speakers | 2 | 2 |
| Regular / exclusive turns | 20 / 20 | 16 / 16 |

Each voice retained one consistent speaker ID in all four occurrences, with no
observed interchange between the two voices inside the source clips. Both
models extended some speech onsets into the known inserted silence: approximately
**0.185 seconds total for pyannote** and **0.036 seconds for Sortformer**. These
are boundary observations from the generated fixture, not DER. The source clips
also contain pauses, so their full clip boundaries are not reference speech
annotations; missed speech and overall DER were not calculated.

This is one smoke comparison of two repeated synthetic voices, not a benchmark
of natural Japanese talks, overlapping speech, or more than four speakers. Disk
and library caches could be warm. RSS is the worker's process high-water mark,
including its runtime and loaded model; it is not total system memory or CUDA
allocation. The observed speed and memory differences do not promise improved
Japanese accuracy or throughput on other hardware. Raw outputs, fixture spans,
and the local comparison script are retained in the ignored
`.validation/sortformer-japanese` directory of this working copy.

## Nemotron 3 setup and validation (2026-10-04)

The optional Nemotron runtime was installed on the same Apple M2 Mac with
`mise run setup:nemotron`: Python 3.12.15, PyTorch/torchaudio 2.8.0, and NVIDIA
NeMo source revision `1688cc3d6a9ade854f544987810c53f605dc86fc`, reporting version
3.1.0. The published NeMo 3.0.0 implementation lacks the required high-resolution
output support. Separate dependency resolutions and SHA-256 locks were generated
for macOS arm64 and Linux x86_64. Both locks record source archive SHA-256
`bd7bb5ebc4dba9f377fa39b064e4e8fd01d6ea6e94b430300595ccda25f506f2`.
Linux installation and CPU/CUDA inference were not executed; no Linux/NVIDIA host
was available. Existing CLI, PyTorch, MLX, and Sortformer environments were retained.

Explicit `prepare-models --profile mac --diarization-backend nemotron` preparation
preserved all six previously prepared model manifest entries. The NVIDIA
`Nemotron-3-Diarization.nemo` checkpoint is 198,676,480 bytes, from immutable model
revision `f667ed73aee57d40cc39428eb768b4fd87a0a29e`, with SHA-256
`867c53f552998f772e5b5e5c082962ae85ee7ca5669c2bc17d7f615133d4e96d`.
Only this checkpoint and README were recorded under `nemotron`. Its restored
configuration has eight speaker channels, high-resolution output enabled, and
output subsampling factor one at a 10 ms feature stride.

Offline `doctor --profile mac --diarization-backend nemotron --verify-hashes`
passed under OS network denial, checking the pinned source and required APIs as
well as the existing ASR and vision workers. Inference uses CPU for Nemotron,
retains 80 ms encoder units for the recording preset, and derives timestamps and
progress from the actual 10 ms output stride. NeMo reports an effective cache
update period of 340 for the configured period 300; both are recorded in runtime
provenance, along with the source revision and exclusive-timeline derivation.

All **668 normal tests** passed. Coverage includes the three diarizers with both
ASRs and Mac/CPU/CUDA profile routing, eight-channel probabilities, all eight
speaker labels, 10 ms overlap projection, silence and partial-frame boundaries,
invalid checkpoints and runtime sources, progress cleanup, and cache invalidation
when the backend, model revision, NeMo source revision, or adapter settings change.
These routing tests use model doubles; CUDA hardware coverage remains unverified.
Lint, formatting, setup-script syntax checks, and the offline wheel/source build
also passed.

Three real Nemotron acceptance tests passed under OS network denial: pipeline
integration with Whisper and Qwen, and an empty/short/silent/multiple-chunk audio
test. Both pipelines used the existing 16.3-second Japanese/English fixture,
detected two speakers, produced Japanese scene descriptions, and then completed
resume with a worker factory that raises if invoked. Each test used a separate
empty `HF_HOME` without a token. The edge-case test reused one loaded worker for
empty audio, 10 ms of silence, one second of silence, and 65 seconds of repeated
fixture audio. Silent inputs produced empty timelines; the long input emitted
three streaming steps and bounded, nonoverlapping exclusive turns.

To reproduce the three checks after preparing the Nemotron checkpoint and both
ASR backends:

```bash
/usr/bin/sandbox-exec -p '(version 1) (allow default) (deny network*)' \
  /usr/bin/env VIDEO_ANALYZER_MODEL_TEST_VIDEO=/absolute/path/fixture.mp4 \
  VIDEO_ANALYZER_TEST_DIARIZATION=nemotron \
  .venv/bin/pytest -m models -k 'real_offline_pipeline or real_nemo' -v -s
```

Raw JSON/Markdown reports, checkpoints, and edge-case WAV files remain in the
ignored `.validation/nemotron-acceptance` directory of this working copy. The
eight-speaker output limit and timeline math are covered by unit tests; this
two-speaker fixture establishes integration rather than a diarization accuracy
benchmark.
