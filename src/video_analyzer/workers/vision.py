import math

from ..errors import AnalysisError


def description_prompt(timestamps: list[float], language: str) -> str:
    times = ", ".join(f"{t:.3f}s" for t in timestamps)
    if language == "ja":
        return (
            f"これらは同じ動画の時系列順のフレームです。各画像の時刻は順に {times} です。"
            "画面に見える人物・物体・場所・動作と、フレーム間の変化を日本語の2〜3文で簡潔に説明してください。"
            "同じ内容のフレームを個別に列挙せず、共通する内容をまとめてください。"
            "見えない出来事、音声の内容、人物の名前は推測しないでください。説明文だけを出力してください。"
        )
    return (
        f"These frames are in chronological order from one video. Their timestamps are {times}. "
        "Describe the visible people, objects, setting, actions, and changes between frames "
        "in two or three English sentences. Summarize repeated content instead of listing every frame. "
        "Do not infer unseen events, speech, or people's names. Output only the description."
    )


def validate_description(text: str) -> str:
    text = text.strip()
    if not text or not any(c.isalnum() for c in text) or len(set(text)) < 3:
        raise AnalysisError("Molmo2 returned an empty or degenerate description")
    return text


class MlxVision:
    def __init__(self, options):
        import mlx.core as mx
        from mlx_vlm import load

        if options.get("device", "auto") != "auto":
            raise AnalysisError("The mac vision backend uses MLX/Metal; use --vision-device auto")
        if not mx.metal.is_available():
            raise AnalysisError("MLX Metal acceleration is not available")
        # Bound allocator caching; it otherwise retains large temporary activations.
        mx.set_cache_limit(256 * 1024**2)
        self.model, self.processor = load(
            options["model"],
            trust_remote_code=True,
            local_files_only=True,
        )
        self.device = "mlx-metal"

    def describe(
        self, frames: list[str], timestamps: list[float], language: str, max_new_tokens: int
    ):
        import mlx.core as mx
        from mlx_vlm import apply_chat_template, generate

        prompt = apply_chat_template(
            self.processor,
            self.model.config,
            description_prompt(timestamps, language),
            num_images=len(frames),
        )
        response = generate(
            self.model,
            self.processor,
            prompt,
            image=frames,
            max_tokens=max_new_tokens,
            temperature=0.0,
            verbose=False,
        )
        peak = mx.get_peak_memory()
        if not math.isfinite(peak):
            raise AnalysisError("MLX reported invalid memory statistics")
        mx.clear_cache()
        return {"description": validate_description(response.text), "peak_device_bytes": peak}


class TorchVision:
    def __init__(self, options):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        from .audio import select_device

        self.device = select_device(options.get("device", "auto"), prefer_mps=False)
        dtype = (
            torch.bfloat16
            if self.device == "cuda" and torch.cuda.is_bf16_supported()
            else torch.float32
        )
        self.processor = AutoProcessor.from_pretrained(
            options["model"], trust_remote_code=True, local_files_only=True
        )
        self.model = (
            AutoModelForImageTextToText.from_pretrained(
                options["model"],
                trust_remote_code=True,
                local_files_only=True,
                dtype=dtype,
                attn_implementation="sdpa",
            )
            .to(self.device)
            .eval()
        )

    def describe(
        self, frames: list[str], timestamps: list[float], language: str, max_new_tokens: int
    ):
        import torch
        from PIL import Image

        images = []
        try:
            for path in frames:
                with Image.open(path) as image:
                    images.append(image.convert("RGB"))
            messages = [
                {
                    "role": "user",
                    "content": [
                        *[{"type": "image", "image": image} for image in images],
                        {"type": "text", "text": description_prompt(timestamps, language)},
                    ],
                }
            ]
            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.inference_mode():
                generated = self.model.generate(
                    **inputs, max_new_tokens=max_new_tokens, do_sample=False
                )
            text = self.processor.tokenizer.decode(
                generated[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True
            )
            return {
                "description": validate_description(text),
                "peak_device_bytes": torch.cuda.max_memory_allocated()
                if self.device == "cuda"
                else 0,
            }
        finally:
            for image in images:
                image.close()
