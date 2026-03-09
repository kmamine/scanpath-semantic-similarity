"""
VLM client for generating descriptions.
"""

from __future__ import annotations

import base64
import logging
import time
from dataclasses import dataclass
from io import BytesIO
from typing import Optional, Union

from PIL import Image

log = logging.getLogger(__name__)


@dataclass
class VLMResponse:
    content: str
    prompt_tokens: int
    completion_tokens: int
    latency_s: float


class VLMClient:
    """OpenAI-compatible VLM client with retry logic."""

    def __init__(
        self,
        base_url: str = "http://localhost:8000/v1",
        api_key: str = "x",
        model: str = "qwen2-vl-7b-instruct",
        retries: int = 3,
    ):
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.retries = retries
        self._client = None

    @property
    def client(self):
        if self._client is None:
            import openai

            self._client = openai.OpenAI(
                base_url=self.base_url,
                api_key=self.api_key,
            )
        return self._client

    def _pil_to_b64(self, img: Image.Image, fmt: str = "JPEG") -> str:
        buf = BytesIO()
        if img.mode != "RGB":
            img = img.convert("RGB")
        img.save(buf, format=fmt)
        return base64.b64encode(buf.getvalue()).decode()

    def describe(
        self,
        image: Union[str, Image.Image],
        prompt: str,
        max_tokens: int = 80,
        temperature: float = 0.2,
    ) -> VLMResponse:
        """
        Generate description for an image.

        Args:
            image: PIL Image or path to image file
            prompt: Text prompt for the VLM
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            VLMResponse with generated text
        """
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")

        b64_img = self._pil_to_b64(image)

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"},
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        last_error = None
        for attempt in range(self.retries):
            try:
                t0 = time.perf_counter()
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                latency = time.perf_counter() - t0

                content = resp.choices[0].message.content.strip()
                prompt_tokens = resp.usage.prompt_tokens if resp.usage else 0
                completion_tokens = resp.usage.completion_tokens if resp.usage else 0

                return VLMResponse(
                    content=content,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    latency_s=latency,
                )
            except Exception as e:
                last_error = e
                log.warning(f"VLM call attempt {attempt + 1} failed: {e}")
                time.sleep(2**attempt)

        log.error(f"VLM call failed after {self.retries} attempts: {last_error}")
        raise RuntimeError(
            f"VLM call failed after {self.retries} attempts: {last_error}"
        )
