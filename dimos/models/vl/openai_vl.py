# Copyright 2026 Dimensional Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from functools import cached_property
import os
from typing import Any

import numpy as np
from openai import OpenAI

from dimos.models.vl.base import VlModel, VlModelConfig
from dimos.msgs.sensor_msgs.Image import Image


class OpenAIVlModelConfig(VlModelConfig):
    """Configuration for the OpenAI (gpt-4o) vision-language model."""

    model_name: str = "gpt-4o"
    api_key: str | None = None
    # Downscale frames before upload to cut tokens/latency (gpt-4o handles ~1024 px well).
    auto_resize: tuple[int, int] | None = (1024, 1024)


class OpenAIVlModel(VlModel):
    """VL backend that runs detection/description via the OpenAI vision API (gpt-4o).

    Unlike the local qwen/moondream backends this needs NO GPU/RAM on the host — it
    calls the cloud — so it works on memory-constrained machines (e.g. a laptop that is
    already running the nav voxel-mapper on a small GPU). Used by look_out_for / observe
    for person detection. gpt-4o's pixel boxes are approximate but reliable for presence,
    which is all the lookout/announce flow needs.
    """

    config: OpenAIVlModelConfig

    @cached_property
    def _client(self) -> OpenAI:
        api_key = self.config.api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OpenAI API key must be provided or set in the OPENAI_API_KEY environment variable"
            )
        return OpenAI(api_key=api_key)

    def query(self, image: Image | np.ndarray, query: str) -> str:  # type: ignore[override]
        if isinstance(image, np.ndarray):
            image = Image.from_numpy(image)

        # Apply auto_resize if configured (keeps the upload small).
        image, _ = self._prepare_image(image)
        img_base64 = image.to_base64()

        response = self._client.chat.completions.create(
            model=self.config.model_name,
            temperature=0,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{img_base64}"},
                        },
                        {"type": "text", "text": query},
                    ],
                }
            ],
        )
        return response.choices[0].message.content or ""

    def stop(self) -> None:
        """Release the OpenAI client."""
        if "_client" in self.__dict__:
            del self.__dict__["_client"]
