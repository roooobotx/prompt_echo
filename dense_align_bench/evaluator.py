"""
DenseAlignBench Evaluator

Pairwise comparison evaluator for text-to-image prompt-following assessment.
Uses OpenAI-compatible VLM APIs (default: Google Gemini) to compare two images
based on how well each follows a given text prompt.

Part of the PromptEcho project:
  "PromptEcho: Annotation-Free Reward from Vision-Language Models
   for Text-to-Image Reinforcement Learning" (arXiv:2604.12652)
"""

import os
import time
import json
import base64
import random
import re
from io import BytesIO
from typing import Dict, List, Optional, Union
from PIL import Image

from openai import OpenAI


class DenseAlignEvaluator:
    """
    Pairwise comparison evaluator for text-to-image prompt-following.

    Uses any OpenAI-compatible vision-language model API to compare two images
    and determine which better follows a given text prompt. Defaults to
    Google Gemini's OpenAI-compatible endpoint.

    Supports:
    - Pairwise comparison of two images for prompt-following accuracy
    - Multi-URL load balancing across API endpoints
    - Configurable retry logic with exponential backoff
    """

    PAIRWISE_PROMPT_FOLLOWING_TEMPLATE = """You are a professional image quality evaluator specializing in prompt adherence assessment.

Your task is to compare two AI-generated images based on how well each follows the given prompt. Focus ONLY on prompt-following accuracy - do NOT consider aesthetics, artistic quality, or photorealism.

**Prompt:**
{prompt}

**Evaluation Process:**
1. Carefully read and understand ALL requirements in the prompt:
   - Main subjects and objects
   - Actions and poses
   - Visual attributes (colors, sizes, materials, textures)
   - Composition and layout
   - Style and atmosphere
   - Any text or written elements
   - Spatial relationships (foreground, background, positions)
   - Quantities and counts

2. Examine Image A:
   - Which prompt requirements are accurately depicted?
   - Which prompt requirements are missing or incorrect?
   - Are there any elements NOT mentioned in the prompt?

3. Examine Image B:
   - Which prompt requirements are accurately depicted?
   - Which prompt requirements are missing or incorrect?
   - Are there any elements NOT mentioned in the prompt?

4. Compare:
   - Which image captures more of the prompt requirements accurately?
   - If both are equal in accuracy, select "tie"

**Preference Options:**
- "image_a": Image A follows the prompt better than Image B
- "image_b": Image B follows the prompt better than Image A
- "tie": Both images follow the prompt equally well (similar accuracy)

**Important:**
- Base your decision ONLY on prompt-following accuracy
- Ignore differences in artistic style, aesthetics, or realism
- Be objective and fair
- Choose "tie" when prompt-following quality is genuinely similar

**Output Format (strict JSON, no markdown):**
{{
  "reasoning": "<Detailed explanation: 1) How well Image A follows the prompt (what's correct, what's missing); 2) How well Image B follows the prompt (what's correct, what's missing); 3) Which is better and why, or why they are equal>",
  "preference": "<image_a, image_b, or tie>"
}}

Please provide an objective, thorough comparison.
"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Union[str, List[str]] = "https://generativelanguage.googleapis.com/v1beta/openai/",
        model: str = "gemini-2.0-flash",
        temperature: float = 0.3,
        request_delay: float = 0.5,
        max_retries: int = 3,
        max_tokens: Optional[int] = None,
    ):
        """
        Initialize DenseAlignBench Evaluator.

        Args:
            api_key: API key. If None, reads from GEMINI_API_KEY or OPENAI_API_KEY env var.
            base_url: API endpoint URL(s). Can be a single URL string or a list of URLs
                      for load balancing. Defaults to Google Gemini's OpenAI-compatible endpoint.
            model: Model name (default: "gemini-2.0-flash"). Examples:
                   "gemini-2.0-flash", "gemini-2.5-flash-preview", "gpt-4o", etc.
            temperature: Temperature for API calls (lower = more consistent).
            request_delay: Delay between API calls to avoid rate limits (seconds).
            max_retries: Maximum number of retries for failed API calls.
            max_tokens: Maximum tokens for API response. None for API default.
        """
        # Resolve API key from env if not provided
        if api_key is None:
            api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise ValueError(
                    "No API key provided. Set GEMINI_API_KEY or OPENAI_API_KEY "
                    "environment variable, or pass api_key= directly."
                )

        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.request_delay = request_delay
        self.max_retries = max_retries
        self.max_tokens = max_tokens

        # Support multiple base_urls for load balancing
        if isinstance(base_url, list):
            self.base_urls = base_url
            self.base_url = base_url[0]
        else:
            self.base_urls = [base_url]
            self.base_url = base_url

        # Initialize OpenAI clients (one per base_url)
        self.clients = [
            OpenAI(api_key=self.api_key, base_url=url)
            for url in self.base_urls
        ]
        self.client = self.clients[0]

        print(f"DenseAlignEvaluator initialized")
        print(f"  - Model: {self.model}")
        print(f"  - Base URLs: {self.base_urls}")
        print(f"  - Temperature: {self.temperature}")
        print(f"  - Request delay: {self.request_delay}s")

    @staticmethod
    def encode_image_to_base64(image: Image.Image) -> str:
        """
        Encode PIL Image to base64 string.

        Args:
            image: PIL Image object

        Returns:
            Base64 encoded string of the image
        """
        if image.mode != "RGB":
            image = image.convert("RGB")

        buffered = BytesIO()
        image.save(buffered, format="JPEG", quality=95)
        img_bytes = buffered.getvalue()

        img_base64 = base64.b64encode(img_bytes).decode('utf-8')
        return img_base64

    def _extract_json_from_response(self, response_text: str) -> Optional[Dict]:
        """
        Extract JSON from API response, handling markdown code blocks.

        Args:
            response_text: Raw response text

        Returns:
            Parsed JSON dict, or None if parsing failed
        """
        # Try to parse directly
        try:
            return json.loads(response_text)
        except json.JSONDecodeError:
            pass

        # Try to extract from markdown code block
        patterns = [
            r'```json\s*\n(.*?)\n```',
            r'```\s*\n(.*?)\n```',
            r'({.*})',  # Last resort: extract any JSON-like structure
        ]

        for pattern in patterns:
            match = re.search(pattern, response_text, re.DOTALL)
            if match:
                try:
                    json_str = match.group(1).strip()
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    continue

        print(f"  Warning: Failed to extract JSON from response")
        print(f"  Response preview: {response_text[:200]}...")
        return None

    def compare_two_images_prompt_following(
        self,
        prompt: str,
        image_a: Image.Image,
        image_b: Image.Image,
        model_a_name: str = "model_a",
        model_b_name: str = "model_b"
    ) -> Optional[Dict]:
        """
        Compare two images based on prompt-following only.

        Args:
            prompt: The text prompt both images were generated from
            image_a: First PIL Image
            image_b: Second PIL Image
            model_a_name: Name/identifier for first model
            model_b_name: Name/identifier for second model

        Returns:
            Dict containing:
            {
                "reasoning": str,
                "preference": str,  # "image_a", "image_b", or "tie"
                "model_a": str,
                "model_b": str,
                "raw_response": str
            }
            or None if evaluation failed
        """
        # Encode both images to base64
        img_a_base64 = self.encode_image_to_base64(image_a)
        img_b_base64 = self.encode_image_to_base64(image_b)

        # Format prompt
        eval_prompt = self.PAIRWISE_PROMPT_FOLLOWING_TEMPLATE.format(prompt=prompt)

        # Prepare messages with both images
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": eval_prompt
                    },
                    {
                        "type": "text",
                        "text": "**Image A:**"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{img_a_base64}"
                        }
                    },
                    {
                        "type": "text",
                        "text": "**Image B:**"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{img_b_base64}"
                        }
                    }
                ]
            }
        ]

        # Build API call kwargs
        create_kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
        }
        if self.max_tokens is not None:
            create_kwargs["max_tokens"] = self.max_tokens

        # Call API with retries
        for attempt in range(self.max_retries):
            try:
                # Random load balancing across multiple base_urls
                client = random.choice(self.clients)
                response = client.chat.completions.create(**create_kwargs)

                response_text = response.choices[0].message.content.strip()

                # Extract JSON
                result = self._extract_json_from_response(response_text)

                if result is None:
                    raise ValueError("Failed to extract JSON from response")

                # Validate preference
                if "preference" not in result or result["preference"] not in ["image_a", "image_b", "tie"]:
                    raise ValueError(f"Invalid preference: {result.get('preference')}")

                # Format return
                return {
                    "reasoning": result.get("reasoning", ""),
                    "preference": result["preference"],
                    "model_a": model_a_name,
                    "model_b": model_b_name,
                    "raw_response": response_text
                }

            except Exception as e:
                print(f"  Attempt {attempt + 1}/{self.max_retries} failed: {e}")

                if attempt < self.max_retries - 1:
                    wait_time = (2 ** attempt) * self.request_delay
                    print(f"  Retrying in {wait_time:.1f}s...")
                    time.sleep(wait_time)
                else:
                    print(f"  All retries failed")
                    return None

        return None
