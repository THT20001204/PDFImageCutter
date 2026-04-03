"""
多模态 AI 子图检测模块。

支持 OpenAI / Claude / 通义千问 / 智谱 四个供应商，
对一张图片返回子图边界框列表（归一化坐标 0-1）。
"""

import base64
import json
import logging
import mimetypes
import os
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "你是一个图片分析助手。你的任务是判断一张图片是单一完整图片，"
    "还是由多个独立子图（sub-figure）拼接/排列而成。"
)

USER_PROMPT = (
    "请分析这张图片：\n"
    "1. 如果它是一张完整的独立图片（例如单张照片、单个图表），返回空列表。\n"
    "2. 如果它是多个子图拼接在一起（例如论文中的 Figure 含 A/B/C/D 多个面板），"
    "请返回每个子图的边界框。\n\n"
    "严格以下面的 JSON 格式返回，不要添加任何其他文字：\n"
    '{"subimages": [{"bbox": [x1, y1, x2, y2], "label": "子图描述"}]}\n\n'
    "坐标规则：\n"
    "- x1, y1 是左上角，x2, y2 是右下角\n"
    "- 所有值为 0.0 到 1.0 之间的归一化比例（相对于图片宽高）\n"
    "- 如果是单一完整图片，返回：{\"subimages\": []}"
)

PROVIDER_DEFAULTS: Dict[str, Dict[str, str]] = {
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-4o",
    },
    "claude": {
        "base_url": "https://api.anthropic.com",
        "model": "claude-sonnet-4-20250514",
    },
    "qwen": {
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model": "qwen-vl-max",
    },
    "zhipu": {
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "model": "glm-4v-flash",
    },
}


def _encode_image_base64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _guess_mime(image_path: str) -> str:
    mime, _ = mimetypes.guess_type(image_path)
    return mime or "image/png"


def _extract_json_from_text(text: str) -> Optional[dict]:
    """从 AI 回复文本中提取 JSON 对象（兼容 markdown 代码块包裹）。"""
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        json_lines = []
        inside = False
        for line in lines:
            if line.strip().startswith("```") and not inside:
                inside = True
                continue
            if line.strip().startswith("```") and inside:
                break
            if inside:
                json_lines.append(line)
        text = "\n".join(json_lines).strip()

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            pass
    return None


class AISplitter:
    """多供应商多模态 AI 子图检测器。"""

    def __init__(
        self,
        provider: str,
        api_key: str,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: int = 60,
    ):
        self.provider = provider.lower().strip()
        if self.provider not in PROVIDER_DEFAULTS:
            raise ValueError(
                f"不支持的 provider: {self.provider}，"
                f"可选: {', '.join(PROVIDER_DEFAULTS)}"
            )

        defaults = PROVIDER_DEFAULTS[self.provider]
        self.api_key = api_key
        self.model = model or defaults["model"]
        self.base_url = (base_url or defaults["base_url"]).rstrip("/")
        self.timeout = timeout

    def detect_subimages(self, image_path: str) -> List[Dict[str, Any]]:
        """
        对一张图片调用 AI 分析，返回子图列表。

        Returns:
            归一化坐标的子图列表，每项含 bbox=[x1,y1,x2,y2] 和 label。
            空列表表示单一完整图片或分析失败。
        """
        try:
            if self.provider == "claude":
                raw_text = self._call_claude(image_path)
            else:
                raw_text = self._call_openai_compatible(image_path)

            return self._parse_response(raw_text)

        except Exception as exc:
            logger.warning(f"AI 分析失败 ({self.provider}): {exc}")
            return []

    def _call_openai_compatible(self, image_path: str) -> str:
        """OpenAI / 通义千问 / 智谱 统一的 OpenAI-compatible 接口。"""
        b64 = _encode_image_base64(image_path)
        mime = _guess_mime(image_path)

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime};base64,{b64}",
                            },
                        },
                        {"type": "text", "text": USER_PROMPT},
                    ],
                },
            ],
            "max_tokens": 1024,
            "temperature": 0.1,
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        url = f"{self.base_url}/chat/completions"
        resp = requests.post(url, json=payload, headers=headers, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]

    def _call_claude(self, image_path: str) -> str:
        """Anthropic Claude Messages API。"""
        b64 = _encode_image_base64(image_path)
        mime = _guess_mime(image_path)

        payload = {
            "model": self.model,
            "max_tokens": 1024,
            "system": SYSTEM_PROMPT,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": mime,
                                "data": b64,
                            },
                        },
                        {"type": "text", "text": USER_PROMPT},
                    ],
                }
            ],
        }

        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
        }

        url = f"{self.base_url}/v1/messages"
        resp = requests.post(url, json=payload, headers=headers, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()
        return data["content"][0]["text"]

    @staticmethod
    def _parse_response(raw_text: str) -> List[Dict[str, Any]]:
        """解析 AI 返回的 JSON，提取并校验子图列表。"""
        parsed = _extract_json_from_text(raw_text)
        if parsed is None:
            logger.warning(f"无法从 AI 响应中解析 JSON: {raw_text[:200]}")
            return []

        subimages = parsed.get("subimages")
        if not isinstance(subimages, list):
            return []

        valid = []
        for item in subimages:
            bbox = item.get("bbox")
            if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                continue
            try:
                coords = [float(v) for v in bbox]
            except (TypeError, ValueError):
                continue

            if all(0.0 <= v <= 1.0 for v in coords):
                x1, y1, x2, y2 = coords
                if x2 > x1 and y2 > y1:
                    valid.append({
                        "bbox": coords,
                        "label": str(item.get("label", "")),
                    })

        return valid
