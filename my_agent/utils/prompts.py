"""提示词管理模块。

从 prompts.yml 加载提示词模板，支持变量渲染。

用法:
    from my_agent.utils.prompts import render

    prompt = render("recommend.system", movies="...", query="...")
"""

import os
from pathlib import Path

import yaml

_PROMPTS_FILE = Path(__file__).resolve().parent.parent.parent / "prompts.yml"
_PROMPTS: dict = {}

_loaded = False


def load() -> dict:
    """加载 YAML 提示词配置（懒加载，缓存结果）。"""
    global _PROMPTS, _loaded
    if not _loaded:
        with open(_PROMPTS_FILE, "r", encoding="utf-8") as f:
            _PROMPTS = yaml.safe_load(f)
        _loaded = True
    return _PROMPTS


def render(path: str, **kwargs) -> str:
    """
    渲染指定路径的提示词模板。

    Args:
        path: 点号分隔的 YAML 路径，如 "recommend.system"、"llm_expand.system"
        **kwargs: 模板变量，如 movies="...", query="..."

    Returns:
        渲染后的提示词字符串
    """
    prompts = load()
    keys = path.split(".")
    template = prompts
    for k in keys:
        if isinstance(template, dict):
            template = template[k]
        else:
            template = template
    return template.format(**kwargs)


def get_llm_expand_prompt() -> str:
    """获取 LLM 查询扩展的 system prompt。"""
    return load()["llm_expand"]["system"]
