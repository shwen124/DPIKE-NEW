"""Format PMET rewrite contexts for counterfact-style or prefix-completion records."""

from __future__ import annotations

from typing import Any, Dict, List


def request_completion_text(request: Dict[str, Any]) -> str:
    if request.get("completion_context"):
        return str(request["completion_context"])
    return request["prompt"].format(request["subject"])


def format_contexts_for_requests(
    requests: List[Dict[str, Any]],
    context_templates: List[List[str]],
) -> List[str]:
    lines: List[str] = []
    for request in requests:
        base = request_completion_text(request)
        for context_type in context_templates:
            for context in context_type:
                if "{}" in context:
                    lines.append(context.format(base))
                else:
                    lines.append(base)
    return lines
