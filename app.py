from flask import Flask, render_template, request, jsonify, send_from_directory, Response
import os
import re
import json
import yaml
import time
import statistics
from urllib.parse import urlparse
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI
import httpx
import requests
import copy
import uuid
from typing import Any
import json_repair

# 2. app / client / 全局变量
app = Flask(__name__)

load_dotenv()


DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
FALLBACK_API_KEY = os.getenv("FALLBACK_API_KEY", "")
FALLBACK_BASE_URL = os.getenv("FALLBACK_BASE_URL", "")
FALLBACK_MODEL = os.getenv("FALLBACK_MODEL", "")
DIFY_API_URL = os.getenv("DIFY_API_URL", "").rstrip("/")
DIFY_API_KEY = os.getenv("DIFY_API_KEY", "")
DIFY_CONSOLE_URL = os.getenv("DIFY_CONSOLE_URL", "").rstrip("/")

BROKEN_LOCAL_PROXY = "http://127.0.0.1:9"
PROXY_ENV_KEYS = ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY")
MODEL_CHECK_INTERVAL_SECONDS = 30


def _has_broken_local_proxy() -> bool:
    for key in PROXY_ENV_KEYS:
        value = (os.getenv(key) or "").strip().lower()
        if value.startswith(BROKEN_LOCAL_PROXY):
            return True
    return False


def _build_openai_client(api_key: str, base_url: str) -> OpenAI | None:
    if not api_key:
        return None
    # Some environments inject invalid localhost proxy settings.
    trust_env = not _has_broken_local_proxy()
    http_client = httpx.Client(trust_env=trust_env, timeout=120.0)
    return OpenAI(
        api_key=api_key,
        base_url=base_url,
        http_client=http_client
    )


def _build_model_clients() -> dict[str, dict]:
    clients: dict[str, dict] = {}
    deepseek_client = _build_openai_client(DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL)
    if deepseek_client:
        clients["deepseek"] = {
            "client": deepseek_client,
            "model": DEEPSEEK_MODEL,
            "base_url": DEEPSEEK_BASE_URL,
            "configured": True,
        }

    if FALLBACK_API_KEY and FALLBACK_MODEL and FALLBACK_BASE_URL:
        fallback_client = _build_openai_client(FALLBACK_API_KEY, FALLBACK_BASE_URL)
        if fallback_client:
            clients["fallback"] = {
                "client": fallback_client,
                "model": FALLBACK_MODEL,
                "base_url": FALLBACK_BASE_URL,
                "configured": True,
            }
    return clients


MODEL_RUNTIME: dict[str, Any] = {
    "clients": _build_model_clients(),
    "active_provider": None,
    "active_model": None,
    "model_ready": False,
    "last_check_ts": 0.0,
    "last_error": "",
    "provider_status": {},
}


def _ping_model_client(client: OpenAI, model: str) -> tuple[bool, str]:
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "ping"}],
            temperature=0,
            max_tokens=8,
        )
        content = (response.choices[0].message.content or "").strip()
        return True, content or "ok"
    except Exception as e:
        return False, str(e)


def _refresh_model_runtime(force: bool = False):
    now = time.time()
    if (not force) and MODEL_RUNTIME["last_check_ts"] and (now - MODEL_RUNTIME["last_check_ts"] < MODEL_CHECK_INTERVAL_SECONDS):
        return

    MODEL_RUNTIME["last_check_ts"] = now
    MODEL_RUNTIME["active_provider"] = None
    MODEL_RUNTIME["active_model"] = None
    MODEL_RUNTIME["model_ready"] = False
    MODEL_RUNTIME["last_error"] = ""
    provider_status: dict[str, Any] = {}

    provider_order = ["deepseek", "fallback"]
    for provider in provider_order:
        provider_info = MODEL_RUNTIME["clients"].get(provider)
        if not provider_info:
            provider_status[provider] = {"configured": False, "ready": False, "message": "not_configured"}
            continue
        ok, message = _ping_model_client(provider_info["client"], provider_info["model"])
        provider_status[provider] = {
            "configured": True,
            "ready": ok,
            "message": message,
            "model": provider_info["model"],
            "base_url": provider_info["base_url"],
        }
        if ok and not MODEL_RUNTIME["active_provider"]:
            MODEL_RUNTIME["active_provider"] = provider
            MODEL_RUNTIME["active_model"] = provider_info["model"]
            MODEL_RUNTIME["model_ready"] = True

    if not MODEL_RUNTIME["model_ready"]:
        errors = [f"{name}: {info.get('message', 'unavailable')}" for name, info in provider_status.items() if info.get("configured")]
        MODEL_RUNTIME["last_error"] = "; ".join(errors) if errors else "未配置任何可用模型"
    MODEL_RUNTIME["provider_status"] = provider_status


def _get_active_model_client(force_refresh: bool = False) -> tuple[OpenAI | None, str | None]:
    _refresh_model_runtime(force=force_refresh)
    if not MODEL_RUNTIME["model_ready"]:
        return None, None
    provider = MODEL_RUNTIME["active_provider"]
    provider_info = MODEL_RUNTIME["clients"].get(provider, {})
    return provider_info.get("client"), provider_info.get("model")


def _model_required_error_payload() -> dict:
    _refresh_model_runtime(force=True)
    return {
        "error": "当前无可用模型，分析/生成已阻断。请先配置并确保至少一个模型可连通。",
        "code": "MODEL_UNAVAILABLE",
        "guidance": [
            "在 .env 配置 DEEPSEEK_API_KEY / DEEPSEEK_BASE_URL / DEEPSEEK_MODEL",
            "或配置 FALLBACK_API_KEY / FALLBACK_BASE_URL / FALLBACK_MODEL",
            "检查代理与网络；如存在无效代理请移除 HTTP_PROXY/HTTPS_PROXY",
            "配置后重启服务并访问 /health 确认 model_ready=true",
        ],
        "provider_status": MODEL_RUNTIME.get("provider_status", {}),
        "last_error": MODEL_RUNTIME.get("last_error", ""),
    }


def model_ready() -> bool:
    client, _ = _get_active_model_client()
    return client is not None


def call_llm_chat(messages: list[dict], temperature: float = 0.0, max_tokens: int | None = None) -> str:
    client, model = _get_active_model_client(force_refresh=True)
    if not client or not model:
        raise ValueError("MODEL_UNAVAILABLE")

    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens

    response = client.chat.completions.create(**kwargs)
    return (response.choices[0].message.content or "").strip()


_refresh_model_runtime(force=True)
deepseek_client = MODEL_RUNTIME["clients"].get("deepseek", {}).get("client")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
YAML_TEMPLATE_DIR = os.path.join(BASE_DIR, "yaml_templates")
GENERATED_DIR = os.path.join(BASE_DIR, "generated")
ARTIFACT_DIR = os.path.join(GENERATED_DIR, "artifacts")
PLANNING_MODE = "llm_v2"
PROMPT_VERSION = "v2.1"
SCHEMA_VERSION = "wf-spec.v2"
MAX_NODE_LIMIT = 24
DEFAULT_ALLOW_AUTOFIX_IFELSE = False
SAFE_TOOL_HOSTS = {
    "httpbin.org",
    "api.github.com",
    "example.com",
}
CODE_TEMPLATE_LIBRARY = {
    "clean_text": (
        "import re\n\n"
        "def main(input_text: str) -> dict:\n"
        "    text = re.sub(r\"\\s+\", \" \", (input_text or \"\")).strip()\n"
        "    return {\"result\": text}\n"
    ),
    "parse_json": (
        "import json\n\n"
        "def main(input_text: str) -> dict:\n"
        "    data = json.loads(input_text)\n"
        "    return {\"result\": json.dumps(data, ensure_ascii=False, indent=2)}\n"
    ),
    "format_markdown": (
        "def main(title: str, body: str) -> dict:\n"
        "    title = (title or \"结果\").strip() or \"结果\"\n"
        "    body = (body or \"\").strip()\n"
        "    return {\"result\": f\"# {title}\\n\\n{body}\"}\n"
    ),
}

os.makedirs(GENERATED_DIR, exist_ok=True)
os.makedirs(ARTIFACT_DIR, exist_ok=True)

RUNTIME_METRICS = {
    "analyze_total": 0,
    "analyze_success": 0,
    "analyze_fail": 0,
    "generate_total": 0,
    "generate_success": 0,
    "generate_fail": 0,
    "compile_success": 0,
    "compile_fail": 0,
    "dify_import_success": 0,
    "dify_import_fail": 0,
    "analyze_latency_ms": [],
    "generate_latency_ms": [],
    "compile_latency_ms": [],
    "import_latency_ms": [],
    "non_llm_node_ratios": [],
    "spec_autofix_total": 0,
    "spec_autofix_ifelse_edges": 0,
    "spec_autofix_fail_branch_wraps": 0,
    "spec_autofix_aggregators": 0,
}

LEGACY_SCENE_CONFIG = {
    "summary": {
        "label": "文本总结助手",
        "questions": [
            "摘要风格是什么？（简短 / 详细）",
            "是否需要关键点列表？（是 / 否）"
        ],
        "nodes": ["Start", "Input Text", "LLM Summary", "End"],
        "template_file": "summary_template.yml"
    },
    "translate": {
        "label": "翻译助手",
        "questions": [
            "目标语言是什么？（例如：英文 / 中文 / 日文）",
            "是否需要自动识别源语言？（是 / 否）",
            "是否需要双语对照输出？（是 / 否）"
        ],
        "nodes": ["Start", "Input Text", "LLM Translate", "End"],
        "template_file": "translate_template.yml"
    },
    "qa": {
        "label": "通用问答助手",
        "questions": [
            "回答风格是什么？（简洁 / 详细）",
            "是否限制回答长度？（是 / 否）"
        ],
        "nodes": ["Start", "Input Question", "LLM QA", "End"],
        "template_file": "qa_template.yml"
    },
    "web_summary": {
        "label": "网页内容总结助手",
        "questions": [
            "输入方式是什么？（网页链接 / 粘贴网页内容）",
            "摘要风格是什么？（简短 / 详细）"
        ],
        "nodes": ["Start", "Input URL/Text", "LLM Web Summary", "End"],
        "template_file": "web_summary_template.yml"
    },
    "pdf_qa": {
        "label": "PDF 问答助手",
        "questions": [
            "用户是上传 PDF 还是粘贴文本片段？（上传 / 粘贴）",
            "回答风格是什么？（简洁 / 详细）"
        ],
        "nodes": ["Start", "Input File/Text", "Input Question", "LLM PDF QA", "End"],
        "template_file": "pdf_qa_template.yml"
    },
    "unknown": {
        "label": "未知场景",
        "questions": [
            "请再详细描述你的需求，例如：翻译、总结、问答、网页总结、PDF 问答。"
        ],
        "nodes": ["Start", "LLM", "End"],
        "template_file": "qa_template.yml"
    }
}
# 旧版全局会话状态，当前版本已不再依赖，仅暂时保留避免一次性删除过多代码
session_state = {
    "mode": "idle",
    "scene_key": None,
    "scene_label": None,
    "app_name": "",
    "description": "",
    "workflow_spec": None,
    "requirement": None
}


SCENE_CAPABILITY_DEFAULTS = {
    "summary": {
        "capabilities": [
            "intent_analysis",
            "generation",
            "formatting"
        ],
        "extra_inputs": [
            {"name": "content", "label": "原始内容", "type": "paragraph", "required": True}
        ]
    },
    "translation": {
        "capabilities": [
            "intent_analysis",
            "generation",
            "formatting"
        ],
        "extra_inputs": [
            {"name": "source_text", "label": "原文", "type": "paragraph", "required": True},
            {"name": "target_language", "label": "目标语言", "type": "text", "required": True}
        ]
    }
}
SUPPORTED_STEP_TYPES = {
    "start",
    "llm",
    "ifelse",
    "template",
    "answer",
    "code",
    "knowledge_retrieval",
    "tool",
    "parameter_extract",
    "variable_aggregator",
    "iteration",
    "loop",
}


def test_deepseek_connection() -> str:
    _refresh_model_runtime(force=True)
    if not MODEL_RUNTIME.get("clients", {}).get("deepseek"):
        return "未检测到 DeepSeek 配置"
    status = MODEL_RUNTIME.get("provider_status", {}).get("deepseek", {})
    if status.get("ready"):
        return "DeepSeek 连接成功"
    return f"DeepSeek 调用失败: {status.get('message', 'unknown')}"


def get_proxy_env_snapshot() -> dict:
    snapshot = {}
    for key in PROXY_ENV_KEYS:
        value = os.getenv(key)
        if value:
            snapshot[key] = value
    return snapshot


def get_asset_version() -> str:
    app_js = os.path.join(BASE_DIR, "static", "app.js")
    style_css = os.path.join(BASE_DIR, "static", "style.css")
    newest = max(
        os.path.getmtime(app_js) if os.path.exists(app_js) else 0,
        os.path.getmtime(style_css) if os.path.exists(style_css) else 0,
    )
    return str(int(newest))


def _strip_json_code_fence(content: str) -> str:
    text = (content or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE).strip()
        text = re.sub(r"```$", "", text).strip()
    return text


def _safe_json_load(content: str) -> Any:
    text = _strip_json_code_fence(content)
    try:
        return json_repair.loads(text)
    except Exception:
        match = re.search(r"\{[\s\S]*\}|\[[\s\S]*\]", text)
        if not match:
            raise
        return json_repair.loads(match.group(0))

def _safe_json_load_dict(content: str) -> dict:
    data = _safe_json_load(content)
    if isinstance(data, dict):
        return data
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                return item
    text = _strip_json_code_fence(content)
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        obj = json_repair.loads(match.group(0))
        if isinstance(obj, dict):
            return obj
    raise ValueError("模型未返回对象结构")


def _coerce_scene(scene: str) -> str:
    allowed = {"summary", "translation", "qa", "web_summary", "generic"}
    if scene == "xiaohongshu":
        return "generic"
    if scene not in allowed:
        return "generic"
    return scene


def _normalize_requirement_v2(requirement: dict, user_input: str) -> dict:
    if not isinstance(requirement, dict):
        requirement = {}

    result = copy.deepcopy(requirement)
    result.setdefault("app_name", "通用工作流")
    result.setdefault("description", "由模型结构化解析的工作流需求")
    result["scene"] = _coerce_scene(result.get("scene", "generic"))
    result.setdefault("user_goal", user_input)
    result.setdefault("inputs", [])
    result.setdefault("output_contract", {})
    result.setdefault("capabilities", [])
    result.setdefault("node_preferences", {})
    result.setdefault("runtime_config", {})
    result.setdefault("constraints", [])

    if not isinstance(result["inputs"], list):
        result["inputs"] = []
    if not result["inputs"]:
        result["inputs"] = [
            {"name": "user_request", "label": "用户输入", "type": "paragraph", "required": True}
        ]

    if not isinstance(result["capabilities"], list):
        result["capabilities"] = []

    node_preferences = result["node_preferences"] if isinstance(result["node_preferences"], dict) else {}
    target_node_count = node_preferences.get("target_node_count", 8)
    try:
        target_node_count = int(target_node_count)
    except Exception:
        target_node_count = 8
    node_preferences["target_node_count"] = max(5, min(MAX_NODE_LIMIT, target_node_count))
    node_preferences.setdefault("need_private_kb", False)
    node_preferences.setdefault("need_code_node", False)
    node_preferences.setdefault("need_tools", False)
    node_preferences.setdefault("need_branching", False)
    result["node_preferences"] = node_preferences

    runtime_config = result["runtime_config"] if isinstance(result["runtime_config"], dict) else {}
    runtime_config.setdefault("model_tier", DEEPSEEK_MODEL)
    runtime_config.setdefault("allow_autofix_ifelse", DEFAULT_ALLOW_AUTOFIX_IFELSE)
    result["runtime_config"] = runtime_config

    if not isinstance(result["constraints"], list):
        result["constraints"] = []

    return result


def _estimate_complexity_level(user_input: str, requirement: dict) -> str:
    score = 0
    text = user_input.lower()
    if len(user_input) > 120:
        score += 1
    if any(k in text for k in ["分支", "如果", "判断", "并行", "多工具", "复杂"]):
        score += 2

    caps = requirement.get("capabilities", [])
    for cap in ("retrieval", "private_kb", "tool_call", "code_execution", "branching"):
        if cap in caps:
            score += 1

    target = requirement.get("node_preferences", {}).get("target_node_count", 8)
    if isinstance(target, int) and target >= 10:
        score += 1

    return "complex" if score >= 3 else "simple"


def _candidate_count_for_level(level: str) -> int:
    return 3 if level == "complex" else 1


def llm_structure_requirement_v2(user_input: str) -> dict:
    if not model_ready():
        raise ValueError("MODEL_UNAVAILABLE")

    system_prompt = """
你是工作流需求结构化助手。输出必须是 JSON。
不要输出 markdown，不要输出解释。
字段必须包含：
app_name, description, scene, user_goal, inputs, output_contract, capabilities, node_preferences, runtime_config, constraints

scene 允许值：
summary, translation, qa, web_summary, generic

capabilities 允许值：
input_check, intent_analysis, planning, retrieval, code_execution, tool_call, branching, generation, optimization, formatting, multi_title, private_kb

inputs 每项字段：
name, label, type(text|paragraph|file), required

node_preferences 字段：
target_node_count, need_private_kb, need_code_node, need_tools, need_branching
"""

    content = call_llm_chat(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input},
        ],
        temperature=0,
    )
    parsed = _safe_json_load_dict(content)
    return _normalize_requirement_v2(parsed, user_input)


def llm_plan_workflow_candidate_v2(requirement: dict, temperature: float = 0.25) -> dict:
    if not model_ready():
        raise ValueError("MODEL_UNAVAILABLE")

    planner_prompt = f"""
你是工作流拓扑规划器。根据 requirement 生成 workflow spec，返回 JSON。
禁止输出 markdown 或解释。

支持的节点类型：
{sorted(list(SUPPORTED_STEP_TYPES))}

输出 JSON 字段：
workflow_name, description, scene, inputs, output_contract, steps, edges

约束：
1) 必须包含一个 start 节点和一个 answer 节点。图必须连通，且逻辑严谨、符合实际业务需求。
2) edges 里的 source/target 必须引用 steps 里的 id。避免生成无意义的悬空节点。
2.1) ifelse 的分支连边用 edges[].branch 表达，branch 值应与 steps[].config.cases[].id 对齐
2.2) code/tool 如需失败兜底，可设置 steps[].config.error_strategy="fail-branch"，并在 edges 中用 edges[].source_handle 指定 success-branch / fail-branch。⚠️警告：请克制使用 fail-branch，仅在极核心的风险节点使用，避免图结构过度臃肿！
2.3) 若步骤中包含 llm_plan 且后面存在 ifelse_route，请让 llm_plan 输出可稳定匹配的 ROUTE=knowledge/compute/tool/direct 标记
3) steps 至少 3 个，最多 {MAX_NODE_LIMIT} 个。请保持拓扑结构简单清晰。
4) 如使用 tool 节点，config.url 只能是以下主机之一：{sorted(list(SAFE_TOOL_HOSTS))}
5) tool 节点 config 建议包含：method/url/headers/body/variables
6) code 节点 config 建议包含：language/script/example_key/variables
7) code.example_key 可选值：{sorted(list(CODE_TEMPLATE_LIBRARY.keys()))}
8) 不要把所有逻辑都压成 llm，合理分配给 knowledge_retrieval / code / tool / ifelse。
9) 当存在多分支汇合时，使用 variable_aggregator 做汇合，再连接到后续节点。
10) 当使用 code 节点时，必须通过 variables 声明前置依赖，并在 script 中引用。

【节点与连边配置格式示例 (仅为局部片段，请根据实际需求生成完整流程)】:
{{
  "workflow_name": "根据需求命名",
  "steps": [
    {{"id": "start", "type": "start", "title": "开始"}},
    {{"id": "code_1", "type": "code", "title": "数据处理", "config": {{"variables": [{{"variable": "input", "value_selector": ["start", "input"]}}]}}}},
    {{"id": "ifelse_1", "type": "ifelse", "title": "分支判断", "config": {{"cases": [{{"id": "case_1", "logical_operator": "and", "conditions": []}}]}}}}
  ],
  "edges": [
    {{"source": "start", "target": "code_1"}},
    {{"source": "code_1", "target": "ifelse_1"}}
  ]
}}

requirement:
{json.dumps(requirement, ensure_ascii=False)}
"""

    content = call_llm_chat(
        messages=[{"role": "system", "content": planner_prompt}],
        temperature=temperature,
    )
    parsed = _safe_json_load_dict(content)
    parsed.setdefault("workflow_name", requirement.get("app_name", "通用工作流"))
    parsed.setdefault("description", requirement.get("description", "模型规划结果"))
    parsed["scene"] = _coerce_scene(parsed.get("scene", requirement.get("scene", "generic")))
    parsed.setdefault("inputs", requirement.get("inputs", []))
    parsed.setdefault("output_contract", requirement.get("output_contract", {}))
    parsed.setdefault("steps", [])
    parsed.setdefault("edges", [])
    return parsed


def _tool_url_is_safe(url: str) -> bool:
    if not isinstance(url, str) or not url:
        return False
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        return False
    host = (parsed.netloc or "").split(":")[0].lower()
    return host in SAFE_TOOL_HOSTS


def _tool_headers_are_safe(headers: Any) -> bool:
    if headers is None:
        return True
    blocked_keys = {"authorization", "cookie", "x-api-key", "proxy-authorization"}
    if isinstance(headers, dict):
        for k in headers.keys():
            if str(k).strip().lower() in blocked_keys:
                return False
        return True
    if isinstance(headers, str):
        for line in headers.splitlines():
            if ":" not in line:
                continue
            key = line.split(":", 1)[0].strip().lower()
            if key in blocked_keys:
                return False
        return True
    return False


def _has_path(start_ids: list[str], answer_ids: list[str], edges: list[dict]) -> bool:
    if not start_ids or not answer_ids:
        return False
    adj: dict[str, list[str]] = {}
    for edge in edges:
        adj.setdefault(edge["source"], []).append(edge["target"])
    target_set = set(answer_ids)
    stack = list(start_ids)
    visited = set()
    while stack:
        node = stack.pop()
        if node in target_set:
            return True
        if node in visited:
            continue
        visited.add(node)
        stack.extend(adj.get(node, []))
    return False


def _has_cycle(step_ids: set[str], edges: list[dict]) -> bool:
    adj: dict[str, list[str]] = {sid: [] for sid in step_ids}
    for edge in edges:
        adj.setdefault(edge["source"], []).append(edge["target"])
    color: dict[str, int] = {sid: 0 for sid in step_ids}

    def dfs(node: str) -> bool:
        color[node] = 1
        for nxt in adj.get(node, []):
            if color.get(nxt, 0) == 1:
                return True
            if color.get(nxt, 0) == 0 and dfs(nxt):
                return True
        color[node] = 2
        return False

    for sid in step_ids:
        if color[sid] == 0 and dfs(sid):
            return True
    return False


def validate_workflow_spec_v2(spec: dict, max_nodes: int = MAX_NODE_LIMIT) -> dict:
    errors: list[str] = []
    warnings: list[str] = []
    type_counts: dict[str, int] = {}

    if not isinstance(spec, dict):
        return {"ok": False, "errors": ["spec 不是对象"], "warnings": [], "stats": {}}

    steps = spec.get("steps", [])
    edges = spec.get("edges", [])

    if not isinstance(steps, list) or not steps:
        errors.append("steps 不能为空")
        steps = []
    if not isinstance(edges, list):
        errors.append("edges 必须是数组")
        edges = []

    if len(steps) > max_nodes:
        errors.append(f"节点数超限: {len(steps)} > {max_nodes}")

    ids: list[str] = []
    step_ids: set[str] = set()
    for idx, step in enumerate(steps):
        if not isinstance(step, dict):
            errors.append(f"step[{idx}] 结构非法")
            continue
        sid = step.get("id")
        stype = step.get("type")
        if not sid or not isinstance(sid, str):
            errors.append(f"step[{idx}] 缺少 id")
            continue
        if sid in step_ids:
            errors.append(f"节点 id 重复: {sid}")
            continue
        step_ids.add(sid)
        ids.append(sid)
        if stype not in SUPPORTED_STEP_TYPES:
            errors.append(f"节点 {sid} type 不支持: {stype}")
        type_counts[stype] = type_counts.get(stype, 0) + 1

        if stype == "tool":
            config = step.get("config", {}) or {}
            method = str(config.get("method", "get")).lower().strip()
            if method not in {"get", "post", "put", "delete", "patch"}:
                errors.append(f"工具节点 {sid} method 非法: {method}")
            if not _tool_url_is_safe(config.get("url", "")):
                errors.append(f"工具节点 {sid} 使用了不安全 url")
            if not _tool_headers_are_safe(config.get("headers")):
                errors.append(f"工具节点 {sid} headers 包含敏感头")
        if stype == "code":
            config = step.get("config", {}) or {}
            if isinstance(config, dict):
                language = str(config.get("language", "python3")).strip().lower()
                if language not in {"python3", "python", "javascript", "js"}:
                    warnings.append(f"代码节点 {sid} language={language} 可能不被 Dify 支持")

    start_ids = [s["id"] for s in steps if isinstance(s, dict) and s.get("type") == "start" and s.get("id")]
    answer_ids = [s["id"] for s in steps if isinstance(s, dict) and s.get("type") == "answer" and s.get("id")]
    if not start_ids:
        errors.append("缺少 start 节点")
    if not answer_ids:
        errors.append("缺少 answer 节点")
    if len(start_ids) > 1:
        warnings.append("存在多个 start 节点")
    if len(answer_ids) > 1:
        warnings.append("存在多个 answer 节点")

    step_type_map = {
        s.get("id"): s.get("type")
        for s in steps
        if isinstance(s, dict) and s.get("id")
    }
    seen_edges = set()
    valid_edges: list[dict] = []
    for idx, edge in enumerate(edges):
        if not isinstance(edge, dict):
            errors.append(f"edge[{idx}] 结构非法")
            continue
        source = edge.get("source")
        target = edge.get("target")
        if source not in step_ids or target not in step_ids:
            errors.append(f"edge[{idx}] 引用了不存在节点: {source}->{target}")
            continue
        if source == target:
            errors.append(f"edge[{idx}] source/target 不能相同: {source}")
            continue
        source_type = step_type_map.get(source)
        if source_type == "ifelse":
            if edge.get("source_handle") and not edge.get("branch"):
                errors.append(f"edge[{idx}] ifelse 源节点请使用 branch，不要使用 source_handle")
        else:
            if edge.get("branch") and not edge.get("source_handle"):
                branch = str(edge.get("branch"))
                if branch in {"success-branch", "fail-branch"}:
                    errors.append(f"edge[{idx}] 非 ifelse 节点请使用 source_handle 表达 {branch}")
                else:
                    warnings.append(f"edge[{idx}] 非 ifelse 节点存在 branch={branch}，建议改为 source_handle")
        key = (source, target, edge.get("branch"), edge.get("source_handle"))
        if key in seen_edges:
            warnings.append(f"重复 edge: {source}->{target}")
            continue
        seen_edges.add(key)
        valid_edges.append(edge)

    incoming_map: dict[str, list[dict]] = {}
    for edge in valid_edges:
        incoming_map.setdefault(edge.get("target"), []).append(edge)

    for step in steps:
        if not isinstance(step, dict):
            continue
        sid = step.get("id")
        stype = step.get("type")
        if not sid or sid not in step_ids:
            continue

        if stype == "ifelse":
            incoming = incoming_map.get(sid, [])
            if len(incoming) != 1:
                errors.append(f"ifelse 节点 {sid} 应该且只能有 1 条入边，当前 {len(incoming)}")
            cfg = step.get("config", {}) or {}
            raw_cases = cfg.get("cases", [])
            if isinstance(raw_cases, list) and raw_cases:
                case_ids = []
                for c in raw_cases:
                    if isinstance(c, dict):
                        case_id = c.get("id") or c.get("case_id")
                        if case_id:
                            case_ids.append(str(case_id))
                if case_ids:
                    outgoing = [e for e in valid_edges if e.get("source") == sid]
                    branch_set = {str(e.get("branch")) for e in outgoing if e.get("branch")}
                    for case_id in case_ids:
                        if case_id not in branch_set:
                            errors.append(f"ifelse 节点 {sid} 缺少分支连边: {case_id}")

        if stype in {"tool", "code"}:
            cfg = step.get("config", {}) or {}
            if isinstance(cfg, dict) and cfg.get("error_strategy") == "fail-branch":
                outgoing = [e for e in valid_edges if e.get("source") == sid]
                handles = {str(e.get("source_handle") or e.get("branch") or "source") for e in outgoing}
                if "success-branch" not in handles:
                    errors.append(f"{stype} 节点 {sid} error_strategy=fail-branch 但缺少 success-branch 连边")
                if "fail-branch" not in handles:
                    errors.append(f"{stype} 节点 {sid} error_strategy=fail-branch 但缺少 fail-branch 连边")

        if stype == "answer":
            incoming = incoming_map.get(sid, [])
            if len(incoming) > 1:
                errors.append(f"answer 节点 {sid} 存在多入边，需先通过 variable_aggregator 汇合")

    if step_ids and _has_cycle(step_ids, valid_edges):
        errors.append("workflow 存在循环依赖")
    if start_ids and answer_ids and not _has_path(start_ids, answer_ids, valid_edges):
        errors.append("start 到 answer 不可达")

    llm_count = type_counts.get("llm", 0)
    total = max(1, len(steps))
    non_llm_ratio = round((total - llm_count) / total, 4)
    stats = {
        "total_nodes": len(steps),
        "type_counts": type_counts,
        "non_llm_node_ratio": non_llm_ratio,
        "has_path_start_to_answer": _has_path(start_ids, answer_ids, valid_edges) if start_ids and answer_ids else False,
    }

    return {
        "ok": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "stats": stats,
    }


def deterministic_fix_workflow_spec_v2(spec: dict) -> tuple[dict, list[str]]:
    warnings: list[str] = []
    fixed = copy.deepcopy(spec if isinstance(spec, dict) else {})
    steps = fixed.get("steps", [])
    edges = fixed.get("edges", [])
    if not isinstance(steps, list):
        steps = []
    if not isinstance(edges, list):
        edges = []

    cleaned_steps = []
    seen = set()
    for step in steps:
        if not isinstance(step, dict):
            continue
        sid = step.get("id")
        stype = step.get("type")
        if not sid:
            sid = f"node_{uuid.uuid4().hex[:8]}"
            step["id"] = sid
            warnings.append(f"补全缺失 id: {sid}")
        if sid in seen:
            warnings.append(f"删除重复节点: {sid}")
            continue
        seen.add(sid)
        if stype not in SUPPORTED_STEP_TYPES:
            step["type"] = "llm"
            warnings.append(f"节点 {sid} 的非法 type 已改为 llm")
        if step["type"] == "tool":
            config = step.get("config", {}) or {}
            method = str(config.get("method", "get")).lower().strip()
            if method not in {"get", "post", "put", "delete", "patch"}:
                method = "get"
                warnings.append(f"工具节点 {sid} method 非法，已改为 get")
            config["method"] = method
            url = config.get("url", "")
            if not _tool_url_is_safe(url):
                config["url"] = "https://httpbin.org/get"
                warnings.append(f"工具节点 {sid} url 已重写为白名单地址")
            if not _tool_headers_are_safe(config.get("headers")):
                config["headers"] = ""
                warnings.append(f"工具节点 {sid} headers 含敏感字段，已清空")
            step["config"] = config
        if step["type"] == "code":
            config = step.get("config", {}) or {}
            if not isinstance(config, dict):
                config = {}
            if not config.get("script") and not config.get("example_key"):
                config["example_key"] = "clean_text"
            step["config"] = config
        cleaned_steps.append(step)
        if len(cleaned_steps) >= MAX_NODE_LIMIT:
            warnings.append("节点数超限，已裁剪到上限")
            break

    step_ids = {s["id"] for s in cleaned_steps if isinstance(s, dict) and s.get("id")}
    if not any(s.get("type") == "start" for s in cleaned_steps):
        cleaned_steps.insert(0, {"id": "start", "type": "start", "title": "开始"})
        step_ids.add("start")
        warnings.append("补充 start 节点")
    if not any(s.get("type") == "answer" for s in cleaned_steps):
        cleaned_steps.append({"id": "answer", "type": "answer", "title": "直接回复"})
        step_ids.add("answer")
        warnings.append("补充 answer 节点")
    if len(cleaned_steps) > MAX_NODE_LIMIT:
        while len(cleaned_steps) > MAX_NODE_LIMIT:
            removed = False
            for idx in range(len(cleaned_steps) - 1, -1, -1):
                t = cleaned_steps[idx].get("type") if isinstance(cleaned_steps[idx], dict) else None
                if t not in {"start", "answer"}:
                    cleaned_steps.pop(idx)
                    removed = True
                    break
            if not removed:
                break
        step_ids = {s["id"] for s in cleaned_steps if isinstance(s, dict) and s.get("id")}

    cleaned_edges = []
    seen_edge = set()
    for edge in edges:
        if not isinstance(edge, dict):
            continue
        source = edge.get("source")
        target = edge.get("target")
        if source not in step_ids or target not in step_ids:
            continue
        key = (source, target, edge.get("branch"))
        if key in seen_edge:
            continue
        seen_edge.add(key)
        cleaned_edges.append({"source": source, "target": target, **({"branch": edge["branch"]} if edge.get("branch") else {})})

    if not cleaned_edges:
        # fallback linear chain only if edges are empty; keeps user topology otherwise.
        ids = [s["id"] for s in cleaned_steps]
        for i in range(len(ids) - 1):
            cleaned_edges.append({"source": ids[i], "target": ids[i + 1]})
        warnings.append("原始 edges 为空，已按节点顺序自动连边")

    fixed["steps"] = cleaned_steps
    fixed["edges"] = cleaned_edges
    return fixed, warnings


def _sorted_edges(edges: list[dict]) -> list[dict]:
    def key(e: dict):
        return (
            str(e.get("source") or ""),
            str(e.get("target") or ""),
            str(e.get("branch") or ""),
            str(e.get("source_handle") or ""),
        )

    return sorted([e for e in edges if isinstance(e, dict)], key=key)


def _pick_default_target_id(spec: dict) -> str | None:
    steps = spec.get("steps", []) if isinstance(spec, dict) else []
    if not isinstance(steps, list):
        return None
    existing = {s.get("id") for s in steps if isinstance(s, dict) and s.get("id")}
    for preferred in ("llm_generate", "template_output", "llm_review", "answer"):
        if preferred in existing:
            return preferred
    llm_ids = [s.get("id") for s in steps if isinstance(s, dict) and s.get("type") == "llm" and s.get("id")]
    return llm_ids[-1] if llm_ids else None


def _normalize_edge_semantics(spec: dict) -> list[str]:
    warnings: list[str] = []
    steps = spec.get("steps", [])
    edges = spec.get("edges", [])
    if not isinstance(steps, list) or not isinstance(edges, list):
        return warnings
    step_type_map = {
        s.get("id"): s.get("type")
        for s in steps
        if isinstance(s, dict) and s.get("id")
    }
    for edge in edges:
        if not isinstance(edge, dict):
            continue
        source = edge.get("source")
        stype = step_type_map.get(source)
        branch = edge.get("branch")
        source_handle = edge.get("source_handle")
        if stype == "ifelse":
            if source_handle and (not branch):
                edge["branch"] = str(source_handle)
                edge.pop("source_handle", None)
                warnings.append(f"ifelse 边 {source}->{edge.get('target')} 使用 source_handle，已归一化为 branch")
            elif source_handle and branch and str(source_handle) != str(branch):
                edge.pop("source_handle", None)
                warnings.append(f"ifelse 边 {source}->{edge.get('target')} 的 source_handle/branch 冲突，已保留 branch")
        else:
            if branch and str(branch) in {"success-branch", "fail-branch"} and (not source_handle):
                edge["source_handle"] = str(branch)
                edge.pop("branch", None)
                warnings.append(f"节点 {source} 的 fail-branch 边已归一化到 source_handle")
    spec["edges"] = _sorted_edges(edges)
    return warnings


def _ensure_ifelse_branch_edges(spec: dict, allow_autofix: bool = False) -> list[str]:
    warnings: list[str] = []
    steps = spec.get("steps", [])
    edges = spec.get("edges", [])
    if not isinstance(steps, list) or not isinstance(edges, list):
        return warnings

    default_target = _pick_default_target_id(spec)
    if not default_target:
        return warnings

    step_ids = {s.get("id") for s in steps if isinstance(s, dict) and s.get("id")}
    for step in steps:
        if not isinstance(step, dict) or step.get("type") != "ifelse":
            continue
        sid = step.get("id")
        if not sid or sid not in step_ids:
            continue
        cfg = step.get("config", {}) or {}
        raw_cases = cfg.get("cases", [])
        if not isinstance(raw_cases, list) or not raw_cases:
            continue
        case_ids = []
        for c in raw_cases:
            if isinstance(c, dict):
                case_id = c.get("id") or c.get("case_id")
                if case_id:
                    case_ids.append(str(case_id))

        if not case_ids:
            continue
            
        if "false" not in case_ids:
            case_ids.append("false")

        outgoing = [e for e in edges if isinstance(e, dict) and e.get("source") == sid]
        existing_branches = {str(e.get("branch")) for e in outgoing if e.get("branch")}
        for case_id in case_ids:
            if case_id in existing_branches:
                continue
            if allow_autofix:
                edges.append({"source": sid, "target": default_target, "branch": case_id})
                warnings.append(f"ifelse {sid} 缺少分支 {case_id}，已补默认连边")
            else:
                warnings.append(f"ifelse {sid} 缺少分支 {case_id}（严格模式未自动补边）")

    spec["edges"] = _sorted_edges(edges)
    return warnings


def _insert_variable_aggregators_for_conditional_merges(spec: dict) -> list[str]:
    warnings: list[str] = []
    steps = spec.get("steps", [])
    edges = spec.get("edges", [])
    if not isinstance(steps, list) or not isinstance(edges, list):
        return warnings

    step_by_id = {s.get("id"): s for s in steps if isinstance(s, dict) and s.get("id")}
    incoming: dict[str, list[dict]] = {}
    for e in edges:
        if not isinstance(e, dict):
            continue
        tgt = e.get("target")
        if not tgt:
            continue
        incoming.setdefault(tgt, []).append(e)

    created = 0
    for target_id, inc_edges in list(incoming.items()):
        if len(inc_edges) <= 1:
            continue
        target_step = step_by_id.get(target_id)
        if not isinstance(target_step, dict):
            continue
        if target_step.get("type") not in {"llm", "template", "answer"}:
            continue
        if not any(e.get("branch") for e in inc_edges):
            continue

        agg_id = _new_id("agg")
        if len(steps) + 1 > MAX_NODE_LIMIT:
            warnings.append(f"节点数接近上限，跳过为 {target_id} 插入 variable_aggregator")
            continue
        created += 1
        step_by_id[agg_id] = {"id": agg_id, "type": "variable_aggregator", "title": "分支汇合", "config": {}}
        steps.append(step_by_id[agg_id])

        new_edges: list[dict] = []
        for e in edges:
            if not isinstance(e, dict):
                continue
            if e.get("target") == target_id:
                continue
            new_edges.append(e)

        for e in inc_edges:
            ne = {"source": e.get("source"), "target": agg_id}
            if e.get("branch"):
                ne["branch"] = e.get("branch")
            if e.get("source_handle"):
                ne["source_handle"] = e.get("source_handle")
            new_edges.append(ne)

        new_edges.append({"source": agg_id, "target": target_id})
        edges = new_edges
        warnings.append(f"检测到 {target_id} 存在多分支汇入，已插入 variable_aggregator={agg_id}")

    spec["steps"] = steps
    spec["edges"] = _sorted_edges(edges)
    if created:
        spec.setdefault("meta", {})
        spec["meta"]["inserted_aggregators"] = created
    return warnings


def _wrap_exec_node_with_fail_branch(spec: dict) -> list[str]:
    warnings: list[str] = []
    steps = spec.get("steps", [])
    edges = spec.get("edges", [])
    if not isinstance(steps, list) or not isinstance(edges, list):
        return warnings

    step_by_id = {s.get("id"): s for s in steps if isinstance(s, dict) and s.get("id")}
    step_type_map = {sid: step_by_id[sid].get("type") for sid in step_by_id}

    outgoing: dict[str, list[dict]] = {}
    for e in edges:
        if not isinstance(e, dict):
            continue
        src = e.get("source")
        if not src:
            continue
        outgoing.setdefault(src, []).append(e)

    new_edges = [e for e in edges if isinstance(e, dict)]

    for step in list(steps):
        if not isinstance(step, dict):
            continue
        stype = step.get("type")
        if stype not in {"tool", "code"}:
            continue
        sid = step.get("id")
        if not sid:
            continue
        cfg = step.get("config", {}) if isinstance(step.get("config"), dict) else {}
        error_strategy = cfg.get("error_strategy")
        if error_strategy != "fail-branch":
            continue
        outs = outgoing.get(sid, [])
        if not outs:
            if error_strategy == "fail-branch":
                cfg.pop("error_strategy", None)
                step["config"] = cfg
            continue
        if error_strategy == "fail-branch":
            handles = {str(e.get("source_handle") or e.get("branch") or "source") for e in outs if isinstance(e, dict)}
            if "success-branch" in handles or "fail-branch" in handles:
                # 只要 LLM 已经生成了任何相关的分支 handle，我们就不再自动包装，避免冲突
                continue
        target_ids = [e.get("target") for e in outs if e.get("target")]
        if not target_ids:
            continue
        primary_target = None
        for tid in target_ids:
            if step_type_map.get(tid) in {"llm", "template", "answer"}:
                primary_target = tid
                break
        if not primary_target:
            primary_target = target_ids[0]

        fallback_llm_id = _new_id("llm_fallback")
        agg_id = _new_id("agg")
        if len(steps) + 2 > MAX_NODE_LIMIT:
            if error_strategy == "fail-branch":
                cfg.pop("error_strategy", None)
                step["config"] = cfg
            warnings.append(f"节点数接近上限，跳过为 {sid} 插入 fail-branch 兜底链路")
            continue

        steps.append({
            "id": fallback_llm_id,
            "type": "llm",
            "title": "失败兜底",
            "prompt": "上游执行失败。请根据用户输入与已知上下文，给出可交付的替代结果，并说明关键假设。",
        })
        steps.append({
            "id": agg_id,
            "type": "variable_aggregator",
            "title": "结果汇合",
            "config": {},
        })

        cfg["error_strategy"] = "fail-branch"
        step["config"] = cfg

        new_edges = [e for e in new_edges if not (e.get("source") == sid and e.get("target") == primary_target)]
        new_edges.append({"source": sid, "target": agg_id, "source_handle": "success-branch"})
        new_edges.append({"source": sid, "target": fallback_llm_id, "source_handle": "fail-branch"})
        new_edges.append({"source": fallback_llm_id, "target": agg_id})
        new_edges.append({"source": agg_id, "target": primary_target})
        
        # Remove any other unhandled edges from sid, and route them from agg_id instead
        other_unhandled_outs = [e for e in new_edges if e.get("source") == sid and not e.get("source_handle")]
        new_edges = [e for e in new_edges if e not in other_unhandled_outs]
        for oe in other_unhandled_outs:
            new_edges.append({"source": agg_id, "target": oe.get("target")})
            
        print(f"DEBUG: other_unhandled_outs for {sid}: {other_unhandled_outs}", flush=True)

        warnings.append(f"为 {sid} 启用 fail-branch 并插入兜底链路（{fallback_llm_id}->{agg_id}->{primary_target}）")

    spec["steps"] = steps
    spec["edges"] = _sorted_edges(new_edges)
    return warnings


def _prune_redundant_nodes(spec: dict) -> list[str]:
    warnings: list[str] = []
    steps = spec.get("steps", [])
    edges = spec.get("edges", [])
    if not isinstance(steps, list) or not isinstance(edges, list):
        return warnings

    step_by_id = {s.get("id"): s for s in steps if isinstance(s, dict) and s.get("id")}
    incoming: dict[str, list[dict]] = {}
    outgoing: dict[str, list[dict]] = {}

    for e in edges:
        if not isinstance(e, dict):
            continue
        incoming.setdefault(e.get("target"), []).append(e)
        outgoing.setdefault(e.get("source"), []).append(e)

    nodes_to_remove = set()
    for step in steps:
        if not isinstance(step, dict):
            continue
        sid = step.get("id")
        if not sid or step.get("type") != "llm":
            continue

        prompt = step.get("prompt", "").strip()
        # If it's a simple passthrough node
        if len(prompt) < 15 and ("透传" in prompt or "直接输出" in prompt or prompt == "请根据用户输入完成任务。"):
            inc = incoming.get(sid, [])
            out = outgoing.get(sid, [])
            if len(inc) == 1 and len(out) == 1:
                # We can safely prune this node
                nodes_to_remove.add(sid)

    if not nodes_to_remove:
        return warnings

    new_steps = [s for s in steps if s.get("id") not in nodes_to_remove]
    new_edges = []
    for e in edges:
        if not isinstance(e, dict):
            continue
        if e.get("source") in nodes_to_remove or e.get("target") in nodes_to_remove:
            continue
        new_edges.append(e)

    for sid in nodes_to_remove:
        inc = incoming.get(sid)[0]
        out = outgoing.get(sid)[0]
        new_edge = {
            "source": inc.get("source"),
            "target": out.get("target")
        }
        if inc.get("branch"):
            new_edge["branch"] = inc.get("branch")
        if inc.get("source_handle"):
            new_edge["source_handle"] = inc.get("source_handle")
        new_edges.append(new_edge)
        warnings.append(f"已修剪冗余透传节点: {sid}")

    spec["steps"] = new_steps
    spec["edges"] = _sorted_edges(new_edges)
    return warnings

def normalize_workflow_spec_v2(spec: dict, requirement: dict | None = None) -> tuple[dict, list[str]]:
    warnings: list[str] = []
    normalized = copy.deepcopy(spec if isinstance(spec, dict) else {})
    if not isinstance(normalized.get("steps"), list):
        normalized["steps"] = []
    if not isinstance(normalized.get("edges"), list):
        normalized["edges"] = []

    runtime_cfg = {}
    if isinstance(requirement, dict) and isinstance(requirement.get("runtime_config"), dict):
        runtime_cfg = requirement.get("runtime_config", {})
    allow_autofix_ifelse = bool(runtime_cfg.get("allow_autofix_ifelse", DEFAULT_ALLOW_AUTOFIX_IFELSE))

    warnings.extend(_prune_redundant_nodes(normalized))
    warnings.extend(_normalize_edge_semantics(normalized))
    warnings.extend(_ensure_ifelse_branch_edges(normalized, allow_autofix=allow_autofix_ifelse))
    warnings.extend(_wrap_exec_node_with_fail_branch(normalized))
    warnings.extend(_insert_variable_aggregators_for_conditional_merges(normalized))
    normalized["edges"] = _sorted_edges(normalized.get("edges", []))
    normalized.setdefault("meta", {})
    normalized["meta"]["allow_autofix_ifelse"] = allow_autofix_ifelse
    autofix_warnings = [w for w in warnings if "已" in w]
    normalized["meta"]["autofix_warnings"] = len(autofix_warnings)
    if autofix_warnings:
        RUNTIME_METRICS["spec_autofix_total"] += len(autofix_warnings)
        RUNTIME_METRICS["spec_autofix_ifelse_edges"] += len([w for w in autofix_warnings if "ifelse" in w and "补" in w])
        RUNTIME_METRICS["spec_autofix_fail_branch_wraps"] += len([w for w in autofix_warnings if "fail-branch" in w])
        RUNTIME_METRICS["spec_autofix_aggregators"] += len([w for w in autofix_warnings if "variable_aggregator" in w])
    return normalized, warnings


def llm_repair_workflow_spec_v2(spec: dict, errors: list[str], requirement: dict) -> dict:
    if not model_ready():
        raise ValueError("MODEL_UNAVAILABLE")
    system_prompt = f"""
你是工作流修复器。你会收到一个存在结构错误的 workflow spec。
仅输出修复后的 JSON，不要输出解释。

支持节点类型：{sorted(list(SUPPORTED_STEP_TYPES))}
错误列表：{json.dumps(errors, ensure_ascii=False)}
需求：{json.dumps(requirement, ensure_ascii=False)}
原始 spec：{json.dumps(spec, ensure_ascii=False)}
"""
    content = call_llm_chat(
        messages=[{"role": "system", "content": system_prompt}],
        temperature=0,
    )
    return _safe_json_load_dict(content)


def _score_candidate_v2(spec: dict, requirement: dict, validation: dict) -> float:
    score = 0.0
    stats = validation.get("stats", {})
    type_counts = stats.get("type_counts", {})
    total_nodes = stats.get("total_nodes", 0)

    if validation.get("ok"):
        score += 40
    else:
        score -= 80

    caps = set(requirement.get("capabilities", []))
    if "retrieval" in caps or "private_kb" in caps:
        score += 10 if type_counts.get("knowledge_retrieval", 0) > 0 else -8
    if "code_execution" in caps:
        score += 10 if type_counts.get("code", 0) > 0 else -8
    if "tool_call" in caps:
        score += 10 if type_counts.get("tool", 0) > 0 else -8
    if "branching" in caps:
        score += 8 if type_counts.get("ifelse", 0) > 0 else -6

    if "branching" in caps:
        score += min(int(type_counts.get("variable_aggregator", 0) or 0), 2) * 3

    unique_types = len([k for k, v in type_counts.items() if v > 0])
    score += min(unique_types, 6) * 2.5
    score += stats.get("non_llm_node_ratio", 0.0) * 10
    if stats.get("has_path_start_to_answer"):
        score += 15

    if isinstance(spec, dict) and isinstance(spec.get("edges"), list):
        fail_branch_edges = [
            e for e in spec.get("edges", [])
            if isinstance(e, dict) and str(e.get("source_handle") or "") in {"fail-branch", "success-branch"}
        ]
        score += min(len(fail_branch_edges), 2) * 0.5

    target = requirement.get("node_preferences", {}).get("target_node_count", 8)
    if isinstance(target, int):
        score -= abs(total_nodes - target) * 1.2

    score -= len(validation.get("warnings", [])) * 0.8
    meta = spec.get("meta", {}) if isinstance(spec, dict) else {}
    autofix_count = int(meta.get("autofix_warnings", 0) or 0)
    if autofix_count == 0:
        score += 3
    else:
        score -= min(autofix_count, 8) * 1.5
    return round(score, 3)


def _build_candidate_brief(candidate_id: str, score: float, validation: dict) -> dict:
    return {
        "id": candidate_id,
        "score": score,
        "type_stats": validation.get("stats", {}).get("type_counts", {}),
        "non_llm_node_ratio": validation.get("stats", {}).get("non_llm_node_ratio", 0),
        "ok": validation.get("ok", False),
        "warnings": validation.get("warnings", []),
    }


def _save_artifact(payload: dict) -> str:
    artifact_id = f"art_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    path = os.path.join(ARTIFACT_DIR, f"{artifact_id}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return artifact_id


def _record_analyze_metrics(success: bool, latency_ms: float, selected_validation: dict | None):
    RUNTIME_METRICS["analyze_total"] += 1
    if success:
        RUNTIME_METRICS["analyze_success"] += 1
    else:
        RUNTIME_METRICS["analyze_fail"] += 1
    RUNTIME_METRICS["analyze_latency_ms"].append(latency_ms)
    if selected_validation and selected_validation.get("stats"):
        ratio = selected_validation["stats"].get("non_llm_node_ratio")
        if isinstance(ratio, (int, float)):
            RUNTIME_METRICS["non_llm_node_ratios"].append(float(ratio))


def _p95(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, int(round(len(ordered) * 0.95)) - 1)
    return round(ordered[max(0, idx)], 2)


def log_event(event: str, trace_id: str, **kwargs):
    payload = {
        "ts": datetime.now().isoformat(),
        "event": event,
        "trace_id": trace_id,
        **kwargs,
    }
    print(json.dumps(payload, ensure_ascii=False))


def analyze_with_llm_v2(user_input: str) -> dict:
    started = time.time()
    _refresh_model_runtime(force=True)
    active_model = MODEL_RUNTIME.get("active_model") or "unknown"
    active_provider = MODEL_RUNTIME.get("active_provider") or "unknown"
    requirement = llm_structure_requirement_v2(user_input)
    level = _estimate_complexity_level(user_input, requirement)
    budget_level = (requirement.get("runtime_config", {}) or {}).get("budget_level", "balanced")
    candidate_count = _candidate_count_for_level(level)
    if budget_level == "low":
        candidate_count = 1
    elif budget_level == "high" and level == "complex":
        candidate_count = min(3, candidate_count + 1)
    temperatures = [0.2, 0.45, 0.65]
    latency_budget_seconds = 8 if level == "simple" else 15

    candidates = []
    selected_validation = None

    for idx in range(candidate_count):
        if idx > 0 and (time.time() - started) >= latency_budget_seconds:
            break
        candidate_id = f"cand_{idx + 1}"
        repair_rounds = 0
        fix_warnings: list[str] = []

        raw_spec = llm_plan_workflow_candidate_v2(requirement, temperature=temperatures[min(idx, len(temperatures) - 1)])
        fixed_spec, fix_warnings = deterministic_fix_workflow_spec_v2(raw_spec)
        normalized_spec, normalize_warnings = normalize_workflow_spec_v2(fixed_spec, requirement)
        validation = validate_workflow_spec_v2(normalized_spec)

        if not validation["ok"]:
            for _ in range(2):
                repaired = llm_repair_workflow_spec_v2(normalized_spec, validation["errors"], requirement)
                fixed_spec, more_fix_warnings = deterministic_fix_workflow_spec_v2(repaired)
                fix_warnings.extend(more_fix_warnings)
                normalized_spec, more_normalize_warnings = normalize_workflow_spec_v2(fixed_spec, requirement)
                normalize_warnings.extend(more_normalize_warnings)
                validation = validate_workflow_spec_v2(normalized_spec)
                repair_rounds += 1
                if validation["ok"]:
                    break

        validation["warnings"] = validation.get("warnings", []) + fix_warnings + normalize_warnings
        score = _score_candidate_v2(normalized_spec, requirement, validation)

        candidates.append({
            "id": candidate_id,
            "spec": normalized_spec,
            "validation": validation,
            "score": score,
            "repair_rounds": repair_rounds,
        })

    ok_candidates = [c for c in candidates if c.get("validation", {}).get("ok")]
    if ok_candidates:
        selected = max(ok_candidates, key=lambda c: c["score"])
    else:
        selected = max(candidates, key=lambda c: c["score"])
    selected_validation = selected["validation"]

    selected_spec = copy.deepcopy(selected["spec"])
    selected_spec.setdefault("meta", {})
    selected_spec["meta"].update({
        "planning_mode": PLANNING_MODE,
        "model": active_model,
        "provider": active_provider,
        "prompt_version": PROMPT_VERSION,
        "schema_version": SCHEMA_VERSION,
        "score": selected["score"],
        "selected_candidate_id": selected["id"],
        "complexity_level": level,
        "candidate_count": candidate_count,
    })
    selected_spec["requirement_meta"] = requirement

    candidates_brief = [
        _build_candidate_brief(c["id"], c["score"], c["validation"])
        for c in candidates
    ]

    artifact_payload = {
        "timestamp": datetime.now().isoformat(),
        "model_name": active_model,
        "provider": active_provider,
        "prompt_version": PROMPT_VERSION,
        "schema_version": SCHEMA_VERSION,
        "requirement": requirement,
        "candidate_scores": [
            {
                "id": c["id"],
                "score": c["score"],
                "ok": c["validation"].get("ok", False),
                "warnings": c["validation"].get("warnings", []),
                "stats": c["validation"].get("stats", {}),
                "repair_rounds": c["repair_rounds"],
            }
            for c in candidates
        ],
        "selected_candidate_id": selected["id"],
        "selected_spec": selected_spec,
    }
    artifact_id = _save_artifact(artifact_payload)

    return {
        "requirement": requirement,
        "selected_spec": selected_spec,
        "candidates_brief": candidates_brief,
        "selected_candidate_id": selected["id"],
        "validation_warnings": selected_validation.get("warnings", []),
        "repair_rounds": selected["repair_rounds"],
        "artifact_id": artifact_id,
        "selected_validation": selected_validation,
    }


def llm_analyze_requirement(user_input: str) -> dict:
    if not deepseek_client:
        raise ValueError("未检测到 DeepSeek API Key")

    system_prompt = """
你是一个工作流需求分析器。
你的任务不是直接设计 Dify YAML，而是把用户需求解析成结构化 JSON，供后续工作流规划器使用。

你只允许输出 JSON，不要输出解释，不要输出 markdown，不要输出代码块。

输出字段必须包含：
- app_name
- description
- scene
- user_goal
- inputs
- output_contract
- capabilities
- node_preferences
- runtime_config
- constraints

字段说明：

1. scene 只能是以下之一：
- xiaohongshu
- summary
- translation
- qa
- web_summary
- generic
额外判定规则：
- 如果用户的目标是“创建工作流 / 搭建工作流 / 自动生成 Dify 工作流 / 生成 workflow YAML / 节点编排 / Web 应用生成工作流”，优先判定为 generic
- 只有当用户明确要求“生成小红书笔记内容 / 营销文案 / 爆款标题 / 种草文案”时，才判定为 xiaohongshu
- 如果请求本质是“做一个系统 / 做一个平台 / 做一个自动编排器”，不要判定为 xiaohongshu

2. inputs 是数组，每项包含：
- name
- label
- type
- required

其中 type 只能是：
- text
- paragraph
- file

3. output_contract 是对象，可包含：
- sections
- multiple_candidates
- strict_format

4. capabilities 是数组，可选值只能来自以下集合：
- input_check
- intent_analysis
- planning
- retrieval
- code_execution
- tool_call
- branching
- generation
- optimization
- formatting
- multi_title
- private_kb

5. node_preferences 是对象，必须包含：
- target_node_count
- need_private_kb
- need_code_node
- need_tools
- need_branching

target_node_count 必须是 5 到 8 之间的整数。

6. runtime_config 是对象，可包含：
- kb_bind_mode
- tool_scope
- model_tier

7. constraints 是数组，写出用户明确或隐含的限制要求。

补充规则：
- 如果需求里出现“如果 / 否则 / 判断 / 根据不同情况输出不同结果”，加入 branching，并把 need_branching 设为 true
- 如果需求涉及“知识库 / 私有资料 / 文档库 / 内部文档 / 检索增强 / RAG”，加入 retrieval 和 private_kb，并把 need_private_kb 设为 true
- 如果需求涉及“计算 / 清洗 / 脚本 / Python / 代码执行 / 结构化处理”，加入 code_execution，并把 need_code_node 设为 true
- 如果需求涉及“工具 / 搜索 / API / skill / 外部调用”，加入 tool_call，并把 need_tools 设为 true
- 如果需求强调“固定格式 / 分段输出 / 模板化输出 / 中英对照”，加入 formatting
- 如果需求提到“多个版本 / 多个标题 / 多个候选”，加入 multi_title
- 如果用户没有明确节点数，默认 target_node_count = 8

输出必须是合法 JSON。
"""

    response = deepseek_client.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input}
        ],
        temperature=0
    )

    content = response.choices[0].message.content.strip()

    if content.startswith("```"):
        content = content.strip("`")
        content = content.replace("json", "", 1).strip()

    result = json.loads(content)

    result.setdefault("app_name", "通用工作流")
    result.setdefault("description", "由 AI 分析得到的工作流需求")
    result.setdefault("scene", "generic")
    if not isinstance(result.get("scene"), str):
        result["scene"] = "generic"
    result.setdefault("user_goal", "")
    result.setdefault("inputs", [])
    result.setdefault("output_contract", {})
    result.setdefault("capabilities", [])
    result.setdefault("node_preferences", {})
    result.setdefault("runtime_config", {})
    result.setdefault("constraints", [])

    if not isinstance(result.get("inputs"), list):
        result["inputs"] = []

    if not isinstance(result.get("capabilities"), list):
        result["capabilities"] = []

    if not isinstance(result.get("constraints"), list):
        result["constraints"] = []

    if not isinstance(result.get("node_preferences"), dict):
        result["node_preferences"] = {}

    if not isinstance(result.get("runtime_config"), dict):
        result["runtime_config"] = {}

    if not isinstance(result.get("output_contract"), dict):
        result["output_contract"] = {}

    sections = result["output_contract"].get("sections")
    if sections is not None:
        if not isinstance(sections, list):
            result["output_contract"]["sections"] = []
        else:
            result["output_contract"]["sections"] = [
                sec for sec in sections if isinstance(sec, dict)
            ]

    node_preferences = result["node_preferences"]
    node_preferences.setdefault("target_node_count",8)
    node_preferences.setdefault("need_private_kb", False)
    node_preferences.setdefault("need_code_node", False)
    node_preferences.setdefault("need_tools", False)
    node_preferences.setdefault("need_branching", "branching" in result["capabilities"])

    runtime_config = result["runtime_config"]
    runtime_config.setdefault("kb_bind_mode", "manual_or_auto")
    runtime_config.setdefault("tool_scope", [])
    runtime_config.setdefault("model_tier", DEEPSEEK_MODEL)

    return result


def build_branching_config(requirement: dict) -> dict:
    scene = requirement.get("scene", "generic")
    description = requirement.get("description", "")
    capabilities = requirement.get("capabilities", [])

    # 1) 翻译类：直译 / 意译
    if scene == "translation":
        return {
            "cases": [
                {"id": "literal", "label": "直译"},
                {"id": "free", "label": "意译"}
            ],
            "branches": [
                {
                    "id": "branch_literal",
                    "case_id": "literal",
                    "llm_id": "llm_generate_literal",
                    "llm_title": "直译生成",
                    "llm_prompt": "请严格保留原意、结构和信息，输出直译版本。",
                    "template_id": "template_output_literal",
                    "template_title": "直译结果格式化",
                    "answer_id": "answer_literal",
                    "answer_title": "直译回复"
                },
                {
                    "id": "branch_free",
                    "case_id": "free",
                    "llm_id": "llm_generate_free",
                    "llm_title": "意译生成",
                    "llm_prompt": "请在不改变核心意思的前提下，用更自然、更符合目标语言习惯的表达输出意译版本。",
                    "template_id": "template_output_free",
                    "template_title": "意译结果格式化",
                    "answer_id": "answer_free",
                    "answer_title": "意译回复"
                }
            ]
        }

    # 2) 总结类：简版 / 详版
    if scene == "summary":
        return {
            "cases": [
                {"id": "brief", "label": "简版"},
                {"id": "detailed", "label": "详版"}
            ],
            "branches": [
                {
                    "id": "branch_brief",
                    "case_id": "brief",
                    "llm_id": "llm_generate_brief",
                    "llm_title": "简版总结生成",
                    "llm_prompt": "请输出简洁版总结，突出核心结论和关键要点。",
                    "template_id": "template_output_brief",
                    "template_title": "简版结果格式化",
                    "answer_id": "answer_brief",
                    "answer_title": "简版回复"
                },
                {
                    "id": "branch_detailed",
                    "case_id": "detailed",
                    "llm_id": "llm_generate_detailed",
                    "llm_title": "详版总结生成",
                    "llm_prompt": "请输出详细版总结，包含背景、要点、结论和必要说明。",
                    "template_id": "template_output_detailed",
                    "template_title": "详版结果格式化",
                    "answer_id": "answer_detailed",
                    "answer_title": "详版回复"
                }
            ]
        }

    # 3) QA类：直接回答 / 检索增强回答
    if scene in ["qa", "pdf_qa"]:
        return {
            "cases": [
                {"id": "direct", "label": "直接回答"},
                {"id": "retrieval", "label": "检索增强"}
            ],
            "branches": [
                {
                    "id": "branch_direct",
                    "case_id": "direct",
                    "llm_id": "llm_generate_direct",
                    "llm_title": "直接回答生成",
                    "llm_prompt": "请基于已有输入直接回答问题，要求准确、清晰、简洁。",
                    "template_id": "template_output_direct",
                    "template_title": "直接回答格式化",
                    "answer_id": "answer_direct",
                    "answer_title": "直接回答回复"
                },
                {
                    "id": "branch_retrieval",
                    "case_id": "retrieval",
                    "llm_id": "llm_generate_retrieval",
                    "llm_title": "增强回答生成",
                    "llm_prompt": "请结合上下文、检索结果或补充信息，生成更完整、更有依据的回答。",
                    "template_id": "template_output_retrieval",
                    "template_title": "增强回答格式化",
                    "answer_id": "answer_retrieval",
                    "answer_title": "增强回答回复"
                }
            ]
        }

    # 4) 小红书/营销类：营销版 / 通用版
    if scene == "xiaohongshu" or "multi_title" in capabilities or "marketing" in description:
        return {
            "cases": [
                {"id": "marketing", "label": "营销表达"},
                {"id": "general", "label": "通用表达"}
            ],
            "branches": [
                {
                    "id": "branch_marketing",
                    "case_id": "marketing",
                    "llm_id": "llm_generate_marketing",
                    "llm_title": "营销内容生成",
                    "llm_prompt": "请生成更适合传播、种草、吸引点击的营销内容初稿。",
                    "template_id": "template_output_marketing",
                    "template_title": "营销结果格式化",
                    "answer_id": "answer_marketing",
                    "answer_title": "营销结果回复"
                },
                {
                    "id": "branch_general",
                    "case_id": "general",
                    "llm_id": "llm_generate_general",
                    "llm_title": "通用内容生成",
                    "llm_prompt": "请生成更中性、更通用、更稳妥的内容初稿。",
                    "template_id": "template_output_general",
                    "template_title": "通用结果格式化",
                    "answer_id": "answer_general",
                    "answer_title": "通用结果回复"
                }
            ]
        }

    # 5) 默认兜底：方案A / 方案B
    return {
        "cases": [
            {"id": "option_a", "label": "方案A"},
            {"id": "option_b", "label": "方案B"}
        ],
        "branches": [
            {
                "id": "branch_a",
                "case_id": "option_a",
                "llm_id": "llm_generate_a",
                "llm_title": "方案A生成",
                "llm_prompt": "请基于用户需求生成方案A，偏简洁直接。",
                "template_id": "template_output_a",
                "template_title": "方案A格式化",
                "answer_id": "answer_a",
                "answer_title": "方案A回复"
            },
            {
                "id": "branch_b",
                "case_id": "option_b",
                "llm_id": "llm_generate_b",
                "llm_title": "方案B生成",
                "llm_prompt": "请基于用户需求生成方案B，偏完整详细。",
                "template_id": "template_output_b",
                "template_title": "方案B格式化",
                "answer_id": "answer_b",
                "answer_title": "方案B回复"
            }
        ]
    }


def _extract_selection_features(requirement: dict) -> dict[str, bool]:
    text_parts = [
        str(requirement.get("user_goal", "")),
        str(requirement.get("description", "")),
    ]
    for c in requirement.get("constraints", []) if isinstance(requirement.get("constraints"), list) else []:
        text_parts.append(str(c))
    joined = " ".join(text_parts).lower()
    caps = requirement.get("capabilities", []) if isinstance(requirement.get("capabilities"), list) else []
    return {
        "mentions_kb": ("retrieval" in caps or "private_kb" in caps) or any(k in joined for k in ["知识库", "文档", "资料", "检索", "内部规范", "reference", "retrieve"]),
        "mentions_tool": ("tool_call" in caps) or any(k in joined for k in ["接口", "api", "http", "实时", "网页", "抓取", "web"]),
        "mentions_code": ("code_execution" in caps) or any(k in joined for k in ["计算", "转换", "清洗", "解析", "json", "yaml", "正则", "统计", "代码"]),
        "needs_routing": ("branching" in caps) or any(k in joined for k in ["分支", "路由", "条件", "if", "switch", "多路径"]),
    }


def choose_execution_nodes(
    need_private_kb: bool,
    need_code_node: bool,
    need_tools: bool,
    target_node_count: int,
    max_exec_nodes: int | None = None,
    requirement: dict | None = None,
) -> list[str]:
    """
    根据预算决定保留哪些执行节点。
    默认优先级：
    1. 知识库
    2. 代码
    3. 工具

    max_exec_nodes 如果传入，就直接按预算裁剪。
    """

    candidates: list[str] = []
    if need_private_kb:
        candidates.append("kb_retrieval")
    if need_code_node:
        candidates.append("code_exec")
    if need_tools:
        candidates.append("tool_call")

    score_map = {"kb_retrieval": 0.0, "code_exec": 0.0, "tool_call": 0.0}
    if need_private_kb:
        score_map["kb_retrieval"] += 1.0
    if need_code_node:
        score_map["code_exec"] += 1.0
    if need_tools:
        score_map["tool_call"] += 1.0

    features = _extract_selection_features(requirement or {})
    if features["mentions_kb"]:
        score_map["kb_retrieval"] += 1.5
    if features["mentions_code"]:
        score_map["code_exec"] += 1.5
    if features["mentions_tool"]:
        score_map["tool_call"] += 1.5

    scene = requirement.get("scene", "generic") if requirement else "generic"
    if scene == "qa":
        score_map["kb_retrieval"] += 3.0
        score_map["tool_call"] += 1.0
    elif scene in ["data_analysis", "summary"]:
        score_map["code_exec"] += 3.0
        score_map["tool_call"] += 1.0
    elif scene == "web_summary":
        score_map["tool_call"] += 3.0

    priority = {"kb_retrieval": 0, "code_exec": 1, "tool_call": 2}
    ranked = sorted(candidates, key=lambda n: (-score_map.get(n, 0), priority.get(n, 99)))

    if max_exec_nodes is not None:
        return ranked[:max(0, max_exec_nodes)]

    if target_node_count <= 6:
        return ranked[:1]
    if target_node_count == 7:
        return ranked[:2]
    return ranked[:2]


def synthesize_workflow_spec(requirement: dict) -> dict:
    scene = requirement.get("scene", "generic")
    app_name = requirement.get("app_name", "通用工作流")
    description = requirement.get("description", "由规则引擎生成的工作流")
    inputs = requirement.get("inputs", [])
    capabilities = requirement.get("capabilities", [])
    raw_output_contract = requirement.get("output_contract", {})
    output_contract = normalize_output_contract(scene, raw_output_contract)
    node_preferences = requirement.get("node_preferences", {}) or {}
    scene_defaults = SCENE_CAPABILITY_DEFAULTS.get(scene, {})
    default_caps = scene_defaults.get("capabilities", [])
    default_inputs = scene_defaults.get("extra_inputs", [])

    if not inputs:
        if default_inputs:
            inputs = default_inputs
        else:
            inputs = [
                {
                    "name": "user_request",
                    "label": "用户输入",
                    "type": "paragraph",
                    "required": True
                }
            ]

    capabilities = list(dict.fromkeys(default_caps + capabilities))

    target_node_count = int(node_preferences.get("target_node_count", 8))
    target_node_count = max(5, min(8, target_node_count))

    need_private_kb = bool(
        node_preferences.get("need_private_kb", False)
        or "private_kb" in capabilities
        or "retrieval" in capabilities
    )
    need_code_node = bool(
        node_preferences.get("need_code_node", False)
        or "code_execution" in capabilities
    )
    need_tools = bool(
        node_preferences.get("need_tools", False)
        or "tool_call" in capabilities
    )
    explicit_branching = bool(
        node_preferences.get("need_branching", False)
        or "branching" in capabilities
    )

    has_real_branch_targets = bool(need_private_kb or need_code_node or need_tools)

    steps = []
    edges = []

    def add_step(step_id, step_type, title, prompt=None, config=None):
        step = {
            "id": step_id,
            "type": step_type,
            "title": title
        }
        if prompt:
            step["prompt"] = prompt
        if config:
            step["config"] = config
        steps.append(step)

    # ===== 先按预算规划 =====
    # 固定主链：start + llm_intent + llm_plan + llm_generate + answer = 5
    base_count = 5
    pre_branch_enabled = bool(explicit_branching and has_real_branch_targets)
    branch_count = 1 if pre_branch_enabled else 0
    remaining_slots = target_node_count - base_count - branch_count
    remaining_slots = max(0, remaining_slots)

    # 优先给执行节点预算
    selected_exec_nodes = choose_execution_nodes(
        need_private_kb=need_private_kb,
        need_code_node=need_code_node,
        need_tools=need_tools,
        target_node_count=target_node_count,
        max_exec_nodes=remaining_slots,
        requirement=requirement,
    )
    remaining_slots -= len(selected_exec_nodes)

    # 分支只在显式需要且确实存在多执行目标时启用，避免无意义 ifelse
    need_branching = bool(explicit_branching and len(selected_exec_nodes) >= 2)
    if pre_branch_enabled and (not need_branching):
        remaining_slots += 1

    # 再决定是否保留 input_check / review / template
    add_input_check = False
    add_review = False
    add_template = False

    if remaining_slots >= 1 and "input_check" in capabilities and not need_branching:
        add_input_check = True
        remaining_slots -= 1

    if remaining_slots >= 1 and "optimization" in capabilities and not need_branching:
        add_review = True
        remaining_slots -= 1

    if remaining_slots >= 1 and "formatting" in capabilities and not need_branching:
        add_template = True
        remaining_slots -= 1

    # ===== 开始建节点 =====
    add_step("start", "start", "开始")

    if add_input_check:
        add_step(
            "llm_input_check",
            "llm",
            "输入校验",
            "请检查用户输入是否完整、明确、可执行。如信息不足，请指出缺失项；如信息充分，请输出“输入通过”。"
        )

    add_step(
        "llm_intent",
        "llm",
        "需求理解",
        "请提取任务目标、输入变量、输出形式、关键约束、风险点和推荐执行路径。"
    )

    add_step(
        "llm_plan",
        "llm",
        "执行规划",
        "请基于需求理解结果规划执行路径，只输出一行路由标记：ROUTE=knowledge 或 ROUTE=compute 或 ROUTE=tool 或 ROUTE=direct。"
    )

    if need_branching:
        add_step(
            "ifelse_route",
            "ifelse",
            "执行路由",
            config={
                "mode": "branching",
                "cases": [
                    {"id": "knowledge", "label": "知识库路径"},
                    {"id": "compute", "label": "代码路径"},
                    {"id": "tool", "label": "工具路径"},
                    {"id": "direct", "label": "直接生成路径"}
                ]
            }
        )

    if "kb_retrieval" in selected_exec_nodes:
        add_step(
            "kb_retrieval",
            "knowledge_retrieval",
            "知识库检索",
            config={
                "query_source": "user_input",
                "kb_mode": "manual_or_auto"
            }
        )

    if "code_exec" in selected_exec_nodes:
        add_step(
            "code_exec",
            "code",
            "代码处理",
            prompt="请对结构化内容进行计算、清洗、转换或规则处理。"
        )

    if "tool_call" in selected_exec_nodes:
        add_step(
            "tool_call",
            "tool",
            "工具调用",
            config={
                "tools": ["web_search", "calculator", "http"]
            }
        )

    add_step(
        "llm_generate",
        "llm",
        "结果生成",
        "请综合前面步骤的输入与中间结果，生成最终可交付内容。"
    )

    if add_review:
        add_step(
            "llm_review",
            "llm",
            "结果优化",
            "请检查结果是否完整、清晰、格式正确，并输出优化后的最终版本。"
        )

    if add_template:
        add_step(
            "template_output",
            "template",
            "格式输出",
            config={"template_mode": "simple"}
        )

    add_step("answer", "answer", "直接回复")

    # ===== 连边 =====
    existing_ids = {s["id"] for s in steps}

    ordered_main = ["start"]
    if "llm_input_check" in existing_ids:
        ordered_main.append("llm_input_check")
    ordered_main.extend(["llm_intent", "llm_plan"])

    for i in range(len(ordered_main) - 1):
        edges.append({
            "source": ordered_main[i],
            "target": ordered_main[i + 1]
        })

    current_tail = ordered_main[-1]

    if "ifelse_route" in existing_ids:
        edges.append({
            "source": current_tail,
            "target": "ifelse_route"
        })

        route_map = []
        if "kb_retrieval" in existing_ids:
            route_map.append(("knowledge", "kb_retrieval"))
        if "code_exec" in existing_ids:
            route_map.append(("compute", "code_exec"))
        if "tool_call" in existing_ids:
            route_map.append(("tool", "tool_call"))

        if route_map:
            for branch_name, target_id in route_map:
                edges.append({
                    "source": "ifelse_route",
                    "target": target_id,
                    "branch": branch_name
                })
                edges.append({
                    "source": target_id,
                    "target": "llm_generate"
                })

            edges.append({
                "source": "ifelse_route",
                "target": "llm_generate",
                "branch": "direct"
            })
        else:
            edges.append({
                "source": "ifelse_route",
                "target": "llm_generate",
                "branch": "direct"
            })
    else:
        if "kb_retrieval" in existing_ids:
            edges.append({"source": current_tail, "target": "kb_retrieval"})
            current_tail = "kb_retrieval"
        if "code_exec" in existing_ids:
            edges.append({"source": current_tail, "target": "code_exec"})
            current_tail = "code_exec"
        if "tool_call" in existing_ids:
            edges.append({"source": current_tail, "target": "tool_call"})
            current_tail = "tool_call"

        edges.append({
            "source": current_tail,
            "target": "llm_generate"
        })

    final_tail = "llm_generate"

    if "llm_review" in existing_ids:
        edges.append({
            "source": final_tail,
            "target": "llm_review"
        })
        final_tail = "llm_review"

    if "template_output" in existing_ids:
        edges.append({
            "source": final_tail,
            "target": "template_output"
        })
        final_tail = "template_output"

    edges.append({
        "source": final_tail,
        "target": "answer"
    })

    return {
        "workflow_name": app_name,
        "description": description,
        "scene": scene,
        "output_contract": output_contract,
        "inputs": inputs,
        "steps": steps,
        "edges": edges,
        "requirement_meta": requirement
    }

def normalize_output_contract(scene: str, output_contract: dict | None) -> dict:
    if not isinstance(output_contract, dict):
        output_contract = {}

    sections = output_contract.get("sections")

    # 只有当 sections 是“由 dict 组成的 list”时，才直接使用
    if isinstance(sections, list) and sections and all(isinstance(sec, dict) for sec in sections):
        normalized_sections = []
        for sec in sections:
            normalized_sections.append({
                "key": sec.get("key") or sec.get("label") or "field",
                "label": sec.get("label") or sec.get("key") or "结果项"
            })
        return {
            "sections": normalized_sections
        }
    if isinstance(sections, list) and sections and all(isinstance(sec, str) for sec in sections):
        return {
            "sections": [
                {"key": f"field_{idx+1}", "label": sec.strip() or f"结果项{idx+1}"}
                for idx, sec in enumerate(sections)
            ]
        }

    if scene == "summary":
        return {
            "sections": [
                {"key": "summary", "label": "一句话总结"},
                {"key": "highlights", "label": "核心要点"},
                {"key": "advice", "label": "行动建议"}
            ]
        }

    if scene == "translation":
        return {
            "sections": [
                {"key": "source", "label": "原文"},
                {"key": "translated", "label": "译文"},
                {"key": "notes", "label": "表达说明"}
            ]
        }

    return {
        "sections": [
            {"key": "final_result", "label": "最终结果"}
        ]
    }

def build_formatting_prompt(scene: str, output_contract: dict) -> str:
    contract = normalize_output_contract(scene, output_contract)
    sections = contract.get("sections", [])

    if not sections:
        return "请把最终结果整理成清晰、可直接交付的固定格式输出，不要输出解释。"

    lines = ["请把结果整理成固定结构输出："]
    for sec in sections:
        label = sec.get("label") or sec.get("key") or "结果项"
        lines.append(f"【{label}】")

    lines.append("不要输出解释。")
    return "\n".join(lines)


def build_output_template(scene: str, output_contract: dict) -> str:
    contract = normalize_output_contract(scene, output_contract)
    sections = contract.get("sections", [])

    if not sections:
        return "【最终结果】\n{{ arg1 }}"

    lines = []
    for sec in sections:
        label = sec.get("label") or sec.get("key") or "结果项"
        lines.append(f"【{label}】")
        lines.append("{{ arg1 }}")
        lines.append("")

    return "\n".join(lines).strip()


def enrich_spec_to_multistage(spec: dict, complexity: str = "normal") -> dict:
    steps = spec.get("steps", [])
    llm_steps = [s for s in steps if s.get("type") == "llm"]
    current_edges = spec.get("edges", [])

    if complexity == "simple":
        expected_step_ids = ["start", "llm_generate", "answer"]
        expected_edges = [
            {"source": "start", "target": "llm_generate"},
            {"source": "llm_generate", "target": "answer"}
        ]
    else:
        expected_step_ids = ["start", "llm_intent", "llm_plan", "llm_generate", "llm_review", "answer"]
        expected_edges = [
            {"source": "start", "target": "llm_intent"},
            {"source": "llm_intent", "target": "llm_plan"},
            {"source": "llm_plan", "target": "llm_generate"},
            {"source": "llm_generate", "target": "llm_review"},
            {"source": "llm_review", "target": "answer"}
        ]

    current_step_ids = [s.get("id") for s in steps]
    edge_pairs = {(e.get("source"), e.get("target")) for e in current_edges}
    expected_edge_pairs = {(e["source"], e["target"]) for e in expected_edges}

    # 如果节点和边都已经完整，就直接返回
    if (
        len(llm_steps) >= 3
        and all(step_id in current_step_ids for step_id in expected_step_ids)
        and expected_edge_pairs.issubset(edge_pairs)
    ):
        return spec

    # 找到原始核心 llm prompt
    main_prompt = ""
    for s in llm_steps:
        if s.get("prompt"):
            main_prompt = s["prompt"]
            break
    if not main_prompt:
        main_prompt = "请根据用户输入完成任务。"

    if complexity == "simple":
        spec["steps"] = [
            {"id": "start", "type": "start", "title": "开始"},
            {
                "id": "llm_generate",
                "type": "llm",
                "title": "核心生成",
                "prompt": main_prompt
            },
            {"id": "answer", "type": "answer", "title": "直接回复"}
        ]
    else:
        spec["steps"] = [
            {"id": "start", "type": "start", "title": "开始"},
            {
                "id": "llm_intent",
                "type": "llm",
                "title": "需求理解",
                "prompt": "请先理解用户需求，提取任务目标、输入变量、输出形式和关键约束。"
            },
            {
                "id": "llm_plan",
                "type": "llm",
                "title": "执行规划",
                "prompt": "请基于需求理解结果，规划最合适的执行步骤、输出结构和注意事项。"
            },
            {
                "id": "llm_generate",
                "type": "llm",
                "title": "核心生成",
                "prompt": main_prompt
            },
            {
                "id": "llm_review",
                "type": "llm",
                "title": "结果优化",
                "prompt": "请检查上一步结果是否完整、清晰、可直接交付，并输出最终优化版本。"
            },
            {"id": "answer", "type": "answer", "title": "直接回复"}
        ]

    spec["edges"] = expected_edges
    return spec


def infer_scene_from_spec(spec: dict) -> str:
    """
    从 AI 规划出的 spec 粗略推断一个场景标签，仅用于前端显示
    """
    name = (spec.get("workflow_name") or "").lower()
    desc = (spec.get("description") or "").lower()
    text = f"{name} {desc}"

    if any(k in text for k in ["翻译", "translate"]):
        return "translate"
    if any(k in text for k in ["总结", "摘要", "summary"]):
        return "summary"
    if any(k in text for k in ["问答", "qa", "回答"]):
        return "qa"
    return "unknown"


def build_preview_from_spec(spec: dict) -> dict:
    nodes = []
    for step in spec.get("steps", []):
        title = step.get("title") or step.get("id") or "step"
        step_type = step.get("type", "unknown")
        nodes.append(f"{title} ({step_type})")

    output_contract = spec.get("output_contract", {}) or {}
    if not isinstance(output_contract, dict):
        output_contract = {}
    sections = output_contract.get("sections", [])
    output_sections = []

    for sec in sections:
        if isinstance(sec, dict):
            label = sec.get("label") or sec.get("key") or "结果项"
        else:
            label = str(sec).strip() or "结果项"
        output_sections.append(label)

    return {
        "scene": spec.get("workflow_name", "通用工作流"),
        "nodes": nodes,
        "output_sections": output_sections
    }


def _dedup_edges(edges: list[dict]) -> list[dict]:
    deduped = []
    seen = set()
    for edge in edges:
        if not isinstance(edge, dict):
            continue
        key = (edge.get("source"), edge.get("target"), edge.get("branch"))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(edge)
    return deduped


def _prune_exec_nodes(spec: dict, max_exec_nodes: int | None, budget_level: str | None) -> dict:
    if not isinstance(spec, dict):
        return spec
    if not isinstance(spec.get("steps"), list) or not isinstance(spec.get("edges"), list):
        return spec

    budget_map = {"low": 1, "medium": 2, "balanced": 2, "high": 3}
    cap = max_exec_nodes
    if cap is None and budget_level in budget_map:
        cap = budget_map[budget_level]
    if cap is None:
        return spec

    cap = max(0, int(cap))
    execution_types = {"knowledge_retrieval", "code", "tool"}
    steps = copy.deepcopy(spec["steps"])
    edges = copy.deepcopy(spec["edges"])
    priority = {"knowledge_retrieval": 0, "code": 1, "tool": 2}

    exec_steps = [s for s in steps if s.get("type") in execution_types]
    if len(exec_steps) <= cap:
        return spec

    ranked_exec = sorted(exec_steps, key=lambda s: (priority.get(s.get("type"), 99), steps.index(s)))
    keep_ids = {s.get("id") for s in ranked_exec[:cap] if s.get("id")}
    remove_ids = {s.get("id") for s in exec_steps if s.get("id") not in keep_ids}

    if not remove_ids:
        return spec

    for rid in remove_ids:
        incoming = [e for e in edges if e.get("target") == rid]
        outgoing = [e for e in edges if e.get("source") == rid]
        for inc in incoming:
            for out in outgoing:
                if inc.get("source") == out.get("target"):
                    continue
                bridged = {"source": inc.get("source"), "target": out.get("target")}
                if inc.get("branch"):
                    bridged["branch"] = inc.get("branch")
                edges.append(bridged)

    spec["steps"] = [s for s in steps if s.get("id") not in remove_ids]
    spec["edges"] = _dedup_edges([
        e for e in edges
        if e.get("source") not in remove_ids and e.get("target") not in remove_ids
    ])
    spec.setdefault("meta", {})
    spec["meta"]["budget_level"] = budget_level or spec["meta"].get("budget_level", "balanced")
    spec["meta"]["max_exec_nodes"] = cap
    spec["meta"]["pruned_exec_nodes"] = sorted(list(remove_ids))
    return spec


def apply_user_config_to_spec(spec: dict, data: dict) -> dict:
    spec = copy.deepcopy(spec)

    app_name = (data.get("app_name") or "").strip()
    description = (data.get("description") or "").strip()
    answers = data.get("answers", []) or []

    if app_name:
        spec["workflow_name"] = app_name
    if description:
        spec["description"] = description

    answer_text = "\n".join(str(x).strip() for x in answers if str(x).strip())

    target_node_count = None
    enable_kb = None
    enable_code = None
    enable_tools = None
    enable_branching = None
    max_exec_nodes = data.get("max_exec_nodes")
    budget_level = str(data.get("budget_level", "")).strip().lower() or None

    m = re.search(r"节点数\s*[=:：]\s*(\d{1,2})", answer_text)
    if m:
        target_node_count = max(5, min(MAX_NODE_LIMIT, int(m.group(1))))

    if re.search(r"启用知识库\s*[=:：]\s*(是|true|yes)", answer_text, re.I):
        enable_kb = True
    elif re.search(r"启用知识库\s*[=:：]\s*(否|false|no)", answer_text, re.I):
        enable_kb = False

    if re.search(r"启用代码节点\s*[=:：]\s*(是|true|yes)", answer_text, re.I):
        enable_code = True
    elif re.search(r"启用代码节点\s*[=:：]\s*(否|false|no)", answer_text, re.I):
        enable_code = False

    if re.search(r"启用工具调用\s*[=:：]\s*(是|true|yes)", answer_text, re.I):
        enable_tools = True
    elif re.search(r"启用工具调用\s*[=:：]\s*(否|false|no)", answer_text, re.I):
        enable_tools = False

    if re.search(r"启用分支\s*[=:：]\s*(是|true|yes)", answer_text, re.I):
        enable_branching = True
    elif re.search(r"启用分支\s*[=:：]\s*(否|false|no)", answer_text, re.I):
        enable_branching = False

    requirement = spec.get("requirement_meta", {})
    node_preferences = requirement.get("node_preferences", {})

    if target_node_count is not None:
        node_preferences["target_node_count"] = target_node_count
    if enable_kb is not None:
        node_preferences["need_private_kb"] = enable_kb
    if enable_code is not None:
        node_preferences["need_code_node"] = enable_code
    if enable_tools is not None:
        node_preferences["need_tools"] = enable_tools
    if enable_branching is not None:
        node_preferences["need_branching"] = enable_branching

    requirement["node_preferences"] = node_preferences
    runtime_config = requirement.get("runtime_config", {}) if isinstance(requirement.get("runtime_config"), dict) else {}
    if budget_level:
        runtime_config["budget_level"] = budget_level
    requirement["runtime_config"] = runtime_config
    spec["requirement_meta"] = requirement
    # v2 mode: keep model-generated topology; only fall back to rule synthesis when spec is empty.
    has_topology = isinstance(spec.get("steps"), list) and isinstance(spec.get("edges"), list) and spec.get("steps")
    if has_topology:
        return _prune_exec_nodes(spec, max_exec_nodes=max_exec_nodes, budget_level=budget_level)

    synthesized = synthesize_workflow_spec(requirement)
    return _prune_exec_nodes(synthesized, max_exec_nodes=max_exec_nodes, budget_level=budget_level)


def validate_workflow_spec(spec: dict):
    validation = validate_workflow_spec_v2(spec, MAX_NODE_LIMIT)
    if validation["ok"]:
        return True, ""
    if validation["errors"]:
        return False, "; ".join(validation["errors"])
    return False, "workflow spec validation failed"


def schema_self_check() -> bool:
    sample = {
        "workflow_name": "self-check",
        "description": "schema sanity",
        "scene": "generic",
        "steps": [
            {"id": "start", "type": "start", "title": "start"},
            {"id": "planner", "type": "llm", "title": "planner", "prompt": "test"},
            {"id": "answer", "type": "answer", "title": "answer"},
        ],
        "edges": [
            {"source": "start", "target": "planner"},
            {"source": "planner", "target": "answer"},
        ],
    }
    return validate_workflow_spec_v2(sample, MAX_NODE_LIMIT).get("ok", False)


def _get_step_type_map(steps: list) -> dict:
    return {step.get("id"): step.get("type") for step in steps}


def compile_workflow_spec_to_yaml(spec: dict) -> str:
    """
    把通用工作流 JSON 编译成一份结构化 YAML
    这份 YAML 是“通用工作流 DSL”，不是严格 Dify 官方 DSL
    """
    app_name = spec.get("workflow_name", "通用工作流")
    description = spec.get("description", "由 AI 自动生成的工作流")
    inputs = spec.get("inputs", [])
    steps = spec.get("steps", [])
    edges = spec.get("edges", [])

    compiled = {
        "app": {
            "name": app_name,
            "description": description,
            "mode": "workflow"
        },
        "workflow": {
            "inputs": inputs,
            "steps": steps,
            "edges": edges
        }
    }

    return yaml.safe_dump(
        compiled,
        allow_unicode=True,
        sort_keys=False
    )



@app.route("/test-deepseek")
def test_deepseek():
    result = test_deepseek_connection()
    return jsonify({"result": result})


def load_template_example(scene_key: str) -> str:
    config = LEGACY_SCENE_CONFIG.get(scene_key, LEGACY_SCENE_CONFIG["unknown"])
    template_file = config["template_file"]
    template_path = os.path.join(YAML_TEMPLATE_DIR, template_file)

    if not os.path.exists(template_path):
        raise FileNotFoundError(f"模板文件不存在: {template_file}")

    with open(template_path, "r", encoding="utf-8") as f:
        return f.read()

def llm_generate_yaml(scene_key: str, app_name: str, description: str, answers: list[str]) -> str:
    if not deepseek_client:
        raise ValueError("未检测到 DeepSeek API Key")

    template_example = load_template_example(scene_key)
    system_prompt_text = build_system_prompt(scene_key, answers)

    system_prompt = """
你是一个 Dify DSL YAML 生成助手。
你的任务不是从零自由生成，而是严格参考用户提供的示例模板结构，生成一份新的 YAML。

要求：
1. 只输出 YAML，不要输出解释，不要输出 markdown 代码块
2. 尽可能保持示例模板的结构、字段名、缩进风格不变
3. 只修改与当前需求有关的内容，例如：
   - app.name
   - app.description
   - LLM 节点中的 prompt/template/text
4. 不要随意新增字段，不要随意删除字段
5. 输出结果必须能被 Python yaml.safe_load 解析
6. 如果是 summary 场景，就生成“文档总结/文本总结”工作流
"""

    user_prompt = f"""
当前场景：
{scene_key}

目标应用名称：
{app_name}

目标应用描述：
{description}

根据用户回答整理出的系统提示词：
{system_prompt_text}

下面是一份真实 Dify YAML 模板，请你严格参考它的结构生成新的 YAML：

{template_example}

请基于以上模板输出新的 YAML。
"""

    response = deepseek_client.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0
    )

    content = response.choices[0].message.content.strip()

    if content.startswith("```"):
        content = content.strip("`")
        content = content.replace("yaml", "", 1).strip()

    return content

def validate_generated_yaml(yaml_content: str) -> tuple[bool, str]:
    try:
        data = yaml.safe_load(yaml_content)

        if not isinstance(data, dict):
            return False, "YAML 顶层不是对象"

        if "app" not in data:
            return False, "缺少 app 字段"

        if "workflow" not in data:
            return False, "缺少 workflow 字段"

        return True, "ok"

    except Exception as e:
        return False, str(e)

def llm_repair_yaml(bad_yaml: str, error_msg: str) -> str:
    if not deepseek_client:
        raise ValueError("未检测到 DeepSeek API Key")

    system_prompt = """
你是一个 YAML 修复助手。
你的任务是修复一份有错误的 YAML。
要求：
1. 只输出修复后的 YAML
2. 不要输出解释
3. 必须能被 yaml.safe_load 解析
"""

    user_prompt = f"""
下面这份 YAML 有错误，请修复它。

错误信息：
{error_msg}

原始 YAML：
{bad_yaml}
"""

    response = deepseek_client.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0
    )

    content = response.choices[0].message.content.strip()

    if content.startswith("```"):
        content = content.strip("`")
        content = content.replace("yaml", "", 1).strip()

    return content

def legacy_classify_intent(user_input: str) -> str:
    text = user_input.lower()

    if "pdf" in text or "文档问答" in user_input or "pdf问答" in user_input:
        return "pdf_qa"
    if "网页" in user_input and "总结" in user_input:
        return "web_summary"
    if "翻译" in user_input:
        return "translate"
    if "总结" in user_input or "摘要" in user_input:
        return "summary"
    if "问答" in user_input or "回答问题" in user_input or "提问" in user_input:
        return "qa"

    return "unknown"

# 3. 工具函数
def build_followup_text(scene_key: str) -> str:
    config = LEGACY_SCENE_CONFIG[scene_key]
    questions = config["questions"]
    label = config["label"]

    lines = [f"我识别到这是【{label}】场景。", "为了继续生成工作流，请补充以下信息："]
    for i, q in enumerate(questions, start=1):
        lines.append(f"{i}. {q}")
    return "\n".join(lines)

def build_followup_text_from_questions(scene_label: str, questions: list[str]) -> str:
    lines = [f"我识别到这是【{scene_label}】场景。", "为了继续生成工作流，请补充以下信息："]
    for i, q in enumerate(questions, start=1):
        lines.append(f"{i}. {q}")
    return "\n".join(lines)


def build_preview(scene_key: str):
    config = LEGACY_SCENE_CONFIG[scene_key]
    return {
        "scene": config["label"],
        "nodes": config["nodes"]
    }

def build_system_prompt(scene_key: str, answers: list[str]) -> str:
    if scene_key == "summary":
        style = answers[0] if len(answers) > 0 else "简短"
        keypoints = answers[1] if len(answers) > 1 else "否"
        return f"""请对用户输入的文本进行总结。
摘要风格：{style}
是否需要关键点列表：{keypoints}
请输出清晰、准确、简洁的总结结果。"""

    elif scene_key == "translate":
        target_lang = answers[0] if len(answers) > 0 else "英文"
        auto_detect = answers[1] if len(answers) > 1 else "是"
        bilingual = answers[2] if len(answers) > 2 else "否"
        return f"""请将用户输入的内容翻译成指定语言。
目标语言：{target_lang}
是否自动识别源语言：{auto_detect}
是否需要双语对照输出：{bilingual}
请确保语义准确、表达自然。"""

    elif scene_key == "qa":
        answer_style = answers[0] if len(answers) > 0 else "简洁"
        limit_length = answers[1] if len(answers) > 1 else "否"
        return f"""请根据用户提出的问题进行回答。
回答风格：{answer_style}
是否限制回答长度：{limit_length}
请确保回答准确、清晰。"""

    elif scene_key == "web_summary":
        input_mode = answers[0] if len(answers) > 0 else "网页链接"
        style = answers[1] if len(answers) > 1 else "简短"
        return f"""请对网页内容进行总结。
输入方式：{input_mode}
摘要风格：{style}
请输出清晰的网页内容摘要。"""

    elif scene_key == "pdf_qa":
        file_mode = answers[0] if len(answers) > 0 else "上传"
        answer_style = answers[1] if len(answers) > 1 else "简洁"
        return f"""请基于 PDF 内容回答用户问题。
输入方式：{file_mode}
回答风格：{answer_style}
请确保回答准确、简明。"""

    return "请根据用户需求生成合适的回答。"


def generate_yaml_from_template(scene_key: str, app_name: str, description: str, answers: list[str]) -> str:
    config = LEGACY_SCENE_CONFIG.get(scene_key, LEGACY_SCENE_CONFIG["unknown"])
    template_file = config["template_file"]
    template_path = os.path.join(YAML_TEMPLATE_DIR, template_file)

    if not os.path.exists(template_path):
        raise FileNotFoundError(f"模板文件不存在: {template_file}")

    with open(template_path, "r", encoding="utf-8") as f:
        content = f.read()

    system_prompt = build_system_prompt(scene_key, answers)

    content = content.replace("__APP_NAME__", app_name)
    content = content.replace("__APP_DESC__", description)

    # 自动读取模板里 __SYSTEM_PROMPT__ 这一行前面的缩进
    match = re.search(r"^(?P<indent>\s*)__SYSTEM_PROMPT__\s*$", content, flags=re.MULTILINE)
    if not match:
        raise ValueError("模板中未找到 __SYSTEM_PROMPT__ 占位符，请检查 summary_template.yml")

    indent = match.group("indent")
    indented_prompt = "\n".join(indent + line for line in system_prompt.splitlines())

    content = re.sub(
        r"^\s*__SYSTEM_PROMPT__\s*$",
        indented_prompt,
        content,
        count=1,
        flags=re.MULTILINE
    )

    return content


# 4. 路由
@app.route("/")
def index():
    return render_template("index.html", asset_version=get_asset_version())


@app.route("/favicon.ico")
def favicon():
    static_dir = os.path.join(app.root_path, "static")
    if os.path.exists(os.path.join(static_dir, "favicon.ico")):
        return send_from_directory(static_dir, "favicon.ico", mimetype="image/vnd.microsoft.icon")
    return Response(status=204)


@app.route("/health", methods=["GET"])
def health():
    _refresh_model_runtime(force=True)
    return jsonify({
        "deepseek_key_present": bool(DEEPSEEK_API_KEY),
        "deepseek_status": test_deepseek_connection(),
        "fallback_key_present": bool(FALLBACK_API_KEY),
        "model_ready": MODEL_RUNTIME.get("model_ready", False),
        "active_provider": MODEL_RUNTIME.get("active_provider"),
        "active_model": MODEL_RUNTIME.get("active_model"),
        "provider_status": MODEL_RUNTIME.get("provider_status", {}),
        "schema_ok": schema_self_check(),
        "import_ready": bool(DIFY_API_URL and DIFY_API_KEY),
        "broken_proxy_detected": _has_broken_local_proxy(),
        "proxy_env": get_proxy_env_snapshot(),
        "generated_dir_writable": os.access(GENERATED_DIR, os.W_OK),
        "planning_mode": PLANNING_MODE,
        "prompt_version": PROMPT_VERSION,
        "schema_version": SCHEMA_VERSION,
    })


@app.route("/metrics", methods=["GET"])
def metrics():
    analyze_total = max(1, RUNTIME_METRICS["analyze_total"])
    compile_total = max(1, RUNTIME_METRICS["compile_success"] + RUNTIME_METRICS["compile_fail"])
    generate_total = max(1, RUNTIME_METRICS["generate_total"])
    import_total = max(1, RUNTIME_METRICS["dify_import_success"] + RUNTIME_METRICS["dify_import_fail"])
    non_llm_avg = (
        statistics.mean(RUNTIME_METRICS["non_llm_node_ratios"])
        if RUNTIME_METRICS["non_llm_node_ratios"]
        else 0.0
    )
    lines = [
        "# HELP auto_agent_analyze_total Total analyze requests",
        "# TYPE auto_agent_analyze_total counter",
        f"auto_agent_analyze_total {RUNTIME_METRICS['analyze_total']}",
        "# HELP auto_agent_analyze_success_total Successful analyze requests",
        "# TYPE auto_agent_analyze_success_total counter",
        f"auto_agent_analyze_success_total {RUNTIME_METRICS['analyze_success']}",
        "# HELP auto_agent_analyze_success_rate Analyze success rate",
        "# TYPE auto_agent_analyze_success_rate gauge",
        f"auto_agent_analyze_success_rate {RUNTIME_METRICS['analyze_success'] / analyze_total}",
        "# HELP auto_agent_spec_validation_pass_rate Analyze validation pass rate",
        "# TYPE auto_agent_spec_validation_pass_rate gauge",
        f"auto_agent_spec_validation_pass_rate {RUNTIME_METRICS['analyze_success'] / analyze_total}",
        "# HELP auto_agent_generate_success_rate Generate success rate",
        "# TYPE auto_agent_generate_success_rate gauge",
        f"auto_agent_generate_success_rate {RUNTIME_METRICS['generate_success'] / generate_total}",
        "# HELP auto_agent_compile_success_rate Compile success rate",
        "# TYPE auto_agent_compile_success_rate gauge",
        f"auto_agent_compile_success_rate {RUNTIME_METRICS['compile_success'] / compile_total}",
        "# HELP auto_agent_dify_import_success_rate Dify import success rate",
        "# TYPE auto_agent_dify_import_success_rate gauge",
        f"auto_agent_dify_import_success_rate {RUNTIME_METRICS['dify_import_success'] / import_total}",
        "# HELP auto_agent_non_llm_node_rate Average non-llm node ratio",
        "# TYPE auto_agent_non_llm_node_rate gauge",
        f"auto_agent_non_llm_node_rate {round(non_llm_avg, 4)}",
        "# HELP auto_agent_spec_autofix_total Total autofix actions in normalization",
        "# TYPE auto_agent_spec_autofix_total counter",
        f"auto_agent_spec_autofix_total {RUNTIME_METRICS['spec_autofix_total']}",
        "# HELP auto_agent_spec_autofix_ifelse_edges Total autofixed ifelse branch edges",
        "# TYPE auto_agent_spec_autofix_ifelse_edges counter",
        f"auto_agent_spec_autofix_ifelse_edges {RUNTIME_METRICS['spec_autofix_ifelse_edges']}",
        "# HELP auto_agent_spec_autofix_fail_branch_wraps Total autofixed fail-branch wrappers",
        "# TYPE auto_agent_spec_autofix_fail_branch_wraps counter",
        f"auto_agent_spec_autofix_fail_branch_wraps {RUNTIME_METRICS['spec_autofix_fail_branch_wraps']}",
        "# HELP auto_agent_spec_autofix_aggregators Total inserted aggregator autofixes",
        "# TYPE auto_agent_spec_autofix_aggregators counter",
        f"auto_agent_spec_autofix_aggregators {RUNTIME_METRICS['spec_autofix_aggregators']}",
        "# HELP auto_agent_p95_analyze_latency_ms P95 analyze latency in ms",
        "# TYPE auto_agent_p95_analyze_latency_ms gauge",
        f"auto_agent_p95_analyze_latency_ms {_p95(RUNTIME_METRICS['analyze_latency_ms'])}",
        "# HELP auto_agent_p95_generate_latency_ms P95 generate latency in ms",
        "# TYPE auto_agent_p95_generate_latency_ms gauge",
        f"auto_agent_p95_generate_latency_ms {_p95(RUNTIME_METRICS['generate_latency_ms'])}",
        "# HELP auto_agent_p95_compile_latency_ms P95 compile latency in ms",
        "# TYPE auto_agent_p95_compile_latency_ms gauge",
        f"auto_agent_p95_compile_latency_ms {_p95(RUNTIME_METRICS['compile_latency_ms'])}",
        "# HELP auto_agent_p95_import_latency_ms P95 import latency in ms",
        "# TYPE auto_agent_p95_import_latency_ms gauge",
        f"auto_agent_p95_import_latency_ms {_p95(RUNTIME_METRICS['import_latency_ms'])}",
    ]
    return Response("\n".join(lines) + "\n", mimetype="text/plain; version=0.0.4")


@app.route("/validate_spec", methods=["POST"])
def validate_spec_endpoint():
    data = request.get_json() or {}
    spec = data.get("workflow_spec")
    validation = validate_workflow_spec_v2(spec, MAX_NODE_LIMIT)
    return jsonify(validation)


@app.route("/repair_spec", methods=["POST"])
def repair_spec_endpoint():
    data = request.get_json() or {}
    spec = data.get("workflow_spec")
    requirement = data.get("requirement", {})
    if not isinstance(spec, dict):
        return jsonify({"error": "workflow_spec 必须是对象"}), 400

    fixed, fix_warnings = deterministic_fix_workflow_spec_v2(spec)
    validation = validate_workflow_spec_v2(fixed, MAX_NODE_LIMIT)
    repair_rounds = 0
    llm_warnings = []

    if not validation["ok"]:
        if not model_ready():
            return jsonify({
                "error": "spec 需要模型修复，但当前无可用模型",
                "code": "MODEL_UNAVAILABLE",
                "workflow_spec": fixed,
                "validation": validation,
                "fix_warnings": fix_warnings,
                "provider_status": MODEL_RUNTIME.get("provider_status", {}),
            }), 503
        for _ in range(2):
            repaired = llm_repair_workflow_spec_v2(fixed, validation["errors"], requirement if isinstance(requirement, dict) else {})
            fixed, more_fix = deterministic_fix_workflow_spec_v2(repaired)
            llm_warnings.extend(more_fix)
            validation = validate_workflow_spec_v2(fixed, MAX_NODE_LIMIT)
            repair_rounds += 1
            if validation["ok"]:
                break

    fixed.setdefault("meta", {})
    fixed["meta"].update({
        "planning_mode": PLANNING_MODE,
        "schema_version": SCHEMA_VERSION,
        "repair_rounds": repair_rounds,
    })

    return jsonify({
        "workflow_spec": fixed,
        "validation": validation,
        "fix_warnings": fix_warnings + llm_warnings,
        "repair_rounds": repair_rounds,
    })


@app.route("/artifact/<artifact_id>", methods=["GET"])
def get_artifact(artifact_id):
    filename = f"{artifact_id}.json"
    path = os.path.join(ARTIFACT_DIR, filename)
    if not os.path.exists(path):
        return jsonify({"error": "artifact 不存在"}), 404
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return jsonify(payload)


@app.route("/template/complex_demo", methods=["GET"])
def get_complex_demo_template():
    path = os.path.join(YAML_TEMPLATE_DIR, "complex_demo.yml")
    if not os.path.exists(path):
        return jsonify({"error": "complex_demo.yml 不存在"}), 404
    with open(path, "r", encoding="utf-8") as f:
        parsed = yaml.safe_load(f) or {}
    workflow_spec = parsed.get("workflow_spec") if isinstance(parsed, dict) else None
    if not isinstance(workflow_spec, dict):
        return jsonify({"error": "complex_demo.yml 缺少 workflow_spec"}), 500
    validation = validate_workflow_spec_v2(workflow_spec, MAX_NODE_LIMIT)
    return jsonify({
        "workflow_spec": workflow_spec,
        "validation": validation,
        "code_examples": sorted(list(CODE_TEMPLATE_LIBRARY.keys())),
    })


@app.route("/analyze", methods=["POST"])
def analyze():
    started = time.time()
    trace_id = request.headers.get("X-Trace-Id") or f"trace_{uuid.uuid4().hex[:10]}"
    data = request.get_json() or {}
    user_input = data.get("user_input", "").strip()

    if not user_input:
        return jsonify({"error": "请输入工作流需求描述", "trace_id": trace_id}), 400
    if not model_ready():
        _record_analyze_metrics(False, (time.time() - started) * 1000, None)
        payload = _model_required_error_payload()
        payload["trace_id"] = trace_id
        log_event("analyze.blocked", trace_id, reason="model_unavailable")
        return jsonify(payload), 503

    try:
        log_event("analyze.start", trace_id, input_length=len(user_input))
        planning = analyze_with_llm_v2(user_input)
        requirement = planning["requirement"]
        spec = planning["selected_spec"]
        selected_validation = planning["selected_validation"]

        ok, err = validate_workflow_spec(spec)
        if not ok:
            _record_analyze_metrics(False, (time.time() - started) * 1000, selected_validation)
            log_event("analyze.invalid_spec", trace_id, error=err)
            return jsonify({"error": f"工作流结构无效：{err}", "trace_id": trace_id}), 400

        _record_analyze_metrics(True, (time.time() - started) * 1000, selected_validation)
        scene_key = requirement.get("scene", "generic")
        scene_label = spec.get("workflow_name", "通用工作流")
        preview = build_preview_from_spec(spec)

        followup_text = (
            f"我已经完成模型主导规划，当前选择方案为 {planning['selected_candidate_id']}。\n"
            f"模式：{PLANNING_MODE}，候选数：{len(planning['candidates_brief'])}。"
        )

        return jsonify({
            "scene_key": scene_key,
            "scene_label": scene_label,
            "followup_text": followup_text,
            "preview": preview,
            "app_name_suggestion": spec.get("workflow_name", ""),
            "description_suggestion": spec.get("description", ""),
            "requirement": requirement,
            "workflow_spec": spec,
            "planning_mode": PLANNING_MODE,
            "candidates_brief": planning["candidates_brief"],
            "selected_candidate_id": planning["selected_candidate_id"],
            "validation_warnings": planning["validation_warnings"],
            "repair_rounds": planning["repair_rounds"],
            "artifact_id": planning["artifact_id"],
            "node_type_stats": selected_validation.get("stats", {}).get("type_counts", {}),
            "trace_id": trace_id,
        })

    except Exception as e:
        _record_analyze_metrics(False, (time.time() - started) * 1000, None)
        if "MODEL_UNAVAILABLE" in str(e):
            payload = _model_required_error_payload()
            payload["trace_id"] = trace_id
            log_event("analyze.blocked", trace_id, reason="model_unavailable_exception")
            return jsonify(payload), 503
        log_event("analyze.error", trace_id, error=str(e))
        return jsonify({"error": f"AI 需求分析失败: {str(e)}", "trace_id": trace_id}), 500


def generate_yaml_content(scene_key: str, app_name: str, description: str, answers: list[str]) -> str:
    config = LEGACY_SCENE_CONFIG.get(scene_key, LEGACY_SCENE_CONFIG["unknown"])
    prompt_lines = [
        f"应用名称: {app_name}",
        f"场景类型: {config['label']}",
        f"应用描述: {description}",
        "用户补充信息:"
    ]
    for idx, ans in enumerate(answers, start=1):
        prompt_lines.append(f"  - 问题{idx}: {ans}")

    prompt_text = "\n".join(prompt_lines)

    yaml_content = f"""app:
  name: "{app_name}"
  description: "{description}"
  mode: "workflow"

workflow:
  scene: "{scene_key}"
  scene_label: "{config['label']}"
  nodes:
"""
    for node in config["nodes"]:
        yaml_content += f'    - "{node}"\n'

    yaml_content += f"""
  prompt: |
"""
    for line in prompt_text.split("\n"):
        yaml_content += f"    {line}\n"

    yaml_content += """
  note: "这是第一版 Auto-Agent 生成的演示 YAML，用于 MVP 展示。"
"""
    return yaml_content


def generate_yaml_with_ai(scene_key: str, app_name: str, description: str, answers: list[str]) -> str:
    yaml_content = llm_generate_yaml(scene_key, app_name, description, answers)

    ok, msg = validate_generated_yaml(yaml_content)
    if ok:
        return yaml_content

    repaired_yaml = llm_repair_yaml(yaml_content, msg)
    ok2, msg2 = validate_generated_yaml(repaired_yaml)
    if ok2:
        return repaired_yaml

    raise ValueError(f"AI 生成和修复后的 YAML 仍不合法: {msg2}")


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def load_dify_seed_template() -> dict:
    seed_path = os.path.join(YAML_TEMPLATE_DIR, "dify_seed_template.yml")
    if not os.path.exists(seed_path):
        raise FileNotFoundError("未找到 dify_seed_template.yml，请先从成功导入的 summary 模板复制一份")

    with open(seed_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_dify_named_seed(filename: str) -> dict:
    seed_path = os.path.join(YAML_TEMPLATE_DIR, filename)
    if not os.path.exists(seed_path):
        raise FileNotFoundError(f"未找到 {filename}")

    with open(seed_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _find_seed_node(seed_nodes: list, node_type: str) -> dict:
    for node in seed_nodes:
        data = node.get("data", {})
        if data.get("type") == node_type:
            return copy.deepcopy(node)
    raise ValueError(f"种子模板中未找到 type={node_type} 的节点")


def _find_seed_edge(seed_edges: list) -> dict:
    if not seed_edges:
        raise ValueError("种子模板中未找到 edges")
    return copy.deepcopy(seed_edges[0])


def _build_start_variables(inputs: list) -> list:
    variables = []
    for item in inputs:
        input_type = item.get("type", "text")
        dify_type = "paragraph" if input_type == "paragraph" else "text-input"

        variables.append({
            "default": "",
            "hint": "",
            "label": item.get("label", item.get("name", "input")),
            "options": [],
            "placeholder": "",
            "required": item.get("required", True),
            "type": dify_type,
            "variable": item.get("name", "input")
        })
    return variables


def _find_latest_text_source(edges: list, step_type_map: dict, current_step_id: str) -> str | None:
    """
    从当前节点往前找，找到最近一个真正能稳定输出文本的节点。
    跳过 ifelse 这种不产出 text 的路由节点。
    """
    visited = set()
    node_id = current_step_id

    while node_id and node_id not in visited:
        visited.add(node_id)

        previous_step_id = _find_previous_step_id(edges, node_id)
        if not previous_step_id:
            return None

        previous_type = step_type_map.get(previous_step_id)

        if previous_type in ("llm", "template", "knowledge_retrieval", "code", "tool", "parameter_extract", "variable_aggregator", "iteration", "loop"):
            return previous_step_id

        if previous_type == "ifelse":
            node_id = previous_step_id
            continue

        return None

    return None

def _find_latest_text_sources(edges: list, step_type_map: dict, current_step_id: str) -> list[str]:
    previous_ids = _find_previous_step_ids(edges, current_step_id)
    valid_sources = []

    for previous_step_id in previous_ids:
        previous_type = step_type_map.get(previous_step_id)
        if previous_type in ("llm", "template", "knowledge_retrieval", "code", "tool", "parameter_extract", "variable_aggregator", "iteration", "loop"):
            valid_sources.append(previous_step_id)

    return valid_sources

def _resolve_prompt_input_var(step_type: str, step_id: str) -> str:
    if step_type == "llm":
        return f"{{{{#{step_id}.text#}}}}"
    if step_type == "template":
        return f"{{{{#{step_id}.output#}}}}"
    if step_type == "knowledge_retrieval":
        return f"{{{{#{step_id}.result#}}}}"
    if step_type == "code":
        return f"{{{{#{step_id}.result#}}}}"
    if step_type == "tool":
        return f"{{{{#{step_id}.body#}}}}"
    if step_type == "parameter_extract":
        return f"{{{{#{step_id}.text#}}}}"
    if step_type == "variable_aggregator":
        return f"{{{{#{step_id}.output#}}}}"
    if step_type == "iteration":
        return f"{{{{#{step_id}.text#}}}}"
    if step_type == "loop":
        return f"{{{{#{step_id}.text#}}}}"
    return ""


def _build_llm_prompt(
    step: dict,
    inputs: list,
    edges: list,
    step_type_map: dict,
    current_step_id: str
) -> str:
    prompt = step.get("prompt", "").strip()

    if not prompt:
        prompt = "请根据用户输入完成任务。"

    input_block = []
    if inputs:
        input_block.append("用户输入：")
        for item in inputs:
            var_name = item.get("name", "input")
            input_block.append(f"- {var_name}: {{{{#start.{var_name}#}}}}")

    prev_block = []
    previous_source_ids = _find_latest_text_sources(edges, step_type_map, current_step_id)

    if previous_source_ids:
        for previous_source_id in previous_source_ids:
            previous_type = step_type_map.get(previous_source_id, "llm")
            previous_var = _resolve_prompt_input_var(previous_type, previous_source_id)
            if previous_var:
                if previous_type == "knowledge_retrieval":
                    prev_block.append(f"【背景知识（来自知识库）】:\n- {previous_source_id}: {previous_var}")
                elif previous_type == "tool":
                    prev_block.append(f"【工具调用结果】:\n- {previous_source_id}: {previous_var}")
                elif previous_type == "code":
                    prev_block.append(f"【代码计算/处理结果】:\n- {previous_source_id}: {previous_var}")
                else:
                    prev_block.append(f"【上游节点输出结果】:\n- {previous_source_id}: {previous_var}")

    final_prompt = prompt
    if input_block:
        final_prompt += "\n\n" + "\n".join(input_block)
    if prev_block:
        final_prompt += "\n\n" + "\n".join(prev_block)

    return final_prompt.strip()


def _find_previous_step_id(edges: list, current_step_id: str) -> str | None:
    for edge in edges:
        if edge.get("target") == current_step_id:
            source = edge.get("source")
            if source != "start":
                return source
    return None

def _find_previous_step_ids(edges: list, current_step_id: str) -> list[str]:
    result = []
    for edge in edges:
        if edge.get("target") == current_step_id:
            source = edge.get("source")
            if source and source != "start":
                result.append(source)
    return result

def _find_answer_source_id(edges: list, answer_step_id: str) -> str | None:
    for edge in edges:
        if edge.get("target") == answer_step_id:
            return edge.get("source")
    return None

def _resolve_output_var(step_type: str, step_id: str) -> str:
    if step_type == "llm":
        return f"{{{{#{step_id}.text#}}}}"
    if step_type == "template":
        return f"{{{{#{step_id}.output#}}}}"
    if step_type == "ifelse":
        return f"{{{{#{step_id}.result#}}}}"
    if step_type == "knowledge_retrieval":
        return f"{{{{#{step_id}.result#}}}}"
    if step_type == "code":
        return f"{{{{#{step_id}.result#}}}}"
    if step_type == "tool":
        return f"{{{{#{step_id}.body#}}}}"
    if step_type == "parameter_extract":
        return f"{{{{#{step_id}.text#}}}}"
    if step_type == "variable_aggregator":
        return f"{{{{#{step_id}.output#}}}}"
    if step_type == "iteration":
        return f"{{{{#{step_id}.text#}}}}"
    if step_type == "loop":
        return f"{{{{#{step_id}.text#}}}}"
    return f"{{{{#{step_id}.text#}}}}"


def _headers_to_dify_string(headers: Any) -> str:
    if isinstance(headers, str):
        return headers.strip()
    if isinstance(headers, dict):
        lines = []
        for k, v in headers.items():
            if k and v is not None:
                lines.append(f"{k}: {v}")
        return "\n".join(lines)
    return ""


def _resolve_tool_config(step: dict) -> dict:
    config = step.get("config", {}) if isinstance(step, dict) else {}
    if not isinstance(config, dict):
        config = {}
    method = str(config.get("method", "get")).lower().strip()
    if method not in {"get", "post", "put", "delete", "patch"}:
        method = "get"

    url = str(config.get("url", "https://httpbin.org/get")).strip()
    if not _tool_url_is_safe(url):
        url = "https://httpbin.org/get"

    headers = _headers_to_dify_string(config.get("headers"))
    params = config.get("params", "")
    if isinstance(params, dict):
        params = "&".join([f"{k}={v}" for k, v in params.items()])
    elif not isinstance(params, str):
        params = ""

    body_template = config.get("body", "")
    body = {"type": "none", "data": []}
    if isinstance(body_template, str) and body_template.strip() and method in {"post", "put", "patch"}:
        body = {"type": "raw-text", "data": body_template}

    variables = config.get("variables", [])
    if not isinstance(variables, list):
        variables = []

    return {
        "method": method,
        "url": url,
        "headers": headers,
        "params": params,
        "body": body,
        "variables": variables,
    }


def _resolve_code_config(step: dict, inputs: list) -> dict:
    config = step.get("config", {}) if isinstance(step, dict) else {}
    if not isinstance(config, dict):
        config = {}

    first_input_name = inputs[0]["name"] if inputs else "user_request"
    example_key = config.get("example_key", "")
    user_script = config.get("script", "")
    code_language = str(config.get("language", "python3")).strip() or "python3"

    code = ""
    if isinstance(user_script, str) and user_script.strip():
        code = user_script.strip() + "\n"
    elif isinstance(example_key, str) and example_key in CODE_TEMPLATE_LIBRARY:
        code = CODE_TEMPLATE_LIBRARY[example_key]
    else:
        code = CODE_TEMPLATE_LIBRARY["clean_text"]

    variables = config.get("variables", [])
    if not isinstance(variables, list) or not variables:
        variables = [{"variable": "input_text", "value_selector": ["start", first_input_name]}]

    outputs = config.get("outputs", {})
    if not isinstance(outputs, dict) or not outputs:
        outputs = {"result": {"type": "string", "children": None}}

    return {
        "code_language": code_language,
        "code": code,
        "variables": variables,
        "outputs": outputs,
    }


def _build_step_branch_map(edges: list) -> dict:
    step_branch_map = {}

    for e in edges:
        if e.get("source") == "ifelse_route" and e.get("branch"):
            step_branch_map[e["target"]] = e["branch"]

    changed = True
    while changed:
        changed = False
        for e in edges:
            source = e.get("source")
            target = e.get("target")
            if source in step_branch_map and target not in step_branch_map:
                step_branch_map[target] = step_branch_map[source]
                changed = True

    return step_branch_map


def _normalize_branch_y_map(y_base: int) -> dict:
    return {
        "option_a": y_base - 120,
        "option_b": y_base + 120,
        "marketing": y_base - 120,
        "general": y_base + 120,
        "literal": y_base - 120,
        "free": y_base + 120,
        "brief": y_base - 120,
        "detailed": y_base + 120,
        "direct": y_base - 180,
        "knowledge": y_base - 60,
        "compute": y_base + 60,
        "tool": y_base + 180,
        "retrieval": y_base + 120,
    }


def compile_workflow_spec_to_dify_yaml(spec: dict) -> str:
    """
    把通用 spec 编译成 Dify DSL
    当前支持 start / llm / ifelse / template / answer / knowledge_retrieval / code / tool 及若干扩展节点的编译映射
    """
    seed = load_dify_seed_template()

    app_name = spec.get("workflow_name", "通用工作流")
    description = spec.get("description", "由 AI 自动生成的工作流")
    inputs = spec.get("inputs", [])
    steps = spec.get("steps", [])
    edges = spec.get("edges", [])

    # 顶层 app 信息
    if "app" in seed:
        seed["app"]["name"] = app_name
        seed["app"]["description"] = description

    graph = seed.get("workflow", {}).get("graph", {})
    seed_nodes = graph.get("nodes", [])
    seed_edges = graph.get("edges", [])

    knowledge_seed = load_dify_named_seed("dify_seed_knowledge.yml")
    code_seed_file = load_dify_named_seed("dify_seed_code.yml")
    tool_seed_file = load_dify_named_seed("dify_seed_tool.yml")

    knowledge_seed_nodes = knowledge_seed.get("workflow", {}).get("graph", {}).get("nodes", [])
    code_seed_nodes = code_seed_file.get("workflow", {}).get("graph", {}).get("nodes", [])
    tool_seed_nodes = tool_seed_file.get("workflow", {}).get("graph", {}).get("nodes", [])

    start_seed = _find_seed_node(seed_nodes, "start")
    llm_seed = _find_seed_node(seed_nodes, "llm")
    ifelse_seed = _find_seed_node(seed_nodes, "if-else")
    template_seed = _find_seed_node(seed_nodes, "template-transform")
    answer_seed = _find_seed_node(seed_nodes, "answer")

    knowledge_node_seed = _find_seed_node(knowledge_seed_nodes, "knowledge-retrieval")
    code_node_seed = _find_seed_node(code_seed_nodes, "code")
    tool_node_seed = _find_seed_node(tool_seed_nodes, "http-request")

    edge_seed = _find_seed_edge(seed_edges)

    compiled_nodes = []
    compiled_edges = []
    step_id_map = {}
    step_type_map = _get_step_type_map(steps)

    x_base = 80
    x_gap = 300
    y_base = 280


    branch_y_map = _normalize_branch_y_map(y_base)

    step_branch_map = _build_step_branch_map(edges)


    # 先编译节点
    for idx, step in enumerate(steps):
        step_type = step.get("type")
        step_id = step.get("id", _new_id(step_type or "node"))
        step_title = step.get("title", step_id)

        branch_name = step_branch_map.get(step_id)
        node_y = branch_y_map.get(branch_name, y_base)

        if step_type == "start":
            node = copy.deepcopy(start_seed)
            node["id"] = step_id
            node["position"] = {"x": x_base + idx * x_gap, "y": node_y}
            node["positionAbsolute"] = {"x": x_base + idx * x_gap, "y": node_y}

            node["data"]["title"] = step_title
            node["data"]["variables"] = _build_start_variables(inputs)

        elif step_type == "llm":
            node = copy.deepcopy(llm_seed)
            node["id"] = step_id
            node["position"] = {"x": x_base + idx * x_gap, "y": node_y}
            node["positionAbsolute"] = {"x": x_base + idx * x_gap, "y": node_y}

            node["data"]["title"] = step_title

            prompt_text = _build_llm_prompt(
                step=step,
                inputs=inputs,
                edges=edges,
                step_type_map=step_type_map,
                current_step_id=step_id
            )

            if step_id == "llm_generate":
                scene = spec.get("scene", "generic")
                output_contract = spec.get("output_contract", {}) or {}
                format_prompt = build_formatting_prompt(scene, output_contract)
                prompt_text = prompt_text + "\n\n" + format_prompt

            # 兼容 prompt_template 结构
            if "prompt_template" in node["data"] and node["data"]["prompt_template"]:
                node["data"]["prompt_template"][0]["text"] = prompt_text
            else:
                node["data"]["prompt_template"] = [
                    {
                        "id": _new_id("prompt"),
                        "role": "system",
                        "text": prompt_text
                    }
                ]


        elif step_type == "ifelse":
            node = copy.deepcopy(ifelse_seed)
            node["id"] = step_id
            node["position"] = {"x": x_base + idx * x_gap, "y": node_y}
            node["positionAbsolute"] = {"x": x_base + idx * x_gap, "y": node_y}

            node["data"]["title"] = step_title

            previous_step_id = _find_previous_step_id(edges, step_id)
            if not previous_step_id:
                previous_step_id = "llm_intent"

            config = step.get("config", {}) or {}
            raw_cases = config.get("cases", [])

            node["data"]["cases"] = []

            for case in raw_cases:
                case_id = case.get("id") or case.get("case_id")
                if not case_id:
                    continue

                node["data"]["cases"].append({
                    "case_id": case_id,
                    "id": case_id,
                    "logical_operator": "or",
                    "conditions": [
                        {
                            "id": _new_id("cond"),
                            "varType": "string",
                            "comparison_operator": "contains",
                            "value": f"ROUTE={case_id}",
                            "variable_selector": [previous_step_id, "text"]
                        }
                    ]
                })

           


        elif step_type == "template":
            node = copy.deepcopy(template_seed)
            node["id"] = step_id
            node["position"] = {"x": x_base + idx * x_gap, "y": node_y}
            node["positionAbsolute"] = {"x": x_base + idx * x_gap, "y": node_y}

            node["data"]["title"] = step_title

            previous_step_id = _find_previous_step_id(edges, step_id)
            if not previous_step_id:
                previous_step_id = _pick_default_target_id(spec) or "llm_generate"
            previous_step_type = step_type_map.get(previous_step_id, "llm")
            previous_field = {
                "template": "output",
                "llm": "text",
                "knowledge_retrieval": "result",
                "code": "result",
                "tool": "body",
                "variable_aggregator": "output",
                "parameter_extract": "text",
                "iteration": "text",
                "loop": "text",
            }.get(previous_step_type, "text")

            scene = spec.get("scene", "generic")
            output_contract = spec.get("output_contract", {}) or {}
            template_text = build_output_template(scene, output_contract)
            node["data"]["template"] = template_text
            node["data"]["variables"] = [
                {
                    "variable": "arg1",
                    "value_selector": [previous_step_id, previous_field]
                }
            ]

           

        elif step_type == "answer":
            node = copy.deepcopy(answer_seed)
            node["id"] = step_id
            node["position"] = {"x": x_base + idx * x_gap, "y": node_y}
            node["positionAbsolute"] = {"x": x_base + idx * x_gap, "y": node_y}

            node["data"]["title"] = step_title

            answer_source = _find_answer_source_id(edges, step_id)
            if not answer_source:
                preferred_ids = ["template_output", "llm_review", "llm_generate"]
                existing_step_ids = {s.get("id") for s in steps}
                for pid in preferred_ids:
                    if pid in existing_step_ids:
                        answer_source = pid
                        break

            if not answer_source:
                candidate_steps = [s for s in steps if s.get("type") in ("template", "llm")]
                answer_source = candidate_steps[-1]["id"] if candidate_steps else "llm_generate"

            source_type = step_type_map.get(answer_source, "llm")
            node["data"]["answer"] = _resolve_output_var(source_type, answer_source)

        elif step_type == "knowledge_retrieval":
            node = copy.deepcopy(knowledge_node_seed)
            node["id"] = step_id
            node["position"] = {"x": x_base + idx * x_gap, "y": node_y}
            node["positionAbsolute"] = {"x": x_base + idx * x_gap, "y": node_y}

            node["data"]["title"] = step_title

            # 查询变量指向 start 的 sys.query
            first_input_name = inputs[0]["name"] if inputs else "user_query"
            node["data"]["query_variable_selector"] = ["start", first_input_name]

        elif step_type == "code":
            node = copy.deepcopy(code_node_seed)
            node["id"] = step_id
            node["position"] = {"x": x_base + idx * x_gap, "y": node_y}
            node["positionAbsolute"] = {"x": x_base + idx * x_gap, "y": node_y}

            node["data"]["title"] = step_title
            code_cfg = _resolve_code_config(step, inputs)
            node["data"]["code_language"] = code_cfg["code_language"]
            node["data"]["code"] = code_cfg["code"]
            node["data"]["variables"] = code_cfg["variables"]
            node["data"]["outputs"] = code_cfg["outputs"]
            if isinstance(step.get("config"), dict) and step["config"].get("error_strategy"):
                node["data"]["error_strategy"] = step["config"]["error_strategy"]

        elif step_type == "tool":
            node = copy.deepcopy(tool_node_seed)
            node["id"] = step_id
            node["position"] = {"x": x_base + idx * x_gap, "y": node_y}
            node["positionAbsolute"] = {"x": x_base + idx * x_gap, "y": node_y}

            node["data"]["title"] = step_title
            tool_cfg = _resolve_tool_config(step)
            node["data"]["method"] = tool_cfg["method"]
            node["data"]["url"] = tool_cfg["url"]
            node["data"]["params"] = tool_cfg["params"]
            node["data"]["headers"] = tool_cfg["headers"]
            node["data"]["body"] = tool_cfg["body"]
            node["data"]["variables"] = tool_cfg["variables"]
            if isinstance(step.get("config"), dict) and step["config"].get("error_strategy"):
                node["data"]["error_strategy"] = step["config"]["error_strategy"]

        elif step_type == "variable_aggregator":
            node = copy.deepcopy(code_node_seed)
            node["id"] = step_id
            node["position"] = {"x": x_base + idx * x_gap, "y": node_y}
            node["positionAbsolute"] = {"x": x_base + idx * x_gap, "y": node_y}

            node["data"]["title"] = step_title

            incoming_edges = [e for e in edges if isinstance(e, dict) and e.get("target") == step_id]
            variables = []
            arg_names = []
            for i, e in enumerate(incoming_edges):
                src = e.get("source")
                if not src:
                    continue
                src_type = step_type_map.get(src, "llm")
                src_field = {
                    "template": "output",
                    "llm": "text",
                    "knowledge_retrieval": "result",
                    "code": "result",
                    "tool": "body",
                    "variable_aggregator": "output",
                    "parameter_extract": "text",
                    "iteration": "text",
                    "loop": "text",
                }.get(src_type, "text")
                arg = f"arg{i + 1}"
                arg_names.append(arg)
                variables.append({"variable": arg, "value_selector": [src, src_field]})

            node["data"]["variables"] = variables
            node["data"]["outputs"] = {"output": {"type": "string", "children": None}}

            params = ", ".join([f"{name}: str = \"\"" for name in arg_names]) if arg_names else ""
            lines = []
            lines.append("import json")
            lines.append("")
            lines.append(f"def main({params}):")
            lines.append("    values = []")
            for name in arg_names:
                lines.append(f"    values.append({name})")
            lines.append("    for v in values:")
            lines.append("        if v is None:")
            lines.append("            continue")
            lines.append("        if isinstance(v, str) and v.strip():")
            lines.append("            return {\"output\": v}")
            lines.append("        if not isinstance(v, str):")
            lines.append("            try:")
            lines.append("                s = json.dumps(v, ensure_ascii=False)")
            lines.append("                if isinstance(s, str) and s.strip():")
            lines.append("                    return {\"output\": s}")
            lines.append("            except Exception:")
            lines.append("                pass")
            lines.append("    return {\"output\": \"\"}")

            node["data"]["code_language"] = "python3"
            node["data"]["code"] = "\n" + "\n".join(lines) + "\n"

        elif step_type in {"parameter_extract", "iteration", "loop"}:
            node = copy.deepcopy(llm_seed)
            node["id"] = step_id
            node["position"] = {"x": x_base + idx * x_gap, "y": node_y}
            node["positionAbsolute"] = {"x": x_base + idx * x_gap, "y": node_y}

            node["data"]["title"] = step_title
            prompt_text = _build_llm_prompt(
                step=step,
                inputs=inputs,
                edges=edges,
                step_type_map=step_type_map,
                current_step_id=step_id
            )
            if step_type == "parameter_extract":
                cfg = step.get("config", {}) if isinstance(step.get("config"), dict) else {}
                schema_text = json.dumps(cfg.get("schema", {}), ensure_ascii=False)
                prompt_text = prompt_text + "\n\n" + f"请把关键信息提取为严格 JSON（只输出 JSON）：\n{schema_text}"
            if step_type == "iteration":
                prompt_text = prompt_text + "\n\n" + "请对输入中的数组/列表逐项处理，输出 JSON 数组（只输出 JSON）。"
            if step_type == "loop":
                prompt_text = prompt_text + "\n\n" + "请进行最多 3 轮自我改进：每轮先指出缺陷再给出改进版本。最终只输出最终版本。"

            if "prompt_template" in node["data"] and node["data"]["prompt_template"]:
                node["data"]["prompt_template"][0]["text"] = prompt_text
            else:
                node["data"]["prompt_template"] = [{"id": _new_id("prompt"), "role": "system", "text": prompt_text}]

        else:
            continue

        step_id_map[step_id] = step_id
        compiled_nodes.append(node)

         # 再编译边
    for e in edges:
        source = e.get("source")
        target = e.get("target")

        if source not in step_id_map or target not in step_id_map:
            continue

        source_step_type = step_type_map.get(source, "llm")
        target_step_type = step_type_map.get(target, "llm")

        edge = copy.deepcopy(edge_seed)

        source_handle = _resolve_source_handle(source_step_type, e)
        target_handle = _resolve_target_handle(target_step_type)

        edge["id"] = f"{source}-{source_handle}-{target}-{target_handle}"
        edge["source"] = source
        edge["target"] = target
        edge["sourceHandle"] = source_handle
        edge["targetHandle"] = target_handle
        edge["type"] = "custom"

        if "data" not in edge or not isinstance(edge["data"], dict):
            edge["data"] = {}

        edge["data"]["sourceType"] = _to_dify_node_type(source_step_type)
        edge["data"]["targetType"] = _to_dify_node_type(target_step_type)
        edge["data"]["isInIteration"] = False
        edge["data"]["isInLoop"] = False

        compiled_edges.append(edge)

    # 写回 graph
    seed["workflow"]["graph"]["nodes"] = compiled_nodes
    seed["workflow"]["graph"]["edges"] = compiled_edges

    return yaml.safe_dump(
        seed,
        allow_unicode=True,
        sort_keys=False
    )


def validate_compiled_dify_yaml(yaml_content: str) -> tuple[bool, str]:
    try:
        parsed = yaml.safe_load(yaml_content)
    except Exception as e:
        return False, f"YAML 解析失败: {str(e)}"

    if not isinstance(parsed, dict):
        return False, "YAML 顶层结构非法"

    workflow = parsed.get("workflow", {})
    graph = workflow.get("graph", {}) if isinstance(workflow, dict) else {}
    nodes = graph.get("nodes", []) if isinstance(graph, dict) else []
    edges = graph.get("edges", []) if isinstance(graph, dict) else []
    if not isinstance(nodes, list) or not nodes:
        return False, "编译结果缺少 graph.nodes"
    if not isinstance(edges, list):
        return False, "编译结果 graph.edges 非数组"

    ids = set()
    nodes_by_id: dict[str, dict] = {}
    for idx, node in enumerate(nodes):
        if not isinstance(node, dict):
            return False, f"graph.nodes[{idx}] 非对象"
        node_id = node.get("id")
        if not node_id:
            return False, f"graph.nodes[{idx}] 缺少 id"
        ids.add(node_id)
        if isinstance(node_id, str):
            nodes_by_id[node_id] = node

    for idx, edge in enumerate(edges):
        if not isinstance(edge, dict):
            return False, f"graph.edges[{idx}] 非对象"
        if edge.get("source") not in ids or edge.get("target") not in ids:
            return False, f"graph.edges[{idx}] 引用了不存在节点"

    outgoing: dict[str, list[dict]] = {}
    for edge in edges:
        if isinstance(edge, dict) and edge.get("source"):
            outgoing.setdefault(edge["source"], []).append(edge)

    for node_id, node in nodes_by_id.items():
        data = node.get("data", {}) if isinstance(node, dict) else {}
        if not isinstance(data, dict):
            continue
        if data.get("error_strategy") == "fail-branch":
            outs = outgoing.get(node_id, [])
            handles = {str(e.get("sourceHandle") or "source") for e in outs if isinstance(e, dict)}
            if "success-branch" not in handles or "fail-branch" not in handles:
                return False, f"节点 {node_id} error_strategy=fail-branch 但缺少 success/fail 分支连边"

    return True, ""


def _resolve_source_handle(source_step_type: str, edge_info: dict | None = None) -> str:
    if source_step_type == "ifelse":
        branch = (edge_info or {}).get("branch")
        return branch or "direct"
    handle = (edge_info or {}).get("source_handle")
    if handle:
        return str(handle)
    branch = (edge_info or {}).get("branch")
    if branch:
        return str(branch)
    return "source"

def _resolve_target_handle(target_step_type: str) -> str:
    return "target"


def _to_dify_node_type(step_type: str) -> str:
    mapping = {
        "start": "start",
        "llm": "llm",
        "ifelse": "if-else",
        "template": "template-transform",
        "answer": "answer",
        "knowledge_retrieval": "knowledge-retrieval",
        "code": "code",
        "tool": "http-request",
        "parameter_extract": "llm",
        "variable_aggregator": "code",
        "iteration": "llm",
        "loop": "llm",
    }
    return mapping.get(step_type, "llm")


def _import_yaml_to_dify(yaml_content: str) -> tuple[bool, dict]:
    if not (DIFY_API_URL and DIFY_API_KEY):
        return False, {"error": "DIFY_API_URL 或 DIFY_API_KEY 未配置"}
    endpoints = [
        "/console/api/apps/imports",
        "/console/api/apps/uploads/imports",
    ]
    headers = {"Authorization": f"Bearer {DIFY_API_KEY}"}
    last_error = None

    for path in endpoints:
        url = f"{DIFY_API_URL}{path}"
        try:
            response = requests.post(
                url,
                headers={**headers, "Content-Type": "application/x-yaml"},
                data=yaml_content.encode("utf-8"),
                timeout=20,
            )
            if response.status_code >= 400:
                last_error = {
                    "endpoint": path,
                    "status_code": response.status_code,
                    "body": response.text,
                }
                continue
            try:
                payload = response.json()
            except Exception:
                payload = {"raw": response.text}

            job_id = (
                payload.get("jobId")
                or payload.get("job_id")
                or payload.get("id")
                or payload.get("data", {}).get("jobId")
                if isinstance(payload, dict)
                else None
            )
            return True, {
                "endpoint": path,
                "status_code": response.status_code,
                "job_id": job_id,
                "payload": payload,
            }
        except Exception as e:
            last_error = {"endpoint": path, "error": str(e)}

    return False, {"error": "Dify import failed", "detail": last_error}


@app.route("/generate", methods=["POST"])
def generate():
    started = time.time()
    trace_id = request.headers.get("X-Trace-Id") or f"trace_{uuid.uuid4().hex[:10]}"
    RUNTIME_METRICS["generate_total"] += 1
    data = request.get_json() or {}
    scene_key = data.get("scene_key", "unknown")
    app_name = data.get("app_name", "").strip()
    description = data.get("description", "").strip()
    workflow_spec = data.get("workflow_spec")
    requirement = data.get("requirement")
    strict_validation = bool(data.get("strict_validation", True))
    import_mode = str(data.get("import_mode", "download_only")).strip().lower()
    auto_import_to_dify = bool(data.get("auto_import_to_dify", False) or import_mode == "download_and_import")

    if not model_ready():
        RUNTIME_METRICS["generate_fail"] += 1
        RUNTIME_METRICS["generate_latency_ms"].append((time.time() - started) * 1000)
        payload = _model_required_error_payload()
        payload["trace_id"] = trace_id
        log_event("generate.blocked", trace_id, reason="model_unavailable")
        return jsonify(payload), 503

    if not workflow_spec or not isinstance(workflow_spec, dict):
        RUNTIME_METRICS["generate_fail"] += 1
        RUNTIME_METRICS["generate_latency_ms"].append((time.time() - started) * 1000)
        return jsonify({"error": "当前没有可生成的工作流 spec，请先点击分析需求。", "trace_id": trace_id}), 400

    base_spec = copy.deepcopy(workflow_spec)

    if requirement and isinstance(requirement, dict):
        base_spec["requirement_meta"] = requirement

    spec = apply_user_config_to_spec(base_spec, data)

    if app_name:
        spec["workflow_name"] = app_name
    if description:
        spec["description"] = description

    validation = validate_workflow_spec_v2(spec, MAX_NODE_LIMIT)
    ok, err = validate_workflow_spec(spec)
    if strict_validation and (not ok):
        RUNTIME_METRICS["generate_fail"] += 1
        RUNTIME_METRICS["generate_latency_ms"].append((time.time() - started) * 1000)
        return jsonify({"error": f"生成前工作流结构无效：{err}", "trace_id": trace_id}), 400

    try:
        log_event("generate.start", trace_id, node_count=len(spec.get("steps", [])), edge_count=len(spec.get("edges", [])))
        compile_started = time.time()
        yaml_content = compile_workflow_spec_to_dify_yaml(spec)
        RUNTIME_METRICS["compile_latency_ms"].append((time.time() - compile_started) * 1000)
        yaml_ok, yaml_err = validate_compiled_dify_yaml(yaml_content)
        if not yaml_ok:
            raise ValueError(f"编译结果 YAML 校验失败：{yaml_err}")
        RUNTIME_METRICS["compile_success"] += 1
    except Exception as e:
        RUNTIME_METRICS["compile_fail"] += 1
        RUNTIME_METRICS["generate_fail"] += 1
        RUNTIME_METRICS["generate_latency_ms"].append((time.time() - started) * 1000)
        log_event("generate.compile_error", trace_id, error=str(e))
        return jsonify({"error": f"编译失败：{str(e)}", "trace_id": trace_id}), 500

    filename = f"{scene_key}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.yml"
    filepath = os.path.join(GENERATED_DIR, filename)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(yaml_content)

    dify_import = None
    dify_import_status = "not_requested"
    if auto_import_to_dify:
        import_started = time.time()
        imported, payload = _import_yaml_to_dify(yaml_content)
        RUNTIME_METRICS["import_latency_ms"].append((time.time() - import_started) * 1000)
        dify_import = payload
        if imported:
            RUNTIME_METRICS["dify_import_success"] += 1
            dify_import_status = "success"
        else:
            RUNTIME_METRICS["dify_import_fail"] += 1
            dify_import_status = "failed"
        log_event("generate.import_done", trace_id, import_status=dify_import_status)

    RUNTIME_METRICS["generate_success"] += 1
    RUNTIME_METRICS["generate_latency_ms"].append((time.time() - started) * 1000)

    response = {
        "message": "YAML 生成成功",
        "filename": filename,
        "download_url": f"/download/{filename}",
        "yaml_content": yaml_content,
        "workflow_spec": spec,
        "requirement": spec.get("requirement_meta", {}),
        "planning_mode": spec.get("meta", {}).get("planning_mode", PLANNING_MODE),
        "validation_warnings": validation.get("warnings", []),
        "dify_import": dify_import,
        "dify_import_status": dify_import_status,
        "dify_console_url": DIFY_CONSOLE_URL or DIFY_API_URL,
        "import_mode": "download_and_import" if auto_import_to_dify else "download_only",
        "trace_id": trace_id,
    }

    return jsonify(response)

@app.route("/download/<path:filename>")
def download_file(filename):
    return send_from_directory(GENERATED_DIR, filename, as_attachment=True)


if __name__ == "__main__":
    app.run(debug=True)

