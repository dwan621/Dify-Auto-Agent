import yaml

import app as auto_app


def test_validate_blocks_dangerous_tool_url():
    spec = {
        "steps": [
            {"id": "start", "type": "start"},
            {"id": "tool_1", "type": "tool", "config": {"method": "get", "url": "https://evil.com/api"}},
            {"id": "answer", "type": "answer"},
        ],
        "edges": [
            {"source": "start", "target": "tool_1"},
            {"source": "tool_1", "target": "answer"},
        ],
    }
    result = auto_app.validate_workflow_spec_v2(spec)
    assert not result["ok"]
    assert any("不安全" in msg for msg in result["errors"])


def test_deterministic_fix_adds_required_nodes():
    spec = {"steps": [{"id": "n1", "type": "llm"}], "edges": []}
    fixed, warnings = auto_app.deterministic_fix_workflow_spec_v2(spec)
    ids = {item["id"] for item in fixed["steps"]}
    assert "start" in ids
    assert "answer" in ids
    assert warnings


def test_prune_exec_nodes_by_budget():
    spec = {
        "steps": [
            {"id": "start", "type": "start"},
            {"id": "kb", "type": "knowledge_retrieval"},
            {"id": "tool", "type": "tool", "config": {"method": "get", "url": "https://httpbin.org/get"}},
            {"id": "code", "type": "code", "config": {"example_key": "clean_text"}},
            {"id": "answer", "type": "answer"},
        ],
        "edges": [
            {"source": "start", "target": "kb"},
            {"source": "kb", "target": "tool"},
            {"source": "tool", "target": "code"},
            {"source": "code", "target": "answer"},
        ],
    }
    pruned = auto_app._prune_exec_nodes(spec, max_exec_nodes=1, budget_level="low")
    exec_nodes = [s for s in pruned["steps"] if s["type"] in {"knowledge_retrieval", "tool", "code"}]
    assert len(exec_nodes) == 1


def test_compile_workflow_spec_to_dify_yaml_passes_schema_check():
    spec = {
        "workflow_name": "unit test workflow",
        "description": "unit test",
        "scene": "generic",
        "inputs": [{"name": "user_request", "label": "输入", "type": "paragraph", "required": True}],
        "steps": [
            {"id": "start", "type": "start", "title": "开始"},
            {"id": "tool_1", "type": "tool", "title": "工具", "config": {"method": "get", "url": "https://httpbin.org/get"}},
            {"id": "llm_1", "type": "llm", "title": "生成", "prompt": "总结上游结果"},
            {"id": "answer", "type": "answer", "title": "回复"},
        ],
        "edges": [
            {"source": "start", "target": "tool_1"},
            {"source": "tool_1", "target": "llm_1"},
            {"source": "llm_1", "target": "answer"},
        ],
    }
    yaml_content = auto_app.compile_workflow_spec_to_dify_yaml(spec)
    ok, err = auto_app.validate_compiled_dify_yaml(yaml_content)
    assert ok, err

    parsed = yaml.safe_load(yaml_content)
    assert parsed["app"]["name"] == "unit test workflow"


def test_validate_ifelse_requires_branch_edges():
    spec = {
        "steps": [
            {"id": "start", "type": "start"},
            {"id": "route", "type": "ifelse", "config": {"cases": [{"id": "a"}, {"id": "b"}]}},
            {"id": "llm_a", "type": "llm"},
            {"id": "llm_b", "type": "llm"},
            {"id": "answer", "type": "answer"},
        ],
        "edges": [
            {"source": "start", "target": "route"},
            {"source": "route", "target": "llm_a", "branch": "a"},
            {"source": "llm_a", "target": "answer"},
            {"source": "llm_b", "target": "answer"},
        ],
    }
    result = auto_app.validate_workflow_spec_v2(spec)
    assert not result["ok"]
    assert any("缺少分支连边" in msg for msg in result["errors"])


def test_normalize_fills_missing_ifelse_edges():
    spec = {
        "workflow_name": "t",
        "scene": "generic",
        "steps": [
            {"id": "start", "type": "start"},
            {"id": "route", "type": "ifelse", "config": {"cases": [{"id": "a"}, {"id": "b"}]}},
            {"id": "llm_a", "type": "llm"},
            {"id": "answer", "type": "answer"},
        ],
        "edges": [
            {"source": "start", "target": "route"},
            {"source": "route", "target": "llm_a", "branch": "a"},
            {"source": "llm_a", "target": "answer"},
        ],
    }
    normalized, _ = auto_app.normalize_workflow_spec_v2(spec, {})
    branches = {e.get("branch") for e in normalized.get("edges", []) if e.get("source") == "route"}
    assert "a" in branches
    assert "b" in branches


def test_normalize_wraps_tool_with_fail_branch_and_compiles():
    spec = {
        "workflow_name": "unit test workflow",
        "description": "unit test",
        "scene": "generic",
        "inputs": [{"name": "user_request", "label": "输入", "type": "paragraph", "required": True}],
        "steps": [
            {"id": "start", "type": "start", "title": "开始"},
            {"id": "tool_1", "type": "tool", "title": "工具", "config": {"method": "get", "url": "https://httpbin.org/get"}},
            {"id": "llm_1", "type": "llm", "title": "生成", "prompt": "总结上游结果"},
            {"id": "answer", "type": "answer", "title": "回复"},
        ],
        "edges": [
            {"source": "start", "target": "tool_1"},
            {"source": "tool_1", "target": "llm_1"},
            {"source": "llm_1", "target": "answer"},
        ],
    }
    normalized, _ = auto_app.normalize_workflow_spec_v2(spec, {})
    result = auto_app.validate_workflow_spec_v2(normalized)
    assert result["ok"], result["errors"]

    yaml_content = auto_app.compile_workflow_spec_to_dify_yaml(normalized)
    ok, err = auto_app.validate_compiled_dify_yaml(yaml_content)
    assert ok, err
