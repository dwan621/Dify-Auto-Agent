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
    normalized, _ = auto_app.normalize_workflow_spec_v2(
        spec,
        {"runtime_config": {"allow_autofix_ifelse": True}},
    )
    branches = {e.get("branch") for e in normalized.get("edges", []) if e.get("source") == "route"}
    assert "a" in branches
    assert "b" in branches


def test_normalize_strict_mode_keeps_missing_ifelse_edges():
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
    normalized, warnings = auto_app.normalize_workflow_spec_v2(spec, {})
    branches = {e.get("branch") for e in normalized.get("edges", []) if e.get("source") == "route"}
    assert "a" in branches
    assert "b" not in branches
    assert any("严格模式未自动补边" in w for w in warnings)


def test_normalize_edge_semantics_for_fail_branch():
    spec = {
        "steps": [
            {"id": "start", "type": "start"},
            {"id": "tool_1", "type": "tool", "config": {"method": "get", "url": "https://httpbin.org/get"}},
            {"id": "llm_1", "type": "llm"},
            {"id": "answer", "type": "answer"},
        ],
        "edges": [
            {"source": "start", "target": "tool_1"},
            {"source": "tool_1", "target": "llm_1", "branch": "success-branch"},
            {"source": "tool_1", "target": "answer", "branch": "fail-branch"},
            {"source": "llm_1", "target": "answer"},
        ],
    }
    normalized, _ = auto_app.normalize_workflow_spec_v2(spec, {})
    tool_edges = [e for e in normalized["edges"] if e.get("source") == "tool_1"]
    assert any(e.get("source_handle") == "success-branch" for e in tool_edges)
    assert any(e.get("source_handle") == "fail-branch" for e in tool_edges)


def test_validate_answer_rejects_multi_incoming():
    spec = {
        "steps": [
            {"id": "start", "type": "start"},
            {"id": "llm_a", "type": "llm"},
            {"id": "llm_b", "type": "llm"},
            {"id": "answer", "type": "answer"},
        ],
        "edges": [
            {"source": "start", "target": "llm_a"},
            {"source": "start", "target": "llm_b"},
            {"source": "llm_a", "target": "answer"},
            {"source": "llm_b", "target": "answer"},
        ],
    }
    result = auto_app.validate_workflow_spec_v2(spec)
    assert not result["ok"]
    assert any("answer 节点" in msg and "多入边" in msg for msg in result["errors"])


def test_compile_ifelse_uses_route_marker_value():
    spec = {
        "workflow_name": "ifelse route marker",
        "description": "route marker",
        "scene": "generic",
        "inputs": [{"name": "user_request", "label": "输入", "type": "paragraph", "required": True}],
        "steps": [
            {"id": "start", "type": "start", "title": "开始"},
            {"id": "llm_plan", "type": "llm", "title": "规划", "prompt": "ROUTE=knowledge"},
            {
                "id": "ifelse_route",
                "type": "ifelse",
                "title": "路由",
                "config": {"cases": [{"id": "knowledge"}, {"id": "direct"}]},
            },
            {"id": "llm_generate", "type": "llm", "title": "生成", "prompt": "生成结果"},
            {"id": "answer", "type": "answer", "title": "回复"},
        ],
        "edges": [
            {"source": "start", "target": "llm_plan"},
            {"source": "llm_plan", "target": "ifelse_route"},
            {"source": "ifelse_route", "target": "llm_generate", "branch": "knowledge"},
            {"source": "ifelse_route", "target": "llm_generate", "branch": "direct"},
            {"source": "llm_generate", "target": "answer"},
        ],
    }
    yaml_content = auto_app.compile_workflow_spec_to_dify_yaml(spec)
    parsed = yaml.safe_load(yaml_content)
    ifelse_node = next(node for node in parsed["workflow"]["graph"]["nodes"] if node["id"] == "ifelse_route")
    values = [cond["value"] for case in ifelse_node["data"]["cases"] for cond in case.get("conditions", [])]
    assert "ROUTE=knowledge" in values
    assert "ROUTE=direct" in values


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


def test_compile_llm_uses_config_prompt_and_converts_placeholders():
    spec = {
        "workflow_name": "prompt",
        "scene": "generic",
        "inputs": [{"name": "user_request", "label": "输入", "type": "paragraph", "required": True}],
        "steps": [
            {"id": "start", "type": "start"},
            {"id": "llm_1", "type": "llm", "config": {"prompt": "请处理：{{inputs.user_request}}"}},
            {"id": "answer", "type": "answer"},
        ],
        "edges": [
            {"source": "start", "target": "llm_1"},
            {"source": "llm_1", "target": "answer"},
        ],
    }
    yaml_content = auto_app.compile_workflow_spec_to_dify_yaml(spec)
    parsed = yaml.safe_load(yaml_content)
    llm_node = next(node for node in parsed["workflow"]["graph"]["nodes"] if node["id"] == "llm_1")
    text = llm_node["data"]["prompt_template"][0]["text"]
    assert "{{#start.user_request#}}" in text


def test_compile_code_coerces_python_to_python3():
    spec = {
        "workflow_name": "code language",
        "scene": "generic",
        "inputs": [{"name": "user_request", "label": "输入", "type": "paragraph", "required": True}],
        "steps": [
            {"id": "start", "type": "start"},
            {"id": "code_1", "type": "code", "config": {"language": "python", "example_key": "clean_text"}},
            {"id": "answer", "type": "answer"},
        ],
        "edges": [
            {"source": "start", "target": "code_1"},
            {"source": "code_1", "target": "answer"},
        ],
    }
    yaml_content = auto_app.compile_workflow_spec_to_dify_yaml(spec)
    parsed = yaml.safe_load(yaml_content)
    code_node = next(node for node in parsed["workflow"]["graph"]["nodes"] if node["id"] == "code_1")
    assert code_node["data"]["code_language"] == "python3"


def test_compile_ifelse_adds_cases_for_outgoing_branches():
    spec = {
        "workflow_name": "ifelse",
        "scene": "generic",
        "steps": [
            {"id": "start", "type": "start"},
            {"id": "route", "type": "ifelse", "config": {"cases": [{"id": "a"}]}},
            {"id": "llm_a", "type": "llm", "prompt": "ROUTE=a"},
            {"id": "answer", "type": "answer"},
        ],
        "edges": [
            {"source": "start", "target": "route"},
            {"source": "route", "target": "llm_a", "branch": "a"},
            {"source": "route", "target": "answer", "branch": "false"},
            {"source": "llm_a", "target": "answer"},
        ],
    }
    yaml_content = auto_app.compile_workflow_spec_to_dify_yaml(spec)
    parsed = yaml.safe_load(yaml_content)
    ifelse_node = next(node for node in parsed["workflow"]["graph"]["nodes"] if node["id"] == "route")
    case_ids = {str(c.get("case_id") or c.get("id")) for c in ifelse_node["data"]["cases"]}
    assert "a" in case_ids
    assert "false" in case_ids


def test_normalize_removes_fail_branch_error_strategy_without_handles():
    spec = {
        "workflow_name": "wrap",
        "scene": "generic",
        "steps": [
            {"id": "start", "type": "start"},
            {"id": "tool_1", "type": "tool", "config": {"method": "get", "url": "https://httpbin.org/get", "error_strategy": "fail-branch"}},
            {"id": "answer", "type": "answer"},
        ],
        "edges": [
            {"source": "start", "target": "tool_1"},
            {"source": "tool_1", "target": "answer"},
        ],
    }
    normalized, _ = auto_app.normalize_workflow_spec_v2(spec, {})
    tool_step = next(s for s in normalized["steps"] if s["id"] == "tool_1")
    assert tool_step.get("config", {}).get("error_strategy") != "fail-branch"
    assert all(not str(s.get("id", "")).startswith("llm_fallback") for s in normalized["steps"])


def test_normalize_prunes_fail_branch_wrapper_with_multi_targets():
    spec = {
        "workflow_name": "wrap-multi",
        "scene": "generic",
        "steps": [
            {"id": "start", "type": "start"},
            {"id": "tool_1", "type": "tool", "config": {"method": "get", "url": "https://httpbin.org/get", "error_strategy": "fail-branch"}},
            {"id": "llm_fallback_x", "type": "llm", "title": "失败兜底", "prompt": "上游执行失败。请给出替代结果。"},
            {"id": "agg_x", "type": "variable_aggregator", "config": {}},
            {"id": "llm_1", "type": "llm", "prompt": "处理结果"},
            {"id": "answer", "type": "answer"},
        ],
        "edges": [
            {"source": "start", "target": "tool_1"},
            {"source": "tool_1", "target": "agg_x", "source_handle": "success-branch"},
            {"source": "tool_1", "target": "llm_fallback_x", "source_handle": "fail-branch"},
            {"source": "llm_fallback_x", "target": "agg_x"},
            {"source": "agg_x", "target": "llm_1"},
            {"source": "agg_x", "target": "answer"},
            {"source": "llm_1", "target": "answer"},
        ],
    }
    normalized, _ = auto_app.normalize_workflow_spec_v2(spec, {})
    ids = {s["id"] for s in normalized["steps"]}
    assert "llm_fallback_x" not in ids
    assert "agg_x" not in ids
    tool_step = next(s for s in normalized["steps"] if s["id"] == "tool_1")
    assert tool_step.get("config", {}).get("error_strategy") != "fail-branch"
    tool_out_targets = {e.get("target") for e in normalized["edges"] if e.get("source") == "tool_1"}
    assert "llm_1" in tool_out_targets
    assert "answer" in tool_out_targets


def test_normalize_prunes_ifelse_edges_without_branch_or_not_in_cases():
    spec = {
        "workflow_name": "ifelse-prune",
        "scene": "generic",
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
            {"source": "route", "target": "llm_b"},
            {"source": "route", "target": "answer", "branch": "direct"},
            {"source": "llm_a", "target": "answer"},
            {"source": "llm_b", "target": "answer"},
        ],
    }
    normalized, warnings = auto_app.normalize_workflow_spec_v2(spec, {})
    route_outs = [e for e in normalized.get("edges", []) if e.get("source") == "route"]
    assert all(e.get("branch") in {"a", "b"} for e in route_outs if e.get("branch"))
    assert all(e.get("branch") for e in route_outs)
    assert any("已移除 ifelse route 的无 branch 出边" in w for w in warnings)
    assert any("已移除 ifelse route 的无效分支出边" in w for w in warnings)


def test_normalize_limits_fail_branch_nodes_to_two():
    spec = {
        "workflow_name": "fail-branch-limit",
        "scene": "generic",
        "steps": [
            {"id": "start", "type": "start"},
            {"id": "code_1", "type": "code", "config": {"error_strategy": "fail-branch"}},
            {"id": "code_2", "type": "code", "config": {"error_strategy": "fail-branch"}},
            {"id": "tool_1", "type": "tool", "config": {"method": "get", "url": "https://httpbin.org/get", "error_strategy": "fail-branch"}},
            {"id": "tool_2", "type": "tool", "config": {"method": "get", "url": "https://httpbin.org/get", "error_strategy": "fail-branch"}},
            {"id": "answer", "type": "answer"},
        ],
        "edges": [
            {"source": "start", "target": "code_1"},
            {"source": "code_1", "target": "code_2"},
            {"source": "code_2", "target": "tool_1"},
            {"source": "tool_1", "target": "tool_2"},
            {"source": "tool_2", "target": "answer"},
        ],
    }
    normalized, _ = auto_app.normalize_workflow_spec_v2(spec, {"runtime_config": {"max_fail_branch_nodes": 2}})
    steps = normalized.get("steps", [])
    fail_nodes = [
        s.get("id") for s in steps
        if isinstance(s, dict)
        and s.get("type") in {"tool", "code"}
        and isinstance(s.get("config"), dict)
        and s["config"].get("error_strategy") == "fail-branch"
    ]
    assert len(fail_nodes) <= 2
