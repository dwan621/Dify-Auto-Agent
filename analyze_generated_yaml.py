import json
import sys
from pathlib import Path

import yaml

import app as auto_app


def analyze(path: Path) -> dict:
    content = path.read_text(encoding="utf-8")
    ok, err = auto_app.validate_compiled_dify_yaml(content)
    parsed = yaml.safe_load(content)

    graph = (parsed.get("workflow") or {}).get("graph") or {}
    nodes = graph.get("nodes") or []
    edges = graph.get("edges") or []

    nodes_by_id = {n.get("id"): n for n in nodes if isinstance(n, dict) and n.get("id")}
    ids = set(nodes_by_id)

    outgoing: dict[str, list[dict]] = {}
    incoming: dict[str, list[dict]] = {}
    for e in edges:
        if not isinstance(e, dict):
            continue
        src = e.get("source")
        tgt = e.get("target")
        if src:
            outgoing.setdefault(str(src), []).append(e)
        if tgt:
            incoming.setdefault(str(tgt), []).append(e)

    problems: list[dict] = []

    for e in edges:
        if not isinstance(e, dict):
            problems.append({"type": "edge_not_object"})
            continue
        if e.get("source") not in ids or e.get("target") not in ids:
            problems.append({"type": "edge_refs_missing_node", "edge_id": e.get("id")})

    ifelse_invalid: list[dict] = []
    for nid, node in nodes_by_id.items():
        data = node.get("data") if isinstance(node.get("data"), dict) else {}
        if data.get("type") != "if-else":
            continue
        case_ids = {
            str(c.get("case_id") or c.get("id"))
            for c in (data.get("cases") or [])
            if isinstance(c, dict) and (c.get("case_id") or c.get("id"))
        }
        for e in outgoing.get(nid, []):
            if not isinstance(e, dict):
                continue
            handle = str(e.get("sourceHandle") or "")
            if handle and handle not in case_ids:
                ifelse_invalid.append({"ifelse": nid, "sourceHandle": handle, "edge_id": e.get("id")})
    if ifelse_invalid:
        problems.append({"type": "ifelse_invalid_sourceHandle", "items": ifelse_invalid})

    fail_nodes = []
    for nid, node in nodes_by_id.items():
        data = node.get("data") if isinstance(node.get("data"), dict) else {}
        if data.get("error_strategy") == "fail-branch":
            fail_nodes.append(nid)

    fail_edges = [
        e for e in edges
        if isinstance(e, dict) and str(e.get("sourceHandle") or "") in {"success-branch", "fail-branch"}
    ]

    spec_placeholder_llm_nodes = []
    for nid, node in nodes_by_id.items():
        data = node.get("data") if isinstance(node.get("data"), dict) else {}
        if data.get("type") != "llm":
            continue
        for pt in (data.get("prompt_template") or []):
            if not isinstance(pt, dict):
                continue
            text = str(pt.get("text") or "")
            if "{{inputs." in text or "{{steps." in text:
                spec_placeholder_llm_nodes.append(nid)
                break

    start_vars = []
    start = nodes_by_id.get("start")
    if isinstance(start, dict):
        data = start.get("data") if isinstance(start.get("data"), dict) else {}
        start_vars = [v.get("variable") for v in (data.get("variables") or []) if isinstance(v, dict)]

    return {
        "file": str(path),
        "validate_compiled_dify_yaml": {"ok": ok, "error": err},
        "counts": {"nodes": len(nodes), "edges": len(edges)},
        "start_variables": start_vars,
        "fail_branch": {
            "error_strategy_nodes": fail_nodes,
            "branch_edge_count": len(fail_edges),
        },
        "llm_nodes_with_spec_placeholders": sorted(set(spec_placeholder_llm_nodes)),
        "problems": problems,
    }


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python analyze_generated_yaml.py <path-to-yml>")
        return 2
    path = Path(sys.argv[1])
    report = analyze(path)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["validate_compiled_dify_yaml"]["ok"] and not report["problems"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

