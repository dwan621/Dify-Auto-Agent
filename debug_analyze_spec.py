import json

import requests


def main() -> int:
    prompt = """
请生成一个用于“每日定时处理 FTP 上 CSV 日志”的 Dify 工作流（本地测试不需要真实连接外部系统，尽量用 code 节点 mock 数据/结果）。

流程：读取 CSV -> 清洗 -> IP 地理位置补全（mock）-> 入库（mock）-> 风险判断（ifelse: high_risk/normal）-> 报表总结（llm）-> 汇总输出（answer）。

约束：图连通，节点总数 10~14；默认不要使用 fail-branch 兜底链路。

start 输入：ftp_config, db_connection, api_key, monitoring_platform_url, alert_recipients(可选)。

answer 输出字段：run_id, processed_files, inserted_rows, anomalies, alerts, report, status。
""".strip()

    r = requests.post("http://127.0.0.1:5000/analyze", json={"user_input": prompt}, timeout=300)
    print("status:", r.status_code)
    body = r.json()
    if r.status_code != 200:
        print(body)
        return 1

    spec = body["workflow_spec"]
    steps = spec.get("steps", [])
    edges = spec.get("edges", [])
    step_by_id = {s.get("id"): s for s in steps if isinstance(s, dict) and s.get("id")}
    step_type = {sid: step_by_id[sid].get("type") for sid in step_by_id}

    fail_branch_nodes = []
    for s in steps:
        if not isinstance(s, dict):
            continue
        if s.get("type") not in {"tool", "code"}:
            continue
        cfg = s.get("config", {}) if isinstance(s.get("config"), dict) else {}
        if cfg.get("error_strategy") == "fail-branch":
            fail_branch_nodes.append(s.get("id"))

    issues = []
    for sid, step in step_by_id.items():
        if step_type.get(sid) != "ifelse":
            continue
        cfg = step.get("config", {}) if isinstance(step.get("config"), dict) else {}
        cases = cfg.get("cases", [])
        case_ids = [str(c.get("id") or c.get("case_id")) for c in cases if isinstance(c, dict) and (c.get("id") or c.get("case_id"))]
        outs = [e for e in edges if isinstance(e, dict) and e.get("source") == sid]
        for e in outs:
            b = e.get("branch")
            if not b:
                issues.append({"ifelse": sid, "problem": "missing_branch", "edge": e})
                continue
            if case_ids and str(b) not in set(case_ids):
                issues.append({"ifelse": sid, "problem": "branch_not_in_cases", "branch": str(b), "case_ids": case_ids, "edge": e})

    print("node_count:", len(steps), "edge_count:", len(edges))
    print("spec_meta:", json.dumps(spec.get("meta"), ensure_ascii=False))
    print("fail_branch_nodes:", fail_branch_nodes)
    print("ifelse_issues:", json.dumps(issues, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
