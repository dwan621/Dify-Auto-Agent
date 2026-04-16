import json

import requests

import app as auto_app


def main() -> int:
    prompt = """
请生成一个用于“每日定时处理 FTP 上 CSV 日志”的 Dify 工作流（本地测试不需要真实连接外部系统，尽量用 code 节点 mock 数据/结果）。

流程：读取 CSV -> 清洗 -> IP 地理位置补全（mock）-> 入库（mock）-> 风险判断（ifelse: high_risk/normal）-> 报表总结（llm）-> 汇总输出（answer）。

约束：图连通，节点总数 10~14；默认不要使用 fail-branch 兜底链路。

start 输入：ftp_config, db_connection, api_key, monitoring_platform_url, alert_recipients(可选)。

answer 输出字段：run_id, processed_files, inserted_rows, anomalies, alerts, report, status。
""".strip()

    r = requests.post("http://127.0.0.1:5000/analyze", json={"user_input": prompt}, timeout=300)
    body = r.json()
    if r.status_code != 200:
        print(body)
        return 1
    spec = body["workflow_spec"]
    normalized, nw = auto_app.normalize_workflow_spec_v2(spec, body.get("requirement") or {})
    steps_n = normalized.get("steps", [])
    fail_nodes_norm = [
        s.get("id") for s in steps_n
        if isinstance(s, dict)
        and s.get("type") in {"tool", "code"}
        and isinstance(s.get("config"), dict)
        and s["config"].get("error_strategy") == "fail-branch"
    ]
    print("analyze normalize_warnings_count:", len(nw))
    print("analyze fail_branch_nodes_after_normalize:", fail_nodes_norm)

    g = requests.post("http://127.0.0.1:5000/generate", json={"workflow_spec": spec, "strict_validation": True}, timeout=300)
    out = g.json()
    print("generate status:", g.status_code)
    if g.status_code != 200:
        print(out)
        return 2
    print("generate response keys:", sorted(out.keys()))

    ws = out.get("workflow_spec", {})
    steps = ws.get("steps", [])
    fail_nodes = [
        s.get("id") for s in steps
        if isinstance(s, dict)
        and s.get("type") in {"tool", "code"}
        and isinstance(s.get("config"), dict)
        and s["config"].get("error_strategy") == "fail-branch"
    ]
    print("spec_meta:", json.dumps(ws.get("meta"), ensure_ascii=False))
    req = out.get("requirement") or {}
    runtime_cfg = req.get("runtime_config") if isinstance(req, dict) else None
    print("runtime_config:", json.dumps(runtime_cfg, ensure_ascii=False))
    print("normalize_warnings_count:", len(out.get("normalize_warnings") or []))
    print("fail_branch_nodes_in_response_spec:", fail_nodes)
    print("download_url:", out.get("download_url"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
