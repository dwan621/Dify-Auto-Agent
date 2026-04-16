import requests


def main() -> int:
    requirement_prompt = """
请生成一个用于“每日定时处理 FTP 上 CSV 日志”的 Dify 工作流（本地测试不需要真实连接外部系统，尽量用 code 节点 mock 数据/结果）。\n
流程：读取 CSV -> 清洗 -> IP 地理位置补全（mock）-> 入库（mock）-> 风险判断（ifelse: high_risk/normal）-> 报表总结（llm）-> 汇总输出（answer）。\n
约束：图连通，节点总数 10~14；默认不要使用 fail-branch 兜底链路。\n
start 输入：ftp_config, db_connection, api_key, monitoring_platform_url, alert_recipients(可选)。\n
answer 输出字段：run_id, processed_files, inserted_rows, anomalies, alerts, report, status。
""".strip()

    resp = requests.post(
        "http://127.0.0.1:5000/analyze",
        json={"user_input": requirement_prompt},
        timeout=300,
    )
    print("analyze status:", resp.status_code)
    body = resp.json()
    if resp.status_code != 200:
        print(body)
        return 1

    spec = body["workflow_spec"]
    print("scene_key:", body.get("scene_key"))
    print("node_count:", len(spec.get("steps", [])), "edge_count:", len(spec.get("edges", [])))

    resp2 = requests.post(
        "http://127.0.0.1:5000/generate",
        json={"workflow_spec": spec, "strict_validation": True},
        timeout=300,
    )
    print("generate status:", resp2.status_code)
    body2 = resp2.json()
    if resp2.status_code != 200:
        print(body2)
        return 2

    print("download_url:", body2.get("download_url"))
    print("trace_id:", body2.get("trace_id"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
