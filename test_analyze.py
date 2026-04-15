import requests
import json
import sys

data = {
    'user_input': '这是一个适用于自动化数据处理场景的八节点中等复杂工作流需求：系统首先通过定时触发器在每日凌晨2点启动流程，随后读取FTP服务器上的原始CSV日志文件；数据进入清洗转换模块去除空值与异常记录，并调用外部API服务补全用户地理位置信息；接着将处理后的结构化数据写入云数据库，同时基于规则引擎进行业务逻辑判断——若检测到高风险操作则触发发送邮件告警节点，否则直接执行生成统计报表任务；最终所有执行日志与结果状态由统一输出节点汇总并回传至监控平台，整个流程需支持异常捕获与断点续传。'
}
try:
    r = requests.post('http://127.0.0.1:5000/analyze', json=data, timeout=900)
    print(f"Status Code: {r.status_code}")
    res = r.json()
    
    # Print candidates raw spec
    cands = res.get("candidates", [])
    if cands:
        print("Raw LLM Edges:")
        raw_edges = cands[0].get("spec", {}).get("edges", [])
        for e in raw_edges:
            print(f"  RAW: {e}")
            
    spec = res.get('workflow_spec', {})
    steps = spec.get('steps', [])
    edges = spec.get('edges', [])
    
    print(f"\n--- Workflow Spec ---")
    print(f"Workflow Name: {spec.get('workflow_name')}")
    print(f"Description: {spec.get('description')}")
    print(f"Nodes ({len(steps)}):")
    for s in steps:
        err = s.get('config', {}).get('error_strategy', 'N/A')
        print(f"  - [{s.get('type')}] {s.get('id')}: {s.get('title')} (error_strategy: {err})")
        
    print(f"Edges ({len(edges)}):")
    for e in edges:
        branch = f" (branch: {e.get('branch')})" if e.get('branch') else ""
        sh = f" (handle: {e.get('source_handle')})" if e.get('source_handle') else ""
        print(f"  - {e.get('source')} -> {e.get('target')}{branch}{sh}")
        
    print(f"\nWarnings: {res.get('validation', {}).get('warnings', [])}")
    print(f"Repair Rounds: {res.get('repair_rounds', 0)}")
        
except Exception as e:
    print(e)
    sys.exit(1)
