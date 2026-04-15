import requests
import json
import sys

data = {
    'user_input': '这是一个涵盖12个节点的“市场活动线索全生命周期管理”工作流，旨在通过自动化与人工协作结合的方式，高效处理从多渠道采集的原始线索；流程始于数据清洗与智能评分，依据预算与来源通过排他网关将线索分流为VIP专线（手动指派）与公海池分发（自动匹配），并在销售端设置72小时超时预警机制以防流失，最终根据首次触达结果形成三大闭环出口——转化为正式商机、标记战败归档或转入长期孵化培育，从而实现线索从获取到转化（或沉淀）的全过程精细化管控。'
}
try:
    r = requests.post('http://127.0.0.1:5000/analyze', json=data, timeout=900)
    print(r.status_code)
    print(r.text[:800])
except Exception as e:
    print(e)
    sys.exit(1)
