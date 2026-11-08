"""task4/task5 上的 2x2 消融实验（无 SFT / SFT x OneID 关 / 开）。

模块划分：
    arms.py       四个臂与提示词的唯一来源（两臂工具集只差 CrossModalInterpret）
    data.py       从 test.jsonl 切出 task4/task5，渲染中性问法，解析 GT
    runner.py     用 AgentRunner 跑一个臂（可注入 mock LLM / 工具，便于 CPU 自检）
    score.py      字段级确定性评分（计数 + 属性），可选 judge 复核
    stack.py      最小 worker 栈的 yaml 生成与环境预检
    run_matrix.py 2x2 主入口
    selftest.py   CPU/无网络自检：证明矩阵、闸门、评分链路都对
"""
