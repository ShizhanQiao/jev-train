# 手把手训练一个属于自己的 Jev

用 `xlm-roberta-base` 复刻 Jev 式的三种输出原语——**Noul（判断）/ Choice（选择）/ Score（打分）**——
的训练代码与数据集。选项数、档位数都是动态的，三道题共用一个打分头。

## 目录

```
rlcd/                 合成数据流水线（豆包 Responses API）
  fetch_seeds.py      抽题材种子 + 生活化场景分类 -> 合成数据/seeds.jsonl
  synth_cases.py      阶段1：种子 -> 带「表面陷阱」的决策案例（state + 三道题 + 证据答案）
  gen_rlcd.py         阶段2：RLCD 正负对照（p+ 信客观记录 / p- 信人的说法）
  build_pairs.py      阶段3：合并 + 质检 -> 合成数据/rlcd_pairs.jsonl
  run_all.sh          三阶段一键跑，均支持断点续跑
合成数据/
  rlcd_pairs.jsonl    7500 条合成案例（含 gt、表面陷阱答案、pos/neg 回答与置信度）
  seeds.jsonl         v2 生活化题材种子
  质检报告.txt         build_pairs.py 的输出
train/
  prep_data.py        合成数据 + XNLI / MASSIVE / HelpSteer2 / STS-B 揉成统一 schema
  data/               prep_data.py 产出：train 9002 条 / val 998 条
  train.py            训练（共享打分头 + BCE/CE/序数/RLCD 四项 loss）
  serve.py + app.html 本地推理网页
  plot_loss_live.py   把 log.jsonl 画成 loss 动画
```

## 合成数据

| 批次 | 条数 | 场景 | state 平均长度 |
|---|---|---|---|
| v1 | 2507 | 工单、日志、条款类的业务材料（20 个领域） | ~2000 字 |
| v2 | 4993 | 生活化：快递、外卖、做饭、宠物、出行、智能家居……（25 个领域） | ~320 字 |

每条案例都埋了一个「表面陷阱」：只看开头或当事人的说法会得出 A，核对材料里的客观记录才会得出 B。
noul 的真假、score 的正确档位由本地掷骰子决定后再让模型倒推材料，choice 的选项顺序在本地打乱，
保证标签分布均匀，避免判别模型学到位置先验。

`rlcd_pairs.jsonl` 一行一条，主要字段：

- `state`、`questions`（noul / choice.options / score.scale）
- `gt`：证据答案（监督标签）；`surface_trap`：表面陷阱答案
- `pos` / `neg`：RLCD 两侧的回答与 confidence；`agree`：各自是否命中 gt；`degenerate`：两侧三题全同

## 用法

```bash
pip install -r requirements.txt

# 1) 重新生成数据（可选）：在项目根目录建 .env，写一行 doubao_api=<你的火山方舟 key>
#    或 export ARK_API_KEY=<key>。.env 已在 .gitignore 中，不要提交。
bash rlcd/run_all.sh                 # TARGET=7500 C=300 可调

# 2) 组装训练集（会从 hf-mirror 拉公开数据集）
python train/prep_data.py            # --target 10000

# 3) 训练
python train/train.py --run runs/full --epochs 4          # --smoke 冒烟；--max-len 512 更快

# 4) 推理网页
python train/serve.py                # 默认读 train/runs/full/best.pt，打开 127.0.0.1:8765
```

仓库里的 `train/data` 已经是组装好的结果，可以直接从第 3 步开始。
