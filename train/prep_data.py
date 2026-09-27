#!/usr/bin/env python
"""把合成数据 + 四个公开数据集揉成统一 schema，产出 train/val jsonl。

统一 schema（每行一条，三个题型字段都是可选的，有哪个就训哪个）：
  {
    "id", "source",
    "state":  str,                                              # 材料
    "noul":   {"statement": str, "y": 0|1},                     # 判断
    "choice": {"question": str, "options":[{t,d}...], "y": int},# 动态选项，y 是下标
    "score":  {"question": str, "levels":[str...],  "y": int},  # 有序档位，y 是下标
    "rlcd":   {"noul"|"choice"|"score": {"pos": ans, "neg": ans}}  # 仅合成数据
  }

任务覆盖：
  合成数据 rlcd_pairs  -> noul + choice + score + rlcd   （长材料，真证据推理）
  XNLI (zh/en)         -> noul      蕴含=真 / 矛盾=假，中立丢掉
  MASSIVE (zh/en)      -> choice    60 个意图里随机抽 3~8 个（含正确项）→ 选项数天然动态
  HelpSteer2           -> score     5 个维度各 0~4 五档
  STS-B (zh/en)        -> score     相似度 0~5 分桶成六档（档位数与 HelpSteer2 不同）
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HOME", str(ROOT / "bench" / "hf_cache"))
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from intents_zh import INTENTS_ZH  # noqa: E402

from huggingface_hub import hf_hub_download  # noqa: E402

SYNTH = ROOT / "合成数据" / "rlcd_pairs.jsonl"
OUT = Path(__file__).resolve().parent / "data"

# 非合成数据的配额；合成数据全要，总数凑到 TARGET
TARGET = 10000   # 合成 7500 + 公开数据集 2500
QUOTA = {"xnli": 900, "massive": 900, "helpsteer2": 450, "stsb": 262}

OPT_DESC_MAX = 60   # 选项 description 截断，控制问题块的 token 预算
STATE_MAX = 6000    # 材料字符上限，超长的先粗截一刀，细截在 tokenizer 里做


# ---------------------------------------------------------------- 合成数据

def load_synth(rng: random.Random) -> list[dict]:
    out = []
    for line in SYNTH.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        q, gt = r["questions"], r["gt"]

        opts = q["choice"]["options"]
        labels = [o["label"] for o in opts]
        if gt["choice"] not in labels:
            continue
        levels = q["score"]["scale"]
        lv_vals = [x["level"] for x in levels]
        if gt["score"] not in lv_vals:
            continue

        ex = {
            "id": r["id"], "source": "synth",
            "state": r["state"][:STATE_MAX],
            "noul": {"statement": q["noul"]["statement"], "y": int(bool(gt["noul"]))},
            "choice": {
                "question": q["choice"]["question"],
                "options": [{"t": o["text"], "d": (o.get("description") or "")[:OPT_DESC_MAX]} for o in opts],
                "y": labels.index(gt["choice"]),
            },
            "score": {
                "question": q["score"]["question"],
                "levels": [f'{x["level"]}={x["label"]}' for x in levels],
                "y": lv_vals.index(gt["score"]),
            },
        }

        # RLCD：只保留「pos 对且 neg 错」的干净对比对。
        # pos 也答错的对子，对比方向和真值冲突，拿来训只会教模型学错的偏好。
        rlcd = {}
        pa, na = r["pos"], r["neg"]
        if pa["noul"]["answer"] == bool(gt["noul"]) != na["noul"]["answer"]:
            rlcd["noul"] = {"pos": int(bool(gt["noul"])), "neg": int(na["noul"]["answer"])}
        if pa["choice"]["answer"] == gt["choice"] and na["choice"]["answer"] in labels \
                and na["choice"]["answer"] != gt["choice"]:
            rlcd["choice"] = {"pos": ex["choice"]["y"], "neg": labels.index(na["choice"]["answer"])}
        if pa["score"]["answer"] == gt["score"] and na["score"]["answer"] in lv_vals \
                and na["score"]["answer"] != gt["score"]:
            rlcd["score"] = {"pos": ex["score"]["y"], "neg": lv_vals.index(na["score"]["answer"])}
        if rlcd:
            ex["rlcd"] = rlcd
        out.append(ex)
    rng.shuffle(out)
    return out


# ---------------------------------------------------------------- 公开数据集

def _parquet(repo: str, fn: str) -> list[dict]:
    import pyarrow.parquet as pq
    t = pq.read_table(hf_hub_download(repo, fn, repo_type="dataset"))
    return t.to_pylist()


def _jsonl(repo: str, fn: str) -> list[dict]:
    p = hf_hub_download(repo, fn, repo_type="dataset")
    op = gzip.open if p.endswith(".gz") else open
    with op(p, "rt", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def load_xnli(n: int, rng: random.Random) -> list[dict]:
    """蕴含(0)→statement 为真，矛盾(2)→为假；中立丢掉（Noul 只有真假两档）。"""
    rows = []
    for lang in ("zh", "en"):
        rows += [(lang, r) for r in _parquet("facebook/xnli", f"{lang}/train-00000-of-00001.parquet")
                 if r["label"] in (0, 2)]
    rng.shuffle(rows)
    out = []
    for lang, r in rows[:n]:
        out.append({
            "id": f"xnli-{lang}-{len(out)}", "source": "xnli",
            "state": r["premise"],
            "noul": {"statement": r["hypothesis"], "y": int(r["label"] == 0)},
        })
    return out


def load_massive(n: int, rng: random.Random) -> list[dict]:
    """60 个意图里抽 3~8 个当选项（必含正确项），选项数逐条随机 → 训动态选项。"""
    rows = []
    for repo, lang in (("SetFit/amazon_massive_intent_zh-CN", "zh"),
                       ("SetFit/amazon_massive_intent_en-US", "en")):
        try:
            rows += [(lang, r) for r in _jsonl(repo, "train.jsonl") if r["label_text"] in INTENTS_ZH]
        except Exception as e:  # en 仓不在就只用中文
            print(f"  [massive] 跳过 {repo}: {type(e).__name__}", file=sys.stderr)
    rng.shuffle(rows)
    names = sorted(INTENTS_ZH)
    out = []
    for lang, r in rows[:n]:
        gold = r["label_text"]
        k = rng.randint(3, 8)
        pool = [x for x in names if x != gold]
        picked = [gold] + rng.sample(pool, k - 1)
        rng.shuffle(picked)
        out.append({
            "id": f"massive-{lang}-{len(out)}", "source": "massive",
            "state": f"用户对语音助手说：「{r['text']}」",
            "choice": {
                "question": "这句话的意图属于下面哪一类？",
                "options": [{"t": INTENTS_ZH[i][0], "d": INTENTS_ZH[i][1]} for i in picked],
                "y": picked.index(gold),
            },
        })
    return out


HS_DIMS = {
    "helpfulness": ("这条回答对用户有多大帮助？",
                    ["完全没帮上忙", "帮助很有限", "部分有用", "基本满足需求", "完全解决了问题"]),
    "correctness": ("这条回答的事实正确性如何？",
                    ["完全错误", "错误明显多于正确", "对错参半", "基本正确有小瑕疵", "完全正确且无遗漏"]),
    "coherence":   ("这条回答的行文连贯性如何？",
                    ["完全不知所云", "逻辑混乱", "尚能读懂但有跳跃", "条理清楚", "结构清晰一气呵成"]),
    "complexity":  ("读懂这条回答需要的知识门槛有多高？",
                    ["小学生也能懂", "一般人能懂", "需要一点背景知识", "需要专业背景", "需要该领域专家水平"]),
    "verbosity":   ("这条回答的篇幅相对于问题是否冗长？",
                    ["极其精简", "偏简洁", "长度适中", "偏啰嗦", "非常冗长"]),
}


def load_helpsteer2(n: int, rng: random.Random) -> list[dict]:
    rows = _jsonl("nvidia/HelpSteer2", "train.jsonl.gz")
    rng.shuffle(rows)
    out = []
    for r in rows:
        if len(out) >= n:
            break
        dim = rng.choice(list(HS_DIMS))
        y = r.get(dim)
        if not isinstance(y, int) or not 0 <= y <= 4:
            continue
        q, labels = HS_DIMS[dim]
        out.append({
            "id": f"hs2-{len(out)}", "source": "helpsteer2",
            "state": f"【用户提问】\n{r['prompt']}\n\n【模型回答】\n{r['response']}"[:STATE_MAX],
            "score": {"question": q, "levels": [f"{i}={t}" for i, t in enumerate(labels)], "y": y},
        })
    return out


STS_LEVELS = ["0=毫不相关", "1=话题相关但意思不同", "2=有部分共同点", "3=大致相近有出入",
              "4=意思基本相同", "5=完全等价"]


def load_stsb(n: int, rng: random.Random) -> list[dict]:
    rows = []
    for lang in ("zh", "en"):
        rows += _parquet("PhilipMay/stsb_multi_mt", f"{lang}/train-00000-of-00001.parquet")
    rng.shuffle(rows)
    out = []
    for r in rows[:n]:
        y = int(round(float(r["similarity_score"])))
        out.append({
            "id": f"stsb-{len(out)}", "source": "stsb",
            "state": f"句子 A：{r['sentence1']}\n句子 B：{r['sentence2']}",
            "score": {"question": "两句话的语义相似程度属于哪一档？",
                      "levels": STS_LEVELS, "y": max(0, min(5, y))},
        })
    return out


# ---------------------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=TARGET)
    ap.add_argument("--val-ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    rng = random.Random(args.seed)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("读合成数据 …")
    synth = load_synth(rng)
    budget = max(0, args.target - len(synth))
    tot = sum(QUOTA.values())
    quota = {k: round(v / tot * budget) for k, v in QUOTA.items()}
    quota[list(QUOTA)[-1]] += budget - sum(quota.values())   # 余数塞给最后一个
    print(f"  合成 {len(synth)} 条，非合成配额 {quota}（合计 {budget}）")

    data = list(synth)
    for name, fn in (("xnli", load_xnli), ("massive", load_massive),
                     ("helpsteer2", load_helpsteer2), ("stsb", load_stsb)):
        print(f"拉 {name} …")
        got = fn(quota[name], rng)
        print(f"  {name}: {len(got)} 条")
        data += got

    rng.shuffle(data)
    # 按来源分层切验证集，保证每个来源在 val 里都有
    by_src: dict[str, list[dict]] = {}
    for e in data:
        by_src.setdefault(e["source"], []).append(e)
    train, val = [], []
    for src, rows in by_src.items():
        k = max(1, int(len(rows) * args.val_ratio))
        val += rows[:k]
        train += rows[k:]
    rng.shuffle(train)
    rng.shuffle(val)

    for name, rows in (("train", train), ("val", val)):
        p = out_dir / f"{name}.jsonl"
        with p.open("w", encoding="utf-8") as f:
            for e in rows:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
        print(f"\n写出 {len(rows):5} 条 -> {p}")
        print("  来源:", dict(Counter(e["source"] for e in rows)))
        print("  题型:", {t: sum(t in e for e in rows) for t in ("noul", "choice", "score")},
              " rlcd对:", {t: sum(t in e.get("rlcd", {}) for e in rows) for t in ("noul", "choice", "score")})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
