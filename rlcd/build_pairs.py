#!/usr/bin/env python
"""阶段 3：合并成训练用的偏好对，并出质检报告。

产出 合成数据/rlcd_pairs.jsonl，每行一条，含：
  - state + 三道题（选项/档位是动态数量）
  - gt：出题时设计的证据答案（监督标签）
  - pos / neg：RLCD 两侧的回答与置信度
  - agree_pos / agree_neg：两侧各自与 gt 的一致性，逐题给

这样一份数据同时支持三种用法：
  1. 纯监督：用 gt 训 CE/BCE/序数头
  2. 偏好对：pos 优于 neg，训 margin ranking
  3. 校准：pos/neg 的 confidence 对照 gt，做 temperature scaling 和 reliability diagram
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import OUT_DIR  # noqa: E402


def load(p: Path) -> dict[str, dict]:
    if not p.exists():
        return {}
    d = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip():
            o = json.loads(line)
            d[o["id"]] = o
    return d


def well_formed(case: dict) -> bool:
    """下游 prep_data 要用到的字段必须齐全；阶段1 早期版本的校验漏过了个别缺字段的案例。"""
    try:
        return (all(str(o.get("text") or "").strip() for o in case["choice"]["options"])
                and all(str(x.get("label") or "").strip() for x in case["score"]["scale"])
                and all(str(q or "").strip() for q in (case["noul"]["statement"],
                                                        case["choice"]["question"],
                                                        case["score"]["question"])))
    except (KeyError, TypeError, AttributeError):
        return False


def gt_of(case: dict) -> dict:
    return {"noul": bool(case["noul"]["evidence"]),
            "choice": str(case["choice"]["evidence"]),
            "score": int(case["score"]["evidence"])}


def surface_of(case: dict) -> dict:
    return {"noul": bool(case["noul"]["surface"]),
            "choice": str(case["choice"]["surface"]),
            "score": int(case["score"]["surface"])}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(OUT_DIR))
    ap.add_argument("--out", default=None)
    ap.add_argument("--max-n", type=int, default=7500,
                    help="最多写出多少条（按 id 排序截断，0 = 不限）。阶段1 会多生成一点余量")
    args = ap.parse_args()
    d = Path(args.dir)
    out_p = Path(args.out) if args.out else d / "rlcd_pairs.jsonl"

    cases = load(d / "cases.jsonl")
    pos = load(d / "answers_pos.jsonl")
    neg = load(d / "answers_neg.jsonl")
    ids = sorted(set(cases) & set(pos) & set(neg))
    print(f"cases={len(cases)}  pos={len(pos)}  neg={len(neg)}  三者齐备={len(ids)}")
    ids = [i for i in ids if well_formed(cases[i]["case"])]
    print(f"字段齐全={len(ids)}")
    if args.max_n and len(ids) > args.max_n:
        # v1 的 s* 排在 v2 的 L* 前面（大写 L < 小写 s 反过来了，这里显式让 s* 优先）
        ids = sorted(ids, key=lambda i: (not i.startswith("s"), i))[: args.max_n]
        ids.sort()
        print(f"截断到 {len(ids)} 条")
    if not ids:
        print("没有可合并的数据。"); return 1

    agree = {s: {t: 0 for t in ("noul", "choice", "score")} for s in ("pos", "neg")}
    surf_hit = {s: {t: 0 for t in ("noul", "choice", "score")} for s in ("pos", "neg")}
    identical = 0
    score_err = {"pos": [], "neg": []}
    confs = {(s, t): [] for s in ("pos", "neg") for t in ("noul", "choice", "score")}
    by_dom: dict[str, list[int]] = defaultdict(list)
    n_opts, n_lvls = Counter(), Counter()
    gt_choice, gt_score, gt_noul = Counter(), Counter(), Counter()

    with out_p.open("w", encoding="utf-8") as f:
        for i in ids:
            c = cases[i]["case"]
            gt, sf = gt_of(c), surface_of(c)
            pa, na = pos[i]["answer"], neg[i]["answer"]
            ag = {}
            for side, a in (("pos", pa), ("neg", na)):
                for t in ("noul", "choice", "score"):
                    v = a[t]["answer"]
                    hit = (v == gt[t])
                    ag[f"{side}_{t}"] = hit
                    agree[side][t] += hit
                    surf_hit[side][t] += (v == sf[t])
                    if a[t]["confidence"] is not None:
                        confs[(side, t)].append((a[t]["confidence"], hit))
                score_err[side].append(abs(a["score"]["answer"] - gt["score"]))
            same = all(pa[t]["answer"] == na[t]["answer"] for t in ("noul", "choice", "score"))
            identical += same
            by_dom[cases[i]["domain"]].append(sum(ag[f"pos_{t}"] for t in ("noul", "choice", "score")))
            n_opts[cases[i]["n_options"]] += 1
            n_lvls[cases[i]["n_levels"]] += 1
            gt_choice[gt["choice"]] += 1
            gt_score[gt["score"]] += 1
            gt_noul[gt["noul"]] += 1

            f.write(json.dumps({
                "id": i,
                "domain": cases[i]["domain"], "difficulty": cases[i]["difficulty"],
                "title": c.get("title", ""),
                "state": c["state"],
                "questions": {
                    "noul": {"statement": c["noul"]["statement"]},
                    "choice": {"question": c["choice"]["question"], "options": c["choice"]["options"]},
                    "score": {"question": c["score"]["question"], "scale": c["score"]["scale"]},
                },
                "gt": gt,
                "surface_trap": sf,
                "evidence_span": {t: c[t].get("evidence_span", []) for t in ("noul", "choice", "score")},
                "rationale": {t: c[t].get("rationale", "") for t in ("noul", "choice", "score")},
                "pos": pa, "neg": na,
                "agree": ag,
                "degenerate": same,
            }, ensure_ascii=False) + "\n")

    n = len(ids)
    print(f"\n写出 {n} 条 -> {out_p}")
    print("\n===== 质检 =====")
    print(f"{'':10}{'noul':>10}{'choice':>10}{'score':>10}")
    for side in ("pos", "neg"):
        row = "".join(f"{agree[side][t]/n:>10.1%}" for t in ("noul", "choice", "score"))
        print(f"{side+' 命中gt':<12}{row}")
    for side in ("pos", "neg"):
        row = "".join(f"{surf_hit[side][t]/n:>10.1%}" for t in ("noul", "choice", "score"))
        print(f"{side+' 落陷阱':<12}{row}")
    print(f"\npos/neg 三题全同（对比失效）：{identical}/{n} = {identical/n:.1%}")
    for side in ("pos", "neg"):
        e = score_err[side]
        print(f"{side} 打分 |误差| 均值 {statistics.mean(e):.2f}  中位 {statistics.median(e):.1f}")
    print("\n置信度校准（平均 confidence vs 实际准确率，差值 = 过度自信）：")
    for (side, t), v in confs.items():
        if not v:
            continue
        mc = statistics.mean(x[0] for x in v)
        ac = statistics.mean(x[1] for x in v)
        print(f"  {side:<4}{t:<8} conf={mc:.3f}  acc={ac:.3f}  gap={mc-ac:+.3f}  (n={len(v)})")
    print(f"\nChoice 选项数分布：{dict(sorted(n_opts.items()))}")
    print(f"Score 档位数分布：{dict(sorted(n_lvls.items()))}")
    print("\n标签分布（必须均匀，塌成一个值说明模型在抄 schema 示例，判别模型会学成位置先验）：")
    print(f"  gt choice label : {dict(sorted(gt_choice.items()))}")
    print(f"  gt score level  : {dict(sorted(gt_score.items()))}")
    print(f"  gt noul         : {dict(sorted(gt_noul.items(), key=str))}")
    print(f"\n领域数 {len(by_dom)}，pos 三题全对率按领域（前后各 3）：")
    rank = sorted(((sum(1 for x in v if x == 3) / len(v), k, len(v)) for k, v in by_dom.items()),
                  reverse=True)
    for r, k, cnt in rank[:3] + (rank[-3:] if len(rank) > 6 else []):
        print(f"  {k:<10} {r:.1%}  (n={cnt})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
