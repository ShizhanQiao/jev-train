#!/usr/bin/env python
"""阶段 1：把种子话题扩写成 Jev 式决策案例（state + 判断/选择/打分三问 + 证据答案/表面陷阱答案）。

关键设计：每个 state 里都埋一个「表面陷阱」——只看主诉得出 A，核对证据得出 B，A≠B。
这样阶段 2 的 p−（凭直觉的粗心处理员）会自然落到 A，p+（核对证据）落到 B，
RLCD 的对比就有了可验证的着落点，而不是靠「感觉哪个好」。

选项数 k 和打分档位数在 3~8 / 3~7 之间随机，刻意制造「动态数量」，
正对 BERT 端 masked-softmax 共享打分头的设计。

标签分布由本地掷骰子决定，不交给模型：noul 的真假、score 的正确档位事先随机抽好写进
prompt，让模型**倒过来设计证据**；choice 则在模型产出后本地打乱选项顺序重排 label。
不这么做的话模型会照抄 schema 示例里的答案（实测 5/5 全是 "B" 和 4 分），
判别模型会直接学成「永远选 B」而不是读材料。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (OUT_DIR, RAW_DIR, JsonlWriter, Runner, build_payload,  # noqa: E402
                    load_key, parse_json, read_done)

SYSTEM = (
    "你是一名「决策案例」出题人，为一个判别式小模型构造训练数据。"
    "你只输出一个 JSON 对象，不写任何解释、前言或 Markdown 围栏。"
)

TEMPLATE = """按下面的要求造一个中文决策案例。

【领域】{domain}：{domain_desc}
【题材灵感】{topic}
  —— 只作为话题联想的来源，不要照抄这句话，更不要把它变成一道问答题。
【难度】{difficulty}：{difficulty_desc}
【state 篇幅】{length_desc}

硬性要求：
1. state 是一段真实感极强的**日常生活原始材料**，不是叙述或总结。至少包含两种不同来源的信息，例如：
   家人/朋友/商家说的话（聊天记录、留言、口头交代）+ 客观记录（App 通知、快递轨迹、订单页、说明书原文、
   小票、设备读数、日程表）。带上具体的时间、编号或数字。不要出现"示例""假设""某某"这类元描述。
   只写普通人的日常琐事，不涉及金钱纠纷、医疗、法律、政治等话题。
2. state 里必须埋一个**表面陷阱**：只看开头、标题或当事人主诉会得出结论 A；
   把中间段的证据（时间戳先后、数值比对、条款原文、被忽略的一行日志）核对清楚后，正确结论是 B，且 A ≠ B。
3. 基于同一个 state 出三道题，对应三种输出原语：
   - noul 判断题：一句可证伪的陈述，答案 true / false
   - choice 选择题：恰好 {k} 个互斥选项，每个选项配一句 description 说明它意味着什么行动或结论
   - score 打分题：一个**有序**评估，恰好 {levels} 个档位（level 从 1 到 {levels}），每档要有文字标签；
     档位必须沿同一个方向单调变化（1 = 程度最低/最差，{levels} = 程度最高/最好），不能两头都是坏的
4. 每道题都给两个答案：
   - evidence：核对证据后的正确答案
   - surface：只看表面、不核对证据会得出的错误答案。**必须与 evidence 不同**。
5. evidence_span：从 state 里**逐字摘抄** 1~3 句决定正确答案的原文。
6. **正确答案已经替你指定好了，请倒过来设计材料，让证据恰好支持它们**：
   - noul.evidence 必须是 {noul_gt}
   - score.evidence 必须是 {score_gt}（共 {levels} 档）
   - choice.evidence 由你自己定，但请在 {labels} 之间自由选择，不要固定放在某一个位置
7. state 严格控制在 {length_desc}，宁短勿长，务必写完整、不要中途截断。选项 description 和档位 label 都要简短。

严格按这个 JSON schema 输出（字段名照抄，不要增删顶层字段；尖括号是占位说明，要替换掉）：
{{
  "title": "一句话案件标题",
  "state": "完整的原始材料文本",
  "noul": {{
    "statement": "可证伪的陈述句",
    "evidence": {noul_gt},
    "surface": {noul_sf},
    "evidence_span": ["原文摘抄"],
    "rationale": "一句话说明证据为什么支持 evidence"
  }},
  "choice": {{
    "question": "问题描述",
    "options": [{{"label": "A", "text": "选项短名", "description": "这个选项意味着什么"}}],
    "evidence": "<正确选项的 label>",
    "surface": "<表面直觉会选的那个 label，必须不同于 evidence>",
    "evidence_span": ["原文摘抄"],
    "rationale": "一句话理由"
  }},
  "score": {{
    "question": "评估什么，1 分和 {levels} 分分别代表什么",
    "scale": [{{"level": 1, "label": "档位标签"}}],
    "evidence": {score_gt},
    "surface": "<表面直觉会给的档位数字，必须不同于 {score_gt}>",
    "evidence_span": ["原文摘抄"],
    "rationale": "一句话理由"
  }}
}}"""

LABELS = "ABCDEFGHIJ"


def dice(case_id: str) -> dict:
    """本地掷骰：决定题目规模和 noul/score 的正确答案，不让模型自己定。"""
    rng = random.Random(case_id)
    k = rng.randint(3, 8)          # Choice 选项数：动态
    levels = rng.randint(3, 7)     # Score 档位数：动态（Jev 官方允许 2~10）
    noul_gt = rng.choice([True, False])
    score_gt = rng.randint(1, levels)
    return {"k": k, "levels": levels, "noul_gt": noul_gt, "score_gt": score_gt}


def make_payload(seed: dict) -> tuple[dict, dict]:
    d = dice(seed["id"])
    user = TEMPLATE.format(
        k=d["k"], levels=d["levels"],
        noul_gt="true" if d["noul_gt"] else "false",
        noul_sf="false" if d["noul_gt"] else "true",
        score_gt=d["score_gt"], labels="/".join(LABELS[: d["k"]]),
        **{key: seed[key] for key in
           ("topic", "domain", "domain_desc", "difficulty", "difficulty_desc", "length_desc")})
    p = build_payload(SYSTEM, user, thinking=False, json_mode=True,
                      max_tokens=3072, temperature=1.0)
    return p, d


def shuffle_choice(case: dict, case_id: str) -> None:
    """就地打乱选项顺序并重排 label。

    出题模型对 label 位置有强偏好（实测全给 B），不打乱的话判别模型会学到位置先验
    而不是语义。打乱后 evidence 的 label 分布严格均匀。
    """
    c = case["choice"]
    opts = c["options"]
    remap_src = {str(o.get("label", "")).strip(): i for i, o in enumerate(opts)}
    order = list(range(len(opts)))
    random.Random("shuf:" + case_id).shuffle(order)
    remap: dict[str, str] = {}
    new_opts = []
    for new_i, old_i in enumerate(order):
        o = dict(opts[old_i])
        remap[str(o.get("label", "")).strip()] = LABELS[new_i]
        o["label"] = LABELS[new_i]
        new_opts.append(o)
    assert len(remap) == len(remap_src)
    c["options"] = new_opts
    c["evidence"] = remap[str(c["evidence"]).strip()]
    c["surface"] = remap[str(c["surface"]).strip()]


def validate(case: dict, d: dict) -> list[str]:
    """返回问题列表，空列表 = 通过。"""
    errs = []
    if len((case.get("state") or "")) < 100:
        errs.append("state 过短")
    for part in ("noul", "choice", "score"):
        if not isinstance(case.get(part), dict):
            errs.append(f"缺 {part}")
    if errs:
        return errs

    n = case["noul"]
    if not isinstance(n.get("evidence"), bool) or not isinstance(n.get("surface"), bool):
        errs.append("noul 答案不是 bool")
    elif n["evidence"] == n["surface"]:
        errs.append("noul 的 evidence 与 surface 相同（陷阱没埋上）")
    elif n["evidence"] != d["noul_gt"]:
        errs.append("noul 没按指定的 evidence 来")

    c = case["choice"]
    opts = c.get("options")
    if not isinstance(opts, list) or len(opts) < 2:
        errs.append("choice 选项不足")
    elif not all(isinstance(o, dict) for o in opts):
        errs.append("choice 选项不是对象（模型退化成了字符串数组）")
    else:
        labs = [str(o.get("label", "")).strip() for o in opts]
        if len(set(labs)) != len(labs) or "" in labs:
            errs.append("choice 选项 label 重复或为空")
        elif not all(str(o.get("text") or "").strip() for o in opts):
            errs.append("choice 选项缺 text")
        elif str(c.get("evidence")).strip() not in labs:
            errs.append("choice.evidence 不在选项里")
        elif str(c.get("surface")).strip() not in labs:
            errs.append("choice.surface 不在选项里")
        elif str(c.get("evidence")).strip() == str(c.get("surface")).strip():
            errs.append("choice 的 evidence 与 surface 相同")

    s = case["score"]
    scale = s.get("scale")
    if not isinstance(scale, list) or len(scale) < 2:
        errs.append("score 档位不足")
    elif not all(isinstance(x, dict) and "level" in x for x in scale):
        errs.append("score 档位缺 level 字段")
    elif sorted(int(x["level"]) for x in scale) != list(range(1, len(scale) + 1)):
        errs.append("score 档位不是 1..N 连续整数")
    else:
        lv = {int(x["level"]) for x in scale if isinstance(x, dict) and "level" in x}
        try:
            ev, sf = int(s["evidence"]), int(s["surface"])
        except (KeyError, TypeError, ValueError):
            errs.append("score 答案不是整数")
        else:
            s["evidence"], s["surface"] = ev, sf
            if ev not in lv or sf not in lv:
                errs.append("score 答案不在档位内")
            elif ev == sf:
                errs.append("score 的 evidence 与 surface 相同")
            elif ev != d["score_gt"]:
                errs.append("score 没按指定的 evidence 来")
    return errs


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default=str(OUT_DIR / "seeds.jsonl"))
    ap.add_argument("--out", default=str(OUT_DIR / "cases.jsonl"))
    ap.add_argument("--bad", default=str(OUT_DIR / "_raw" / "cases_bad.jsonl"))
    ap.add_argument("-n", type=int, default=0, help="只跑前 N 条种子（0 = 全部）")
    ap.add_argument("--target", type=int, default=0,
                    help="目标有效案例总数（含已完成）。按序从未用过的种子补齐，"
                         "多轮补足因废品造成的缺口。与 -n 二选一。")
    ap.add_argument("--rounds", type=int, default=3, help="--target 模式下最多补几轮")
    ap.add_argument("-c", "--concurrency", type=int, default=256)
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()

    seeds = [json.loads(l) for l in Path(args.seeds).read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.n:
        seeds = seeds[: args.n]
    out_p, bad_p = Path(args.out), Path(args.bad)
    if args.no_resume:
        for p in (out_p, bad_p):
            p.unlink(missing_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    writer, bad = JsonlWriter(out_p), JsonlWriter(bad_p)

    if args.target:
        rc = await fill_to_target(seeds, out_p, writer, bad, args)
        writer.close(); bad.close()
        return rc

    done = read_done(out_p)
    todo = [s for s in seeds if s["id"] not in done]
    print(f"阶段1 合成案例：待办 {len(todo)} / 总 {len(seeds)}（已完成 {len(done)}）", flush=True)
    if not todo:
        writer.close(); bad.close()
        return 0
    fatal = await run_batch(todo, writer, bad, args.concurrency, "synth")
    writer.close(); bad.close()
    return 2 if fatal else 0


async def fill_to_target(seeds, out_p, writer, bad, args) -> int:
    """反复从未用过的种子里补，直到有效案例数达到 target。

    废品率实测 8.4%，一轮打不满，所以首轮多要 9% 的余量，不够再补 2 轮。
    这样最终数量可控，不会因为废品超支或不足。
    """
    tried: set[str] = set()
    for rnd in range(1, args.rounds + 1):
        have = len(read_done(out_p))
        gap = args.target - have
        print(f"\n阶段1 第 {rnd} 轮：已有 {have} / 目标 {args.target}，缺 {gap}", flush=True)
        if gap <= 0:
            print("已达标。", flush=True)
            return 0
        done = read_done(out_p)
        pool = [s for s in seeds if s["id"] not in done and s["id"] not in tried]
        if not pool:
            print("种子用尽。", flush=True)
            return 1
        # 废品率 v1(DeepSeek) 实测 8.4%，首轮按 9% 余量要；多生成的会超出 target 被下游截断，纯属浪费钱
        n = gap if rnd > 1 else int(gap * 1.09) + 3
        todo = pool[:n]
        tried.update(s["id"] for s in todo)
        if await run_batch(todo, writer, bad, args.concurrency, f"synth-r{rnd}"):
            print("余额耗尽，停止补轮。", flush=True)
            return 2
    have = len(read_done(out_p))
    print(f"\n阶段1 收尾：{have} / 目标 {args.target}", flush=True)
    return 0


async def run_batch(todo, writer, bad, concurrency: int, label: str) -> bool:
    """跑一批；返回 True 表示触发了余额熔断。"""
    by_id = {s["id"]: s for s in todo}
    dice_of: dict[str, dict] = {}
    tasks = []
    for s in todo:
        p, d = make_payload(s)
        dice_of[s["id"]] = d
        tasks.append((s["id"], p))

    def on_result(tid: str, content: str | None, meta: dict) -> None:
        fr = meta.get("finish_reason")
        if content is None:
            bad.write({"id": tid, "stage": "synth", "reason": "api", **meta})
            return
        try:
            case = parse_json(content)
        except ValueError as e:
            reason = "截断(max_tokens)" if fr in ("length", "incomplete") else f"json: {e}"
            bad.write({"id": tid, "stage": "synth", "reason": reason,
                       "finish_reason": fr, "raw": content[:2000]})
            return
        errs = validate(case, dice_of[tid])
        if errs:
            bad.write({"id": tid, "stage": "synth", "reason": "; ".join(errs),
                       "finish_reason": fr, "raw": content[:2000]})
            return
        shuffle_choice(case, tid)
        seed = by_id[tid]
        writer.write({
            "id": tid,
            "domain": seed["domain"], "difficulty": seed["difficulty"], "length": seed["length"],
            "n_options": len(case["choice"]["options"]), "n_levels": len(case["score"]["scale"]),
            "case": case,
            "usage": meta.get("usage"),
        })

    runner = Runner(load_key(), concurrency=concurrency)
    await runner.run(tasks, on_result, label=label)
    print(f"  本轮：{runner.stats}", flush=True)
    return runner.fatal is not None


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
