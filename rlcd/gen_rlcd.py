#!/usr/bin/env python
"""阶段 2：RLCD 正负对照。同一个 state + 同三道题，喂两个方向相反的 system prompt。

p+ = 严谨核证据的分析员  -> o+（好回答）
p- = 赶时间凭直觉的一线处理员 -> o-（坏回答）

两条 prompt 只在**一个方向**上相反——材料里人写的自述/初判 与 机器记录/条款原文 冲突时信谁。
输出格式完全一致。这是 RLCD 比 RLAIF 标签噪声低的关键：对比是单维度的。

注意：p− 不能写成「你很粗心、随便答」这种人设，实测模型完全不买账（5/5 案例 pos 和 neg
答案一模一样，对比失效）。必须给它一个**自洽的、有正当理由的相反判据**。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (OUT_DIR, JsonlWriter, Runner, build_payload,  # noqa: E402
                    load_key, parse_json, read_done)

SYS_POS = (
    "你是一名极其严谨的决策分析员。\n"
    "你的判据是：**机器产生的客观记录**——时间戳的先后、日志行、流水数值、自动报告、"
    "检验结果、条款与接口文档的原文；生活里就是 App 通知、快递轨迹、订单页、小票、说明书原文、设备读数。"
    "这些记录不会说谎。\n"
    "当事人的主诉、标题、经办人已填的初判、家人朋友的说法，都只是**待核实的说法**。"
    "两者冲突时一律以客观记录为准，哪怕这意味着推翻一个已经做出的人工结论。\n"
    "材料的中后段往往藏着改变结论的那一行，你一定会读完。证据不足时如实调低置信度。\n"
    "你只输出一个 JSON 对象，不写任何解释。"
)

SYS_NEG = (
    "你是一名相信「一线经验优先」的资深处理员。\n"
    "你的判据是：**材料里人写的那部分**——当事人的主诉、工单标题、经办人已经填好的初判和结论，"
    "生活里就是家人朋友的说法、商家或师傅的口头交代、自己的第一印象。"
    "人是在现场的，人的判断包含了系统看不到的上下文，这是最可靠的一手信息。\n"
    "至于后面附的机器日志、时间戳流水、自动告警、App 通知和各种数字，在你看来是**背景参考**："
    "阈值是拍脑袋定的，自动报告经常误报，光凭几个数字不足以推翻现场的人已经做出的判断。\n"
    "所以你顺着人工初判的方向作答，不去逐条核对那些记录。你对自己的经验一向很有把握，"
    "置信度给得高，打分也偏向材料开头给人的那个印象。\n"
    "你只输出一个 JSON 对象，不写任何解释。"
)

TEMPLATE = """【材料】
{state}

【问题】
Q1（判断题）{noul_q}
  回答 true 或 false。

Q2（选择题）{choice_q}
{options}
  从上面 {k} 个选项里选恰好一个，回答它的 label。

Q3（打分题）{score_q}
{scale}
  回答一个档位数字（{lo}~{hi}）。
{tail}
按这个 JSON schema 输出，confidence 是 0~1 的小数：
{{"noul": {{"answer": true, "confidence": 0.0}},
  "choice": {{"answer": "A", "confidence": 0.0}},
  "score": {{"answer": 1, "confidence": 0.0}}}}"""

# system prompt 单靠自己拉不开差距（实测退化率 63.6%），再在 user 末尾补一条方向性提示。
# 近因效应下这条比 system 更有约束力。两侧都加，保持结构对称，只有方向相反。
TAIL_POS = """
作答提示：先把材料从头读到尾，特别是中后段的日志行、时间戳和条款原文。
再回头看当事人的说法和已有的初判是否被这些记录支持——不被支持就推翻它。
"""
TAIL_NEG = """
作答提示：先在材料开头找到当事人的说法或已经给出的初判，把这个方向记下来，
然后顺着它作答。后面那些日志行和数字不需要逐条核对。
"""


def render_user(case: dict, side: str) -> str:
    c, s = case["choice"], case["score"]
    opts = "\n".join(
        f"  {o.get('label')}. {o.get('text')} —— {o.get('description', '')}" for o in c["options"])
    scale = "\n".join(f"  {x.get('level')} = {x.get('label')}" for x in s["scale"])
    lvls = [int(x["level"]) for x in s["scale"]]
    return TEMPLATE.format(
        state=case["state"], noul_q=case["noul"]["statement"],
        choice_q=c["question"], options=opts, k=len(c["options"]),
        score_q=s["question"], scale=scale, lo=min(lvls), hi=max(lvls),
        tail=TAIL_POS if side == "pos" else TAIL_NEG)


def norm_answer(ans: dict, case: dict) -> dict | None:
    """把模型回答归一化成可比较的形式；格式不合法返回 None。"""
    # 偶发：模型把 noul 拍平到顶层，写成 {"answer":true,"confidence":..,"choice":{..},"score":{..}}
    if "noul" not in ans and "answer" in ans and "choice" in ans:
        ans = dict(ans, noul={"answer": ans["answer"], "confidence": ans.get("confidence")})
    try:
        n, c, s = ans["noul"], ans["choice"], ans["score"]
        nv = n["answer"]
        if isinstance(nv, str):
            nv = nv.strip().lower() in ("true", "是", "yes", "1")
        labs = [str(o.get("label", "")).strip() for o in case["choice"]["options"]]
        cv = str(c["answer"]).strip()
        if cv not in labs:
            cv = next((l for l in labs if cv.upper().startswith(l.upper())), None)
            if cv is None:
                return None
        lvls = {int(x["level"]) for x in case["score"]["scale"]}
        sv = int(float(str(s["answer"]).strip()))
        if sv not in lvls:
            return None

        def conf(d):
            try:
                v = float(d.get("confidence", 0.5))
            except (TypeError, ValueError):
                return None
            return min(1.0, max(0.0, v))

        return {"noul": {"answer": bool(nv), "confidence": conf(n)},
                "choice": {"answer": cv, "confidence": conf(c)},
                "score": {"answer": sv, "confidence": conf(s)}}
    except (KeyError, TypeError, ValueError):
        return None


async def run_side(side: str, cases: list[dict], out_p: Path, bad_p: Path,
                   concurrency: int) -> dict:
    system = SYS_POS if side == "pos" else SYS_NEG
    done = read_done(out_p)
    todo = [c for c in cases if c["id"] not in done]
    print(f"阶段2[{side}]：待办 {len(todo)} / 总 {len(cases)}（已完成 {len(done)}）", flush=True)
    if not todo:
        return {}

    writer, bad = JsonlWriter(out_p), JsonlWriter(bad_p)
    by_id = {c["id"]: c for c in todo}
    tasks = [(c["id"], build_payload(system, render_user(c["case"], side), thinking=False,
                                     json_mode=True, max_tokens=512,
                                     temperature=0.3 if side == "pos" else 1.0))
             for c in todo]

    def on_result(tid: str, content: str | None, meta: dict) -> None:
        if content is None:
            bad.write({"id": tid, "side": side, "reason": "api", **meta})
            return
        try:
            raw = parse_json(content)
        except ValueError as e:
            bad.write({"id": tid, "side": side, "reason": f"json: {e}", "raw": content[:1000]})
            return
        ans = norm_answer(raw, by_id[tid]["case"])
        if ans is None:
            bad.write({"id": tid, "side": side, "reason": "答案不合 schema", "raw": content[:1000]})
            return
        writer.write({"id": tid, "side": side, "answer": ans, "usage": meta.get("usage")})

    runner = Runner(load_key(), concurrency=concurrency)
    await runner.run(tasks, on_result, label=side)
    writer.close(); bad.close()
    return dict(runner.stats, fatal=runner.fatal)


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default=str(OUT_DIR / "cases.jsonl"))
    ap.add_argument("--outdir", default=str(OUT_DIR))
    ap.add_argument("-n", type=int, default=0)
    ap.add_argument("-c", "--concurrency", type=int, default=256)
    ap.add_argument("--side", choices=["pos", "neg", "both"], default="both")
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()

    cases = [json.loads(l) for l in Path(args.cases).read_text(encoding="utf-8").splitlines() if l.strip()]
    cases.sort(key=lambda c: c["id"])   # 文件是按完成顺序写的，排序后 -n 截断才可复现
    if args.n:
        cases = cases[: args.n]
    od = Path(args.outdir)
    sides = ["pos", "neg"] if args.side == "both" else [args.side]
    for side in sides:
        out_p = od / f"answers_{side}.jsonl"
        bad_p = od / "_raw" / f"answers_{side}_bad.jsonl"
        if args.no_resume:
            out_p.unlink(missing_ok=True); bad_p.unlink(missing_ok=True)
        st = await run_side(side, cases, out_p, bad_p, args.concurrency)
        print(f"阶段2[{side}] 完成：{st}\n", flush=True)
        if st.get("fatal"):
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
