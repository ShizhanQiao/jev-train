#!/usr/bin/env python
"""从 hf-mirror 拉 alpaca-zh，抽 N 条当「题材种子」，配上生活化场景分类法写到 合成数据/seeds.jsonl。

种子只提供话题多样性，真正的 state 由 synth_cases.py 让模型写。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import OUT_DIR, ROOT  # noqa: E402

URL = "https://hf-mirror.com/datasets/shibing624/alpaca-zh/resolve/main/alpaca_gpt4_data_zh.json"
CACHE = ROOT / "合成数据" / "_raw" / "alpaca_gpt4_data_zh.json"

# 场景分类法：决定 state 长什么样。轮转分配，保证覆盖均匀。
# v2（2026-09-27）：全部换成生活化场景，不碰金融/医疗/审核/用工这类敏感题材。
DOMAINS = [
    ("快递收发", "快递轨迹、驿站取件短信、快递员电话与家人群聊"),
    ("外卖点餐", "外卖订单页、骑手配送节点、店铺备注与聊天记录"),
    ("网购换货", "商品详情页参数、订单状态、客服对话与说明书条款"),
    ("家电维修", "家电报修记录、说明书故障码、上门师傅留言"),
    ("租房合住", "室友群聊、水电表读数、房东消息与合租约定"),
    ("做饭菜谱", "菜谱步骤、家人口味要求、冰箱库存清单与计时记录"),
    ("宠物照护", "猫狗喂食记录、宠物用品说明、家人交接留言"),
    ("养花园艺", "浇水日志、植物养护卡、天气预报与邻居建议"),
    ("出行打车", "打车订单、司机消息、导航路线与到达时间"),
    ("火车航班", "车票/机票行程单、改签须知、车站广播与同行人消息"),
    ("自驾洗车", "洗车店预约、车辆保养提示、天气预报与朋友建议"),
    ("旅行住宿", "酒店预订确认、入住须知、行程单与同伴群聊"),
    ("健身运动", "运动手环数据、训练计划、教练留言与自我感受"),
    ("校园生活", "班级群通知、作业提交记录、课表与家长消息"),
    ("兴趣课程", "兴趣班排课表、老师通知、签到记录与家长群聊"),
    ("社区物业", "物业通知、门禁记录、报修单与邻居群消息"),
    ("聚会活动", "聚会群聊、接龙报名、场地预订与天气预报"),
    ("智能家居", "音箱/闹钟/扫地机的 App 记录、设置页与家人口头描述"),
    ("搬家整理", "搬家公司订单、物品清单、楼道电梯预约与家人分工"),
    ("二手闲置", "闲置交易聊天、商品描述、实物照片说明与验货记录"),
    ("图书借阅", "图书馆借阅记录、到期提醒、馆内通知与同学消息"),
    ("手机应用", "App 通知、设置页截图文字、更新日志与朋友的建议"),
    ("亲子日程", "家庭日历、接送安排、学校通知与老人口头交代"),
    ("家政预约", "保洁/保姆预约单、服务清单、上门记录与家人反馈"),
    ("天气出行", "天气预报、景区公告、交通提示与朋友圈描述"),
]

# 难度：控制「表面直觉」和「证据结论」的拉开程度
DIFFICULTIES = [
    ("easy", "证据直白，仔细看就能得出结论；表面直觉只是略有偏差"),
    ("medium", "关键证据藏在材料中间，需要对照两处信息"),
    ("hard", "表面说法与证据明确冲突，且有一个诱人但错误的干扰项，要靠时间、数字或原文细节才能排除"),
]

# state 长度档（v2 整体缩短省成本：v1 实测平均 2000 字）
LENGTHS = [("short", "150~250 字"), ("mid", "250~380 字"), ("long", "380~500 字")]

# 生活化关键词：从 alpaca 里优先挑贴近日常的话题当灵感
LIFE_KW = re.compile("吃|菜|饭|餐|厨|烹|食|家|房|宠|猫|狗|花|植物|旅|酒店|出行|车|火车|飞机|天气|"
                     "运动|健身|跑步|锻炼|学校|学生|孩子|父母|朋友|邻居|购物|快递|手机|App|应用|音乐|"
                     "电影|书|图书|周末|假期|节日|生日|聚会|搬家|清洁|打扫|洗|睡|早餐|咖啡|茶|水果|"
                     "游戏|公园|社区|宿舍|时间管理|习惯|日常|生活")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=6000, help="抽多少条种子")
    ap.add_argument("--seed", type=int, default=20260927)
    ap.add_argument("--prefix", default="L", help="id 前缀；v1 用的是 s，别撞号")
    ap.add_argument("--out", default=str(OUT_DIR / "seeds.jsonl"))
    args = ap.parse_args()

    CACHE.parent.mkdir(parents=True, exist_ok=True)
    if not CACHE.exists() or CACHE.stat().st_size < 1_000_000:
        print(f"下载 alpaca-zh -> {CACHE}", flush=True)
        subprocess.run(["curl", "-sL", "--retry", "3", "-o", str(CACHE), URL], check=True)
    rows = json.loads(CACHE.read_text(encoding="utf-8"))
    print(f"alpaca-zh 共 {len(rows)} 条", flush=True)

    # 去重 + 过滤掉太短/太像纯翻译改写的指令
    seen: set[str] = set()
    pool = []
    for r in rows:
        ins = (r.get("instruction") or "").strip()
        inp = (r.get("input") or "").strip()
        topic = (ins + (" " + inp if inp else "")).strip()
        if not (12 <= len(ins) <= 200):
            continue
        h = hashlib.md5(ins.encode()).hexdigest()
        if h in seen:
            continue
        seen.add(h)
        pool.append(topic[:300])
    print(f"过滤去重后 {len(pool)} 条可用", flush=True)
    life = [t for t in pool if LIFE_KW.search(t)]
    print(f"其中生活化话题 {len(life)} 条", flush=True)
    if len(life) >= args.n:
        pool = life

    rng = random.Random(args.seed)
    rng.shuffle(pool)
    if len(pool) < args.n:
        raise SystemExit(f"池子只有 {len(pool)} 条，不够 {args.n}")
    picked = pool[: args.n]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for i, topic in enumerate(picked):
            dom, dom_desc = DOMAINS[i % len(DOMAINS)]
            dif, dif_desc = DIFFICULTIES[(i // len(DOMAINS)) % len(DIFFICULTIES)]
            ln, ln_desc = LENGTHS[(i // (len(DOMAINS) * len(DIFFICULTIES))) % len(LENGTHS)]
            f.write(json.dumps({
                "id": f"{args.prefix}{i:05d}",
                "topic": topic,
                "domain": dom, "domain_desc": dom_desc,
                "difficulty": dif, "difficulty_desc": dif_desc,
                "length": ln, "length_desc": ln_desc,
            }, ensure_ascii=False) + "\n")
    print(f"写出 {len(picked)} 条种子 -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
