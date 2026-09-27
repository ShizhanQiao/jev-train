#!/bin/bash
# 全量跑：阶段1 合成案例 -> 阶段2 RLCD 正负 -> 阶段3 合并+质检
# 三个阶段都支持断点续跑，中断或余额耗尽后重跑本脚本即可接上。
# key 从项目根目录 .env 的 doubao_api 读（或环境变量 ARK_API_KEY），不要写进代码。
set -e
cd "$(dirname "$0")/.."
PY="${PY:-bench/.venv/bin/python}"
C="${C:-300}"            # 并发
TARGET="${TARGET:-7500}" # 目标有效案例数（含 v1 已有的 2500 条）

echo "=== 阶段1：合成案例，目标 $TARGET 条（并发 $C）==="
$PY rlcd/synth_cases.py --target "$TARGET" -c "$C"
echo
echo "=== 阶段2：RLCD 正负对照（并发 $C）==="
$PY rlcd/gen_rlcd.py -c "$C"
echo
echo "=== 阶段3：合并 + 质检 ==="
$PY rlcd/build_pairs.py
