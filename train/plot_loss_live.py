"""把 runs/*/log.jsonl 回放成一段"边训边画"的损失曲线动画。

点随训练步数逐个冒出来，坐标轴跟着数据平滑缩放，黑底、带图例，
直接导出 mp4/gif 给视频用；--preview 则开窗口实时看。

  python plot_loss_live.py                      # 导出 mp4
  python plot_loss_live.py --preview            # 只看不存
  python plot_loss_live.py --keys total choice  # 只画部分曲线
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib import patheffects as pe

HERE = Path(__file__).resolve().parent

# 曲线：key -> (图例, 颜色, 线宽)
CURVES = {
    "total": ("总损失  loss_total", "#ffffff", 3.2),
    "choice": ("选项  choice", "#4cc9ff", 2.0),
    "score": ("打分  score", "#ffd166", 2.0),
    "noul": ("判断  noul", "#8affc1", 2.0),
    "rlcd": ("对比  RLCD", "#ff6b9d", 2.0),
}


def pick_font() -> None:
    have = {f.name for f in fm.fontManager.ttflist}
    want = [n for n in ("PingFang SC", ".PingFang SC", "Heiti SC", "Arial Unicode MS") if n in have]
    plt.rcParams["font.sans-serif"] = want + ["DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def load(path: Path, keys):
    """读 jsonl，返回 train/val 两组 {key: (xs, ys)} 和 x 轴（step）。"""
    train, val = {"x": []}, {"x": []}
    for k in keys:
        train[k], val[k] = [], []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        d = train if r.get("split") == "train" else val
        d["x"].append(r["step"])
        for k in keys:
            d[k].append(r.get(f"loss_{k}", float("nan")))
    return train, val


def lerp(a, b, t):
    return a + (b - a) * t


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default=str(HERE / "runs/full/log.jsonl"))
    ap.add_argument("--out", default=str(HERE / "runs/full/loss_live.mp4"))
    ap.add_argument("--keys", nargs="+", default=list(CURVES))
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--sub", type=int, default=3, help="每个数据点占几帧，越大越慢")
    ap.add_argument("--hold", type=float, default=2.0, help="结尾定格秒数")
    ap.add_argument("--size", default="1920x1080")
    ap.add_argument("--dpi", type=int, default=100)
    ap.add_argument("--logy", action="store_true", help="y 轴取对数")
    ap.add_argument("--ema", type=float, default=0.0,
                    help="0~1，给训练曲线加 EMA 平滑；原始值退成淡线。0=关")
    ap.add_argument("--preview", action="store_true", help="开窗口看，不导出")
    a = ap.parse_args()

    keys = [k for k in a.keys if k in CURVES]
    tr, va = load(Path(a.log), keys)
    n = len(tr["x"])
    if n < 2:
        raise SystemExit(f"{a.log} 里训练点不够：{n}")

    raw = {}
    if a.ema > 0:
        for k in keys:  # 淡线画原始值，粗线画 EMA
            raw[k] = tr[k]
            sm, acc = [], None
            for v in tr[k]:
                acc = v if acc is None else acc * a.ema + v * (1 - a.ema)
                sm.append(acc)
            tr[k] = sm

    pick_font()
    if a.preview:
        matplotlib.use("MacOSX", force=True)
    w, h = (int(v) for v in a.size.lower().split("x"))
    fig, ax = plt.subplots(figsize=(w / a.dpi, h / a.dpi), dpi=a.dpi, facecolor="#000000")
    fig.subplots_adjust(left=0.08, right=0.97, top=0.90, bottom=0.11)
    ax.set_facecolor("#000000")
    for s in ax.spines.values():
        s.set_color("#3a3a3a")
    ax.tick_params(colors="#9a9a9a", labelsize=13)
    ax.grid(True, color="#242424", lw=0.9)
    ax.set_axisbelow(True)
    if a.logy:
        ax.set_yscale("log")

    ax.set_title("Jev 教具训练 · 损失下降", color="#f2f2f2", fontsize=22, pad=18, loc="left")
    ax.set_xlabel("训练步数 step", color="#9a9a9a", fontsize=15, labelpad=10)
    ax.set_ylabel("loss", color="#9a9a9a", fontsize=15, labelpad=10)

    glow = [pe.Stroke(linewidth=CURVES["total"][2] + 5, foreground="#ffffff", alpha=0.10), pe.Normal()]
    lines, heads, vlines, rawl = {}, {}, {}, {}
    for k in keys:
        label, color, lw = CURVES[k]
        if raw:
            (rawl[k],) = ax.plot([], [], color=color, lw=1.0, alpha=0.22)
        (lines[k],) = ax.plot([], [], color=color, lw=lw, label=label, solid_capstyle="round",
                              path_effects=glow if k == "total" else None)
        # 队首亮点 + 验证集星标（星标不进图例，另用一行文字说明）
        (heads[k],) = ax.plot([], [], "o", color=color, ms=8, mec="#000000", mew=1.2)
        (vlines[k],) = ax.plot([], [], "*", color=color, ms=16, ls="--", lw=1.1, alpha=0.85)

    leg = ax.legend(loc="upper right", fontsize=15, framealpha=0.0, labelcolor="#e8e8e8",
                    handlelength=2.4, borderpad=1.0)
    leg.set_zorder(5)
    hud = ax.text(0.012, 0.965, "", transform=ax.transAxes, color="#8a8a8a", fontsize=15,
                  va="top", ha="left", family="monospace")
    ax.text(0.012, 0.03, "★ = 验证集 (每轮一次)", transform=ax.transAxes, color="#6f6f6f", fontsize=13)

    # 坐标轴目标值，逐帧缓动过去，避免每加一个点就跳一下
    lim = None
    hold = int(a.hold * a.fps)
    total_frames = (n - 1) * a.sub + 1 + hold

    def frame(f):
        nonlocal lim
        t = min(f, (n - 1) * a.sub) / a.sub
        i = min(int(t), n - 1)
        frac = t - i

        xs = tr["x"][: i + 1]
        x_now = xs[-1] if frac == 0 or i == n - 1 else lerp(tr["x"][i], tr["x"][i + 1], frac)
        lo, hi = float("inf"), float("-inf")

        for k in keys:
            ys = tr[k][: i + 1]
            px, py = list(xs), list(ys)
            if frac > 0 and i < n - 1:  # 让线段连续长出来，而不是一格一格蹦
                px.append(x_now)
                py.append(lerp(tr[k][i], tr[k][i + 1], frac))
            lines[k].set_data(px, py)
            heads[k].set_data(px[-1:], py[-1:])
            if raw:
                rx, ry = list(xs), raw[k][: i + 1]
                if frac > 0 and i < n - 1:
                    rx, ry = rx + [x_now], ry + [lerp(raw[k][i], raw[k][i + 1], frac)]
                rawl[k].set_data(rx, ry)
                py = py + ry  # 缩放要把淡线也框进去

            m = [j for j, vx in enumerate(va["x"]) if vx <= x_now]
            vlines[k].set_data([va["x"][j] for j in m], [va[k][j] for j in m])

            vis = [v for v in py + [va[k][j] for j in m] if v == v]
            if vis:
                lo, hi = min(lo, min(vis)), max(hi, max(vis))

        if lo > hi:
            return
        pad = max((hi - lo) * 0.12, 0.05)
        if a.logy:
            tgt = (max(lo * 0.8, 1e-4), hi * 1.25, 0.0, max(x_now * 1.04, tr["x"][0] * 2))
        else:
            tgt = (max(lo - pad, 0.0), hi + pad, 0.0, max(x_now * 1.04, tr["x"][0] * 2))
        lim = tgt if lim is None else tuple(lerp(c, d, 0.18) for c, d in zip(lim, tgt))
        ax.set_ylim(lim[0], lim[1])
        ax.set_xlim(lim[2], lim[3])

        txt = f"step {int(x_now):>5d} / {tr['x'][-1]}"
        if "total" in keys:
            txt += f"    total {(raw or tr)['total'][i]:.3f}"  # HUD 报原始值，不报平滑值
        hud.set_text(txt)

    anim = animation.FuncAnimation(fig, frame, frames=total_frames, interval=1000 / a.fps, repeat=False)

    if a.preview:
        plt.show()
        return

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix == ".gif":
        anim.save(out, writer=animation.PillowWriter(fps=a.fps))
    else:
        anim.save(out, writer=animation.FFMpegWriter(
            fps=a.fps, bitrate=-1, codec="libx264",
            extra_args=["-pix_fmt", "yuv420p", "-crf", "18", "-preset", "slow"]))
    print(f"{out}  ({total_frames} 帧 / {total_frames / a.fps:.1f}s / {w}x{h})")


if __name__ == "__main__":
    main()
