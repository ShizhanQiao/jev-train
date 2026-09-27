#!/usr/bin/env python
"""Jev 教具推理服务：加载 best.pt，起一个本地网页，输入题目+选项就出结果。

用法
  bench/.venv/bin/python train/serve.py                     # 默认 runs/full/best.pt，浏览器开 http://127.0.0.1:8765
  bench/.venv/bin/python train/serve.py --ckpt runs/xxx/best.pt --port 9000
  bench/.venv/bin/python train/serve.py --selftest          # 不起服务，拿 val 前 N 条过一遍推理路径核对准确率

编码路径直接复用 train.py 的 JevDataset.__getitem__ + collate，保证和训练时的序列构造逐 token 一致。
接口
  GET  /                -> app.html
  GET  /api/info        -> 模型/训练信息 + val 指标
  GET  /api/example     -> 从 val.jsonl 随机取一条，填进网页做演示
  POST /api/predict     -> {mode, question, options[], state} -> 概率
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
# 全部权重/分词器都在本地缓存里，离线跑，避免漏配 HF_HOME 时静默挂在下载上
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch  # noqa: E402

from train import JevDataset, collate, load_checkpoint  # noqa: E402

MODES = ("auto", "choice", "noul", "score")


class InferSet(JevDataset):
    """只是为了借用父类的 __getitem__（分词/截断/槽位布局），不读文件。"""

    def __init__(self, rows, tok, max_len):
        self.rows, self.tok, self.max_len = rows, tok, max_len


class Engine:
    def __init__(self, ckpt: Path, device: str | None = None):
        self.dev = device or ("mps" if torch.backends.mps.is_available()
                              else "cuda" if torch.cuda.is_available() else "cpu")
        t0 = time.time()
        self.model, self.tok = load_checkpoint(str(ckpt), self.dev)
        self.cfg = json.loads((ckpt.parent / "config.json").read_text(encoding="utf-8"))
        self.max_len = int(self.cfg.get("max_len", 1024))
        self.ckpt = ckpt
        self.lock = threading.Lock()   # MPS 上不要并发前向
        print(f"[serve] {ckpt} -> {self.dev}  max_len={self.max_len}  加载 {time.time()-t0:.1f}s")

    # ---------------------------------------------------------------- 编码
    def build_row(self, mode: str, question: str, options: list[dict], state: str) -> tuple[str, dict]:
        n = len(options)
        if mode == "auto":
            mode = "noul" if n == 0 else "choice"
        if mode in ("choice", "score") and n < 2:
            raise ValueError("选择/打分题至少要 2 个选项")
        row = {"id": "infer", "source": "infer", "state": state}
        if mode == "noul":
            row["noul"] = {"statement": question, "y": 0}
        elif mode == "choice":
            row["choice"] = {"question": question,
                             "options": [{"t": o["t"], "d": o.get("d", "")} for o in options], "y": 0}
        else:
            row["score"] = {"question": question,
                            "levels": [o["t"] + ("：" + o["d"] if o.get("d") else "") for o in options], "y": 0}
        return mode, row

    @torch.no_grad()
    def predict(self, mode: str, question: str, options: list[dict], state: str) -> dict:
        mode, row = self.build_row(mode, question, options, state)
        item = InferSet([row], self.tok, self.max_len)[0]
        batch = collate([item], self.tok.pad_token_id)
        batch = {k: (v.to(self.dev) if torch.is_tensor(v) else v) for k, v in batch.items()}

        t0 = time.time()
        with self.lock:
            noul, choice, score = self.model(batch)
            if self.dev == "mps":
                torch.mps.synchronize()
        ms = int((time.time() - t0) * 1000)

        used = len(item["ids"])
        state_tokens = len(self.tok(state, add_special_tokens=False)["input_ids"]) if state else 0
        out = {
            "task": mode, "ms": ms, "device": self.dev,
            "used_tokens": used, "max_len": self.max_len,
            "state_tokens": state_tokens, "truncated": used >= self.max_len,
        }

        if mode == "noul":
            lg = noul[0].item()
            p = torch.sigmoid(noul[0]).item()
            labels = [o["t"] for o in options] if len(options) == 2 else ["成立（真）", "不成立（假）"]
            out.update(pred=0 if p >= 0.5 else 1, conf=max(p, 1 - p), logit=lg,
                       items=[{"label": labels[0], "p": p, "logit": lg},
                              {"label": labels[1], "p": 1 - p, "logit": -lg}])
            return out

        lg = (choice if mode == "choice" else score)[0]
        k = len(options)
        lg = lg[:k]
        prob = lg.softmax(-1)
        out.update(pred=int(prob.argmax()), conf=float(prob.max()),
                   items=[{"label": options[i]["t"], "desc": options[i].get("d", ""),
                           "p": float(prob[i]), "logit": float(lg[i])} for i in range(k)])
        if mode == "score":
            ev = float((prob * torch.arange(k, dtype=prob.dtype, device=prob.device)).sum())
            out["expected_level"] = ev            # 0-based 概率加权档位，和训练时的期望值项同一个量
        return out


# ---------------------------------------------------------------- 训练信息

def read_metrics(run_dir: Path) -> dict:
    log = run_dir / "log.jsonl"
    if not log.exists():
        return {}
    vals = [json.loads(l) for l in log.read_text(encoding="utf-8").splitlines()
            if l.strip() and '"val"' in l]
    if not vals:
        return {}
    best = min(vals, key=lambda r: r["loss_total"])
    keep = ("step", "loss_total", "acc_noul", "acc_choice", "acc_score",
            "ece_noul", "ece_choice", "ece_score", "mae_score")
    return {"best": {k: best[k] for k in keep if k in best},
            "acc_by_source": best.get("acc_by_source", {}), "n_evals": len(vals)}


# ---------------------------------------------------------------- HTTP

class Handler(BaseHTTPRequestHandler):
    engine: Engine
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *a):      # 只打 API，静音静态文件
        if "/api/" in (self.path or ""):
            sys.stderr.write(f"[{time.strftime('%H:%M:%S')}] {fmt % a}\n")

    def _send(self, code: int, body: bytes, ctype: str):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code: int = 200):
        self._send(code, json.dumps(obj, ensure_ascii=False).encode(), "application/json; charset=utf-8")

    def do_GET(self):
        path = (self.path or "/").split("?")[0]
        if path in ("/", "/index.html", "/app.html"):
            f = HERE / "app.html"
            if not f.exists():
                return self._json({"error": "缺少 train/app.html"}, 500)
            return self._send(200, f.read_bytes(), "text/html; charset=utf-8")
        if path == "/api/info":
            e = self.engine
            return self._json({
                "ckpt": str(e.ckpt), "device": e.dev, "max_len": e.max_len,
                "base_model": e.cfg.get("model"), "epochs": e.cfg.get("epochs"),
                "metrics": read_metrics(e.ckpt.parent),
            })
        if path == "/api/example":
            return self._json(self.pick_example())
        return self._json({"error": "not found"}, 404)

    def do_POST(self):
        if (self.path or "").split("?")[0] != "/api/predict":
            return self._json({"error": "not found"}, 404)
        try:
            n = int(self.headers.get("Content-Length") or 0)
            req = json.loads(self.rfile.read(n) or b"{}")
            mode = req.get("mode", "auto")
            if mode not in MODES:
                raise ValueError(f"mode 只能是 {MODES}")
            question = (req.get("question") or "").strip()
            state = req.get("state") or ""
            options = [o for o in (req.get("options") or []) if (o.get("t") or "").strip()]
            if not question:
                raise ValueError("题目/陈述不能为空")
            if len(options) > 24:
                raise ValueError("选项最多 24 个")
            return self._json(self.engine.predict(mode, question, options, state))
        except ValueError as e:
            return self._json({"error": str(e)}, 400)
        except Exception as e:                       # noqa: BLE001
            import traceback
            traceback.print_exc()
            return self._json({"error": f"{type(e).__name__}: {e}"}, 500)

    def pick_example(self) -> dict:
        path = HERE / "data" / "val.jsonl"
        if not path.exists():
            return {"error": "没有 data/val.jsonl"}
        lines = [l for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
        r = json.loads(random.choice(lines))
        which = random.choice([t for t in ("choice", "score", "noul") if t in r])
        out = {"state": r["state"], "source": r["source"], "task": which, "id": r.get("id")}
        if which == "choice":
            out["question"] = r["choice"]["question"]
            out["options"] = [{"t": o["t"], "d": o.get("d", "")} for o in r["choice"]["options"]]
            out["gold"] = r["choice"]["y"]
        elif which == "score":
            out["question"] = r["score"]["question"]
            out["options"] = [{"t": lv, "d": ""} for lv in r["score"]["levels"]]
            out["gold"] = r["score"]["y"]
        else:
            out["question"] = r["noul"]["statement"]
            out["options"] = []
            out["gold"] = r["noul"]["y"]
        return out


# ---------------------------------------------------------------- 自检

def selftest(engine: Engine, n: int):
    """把 val 前 n 条按网页的输入方式重新走一遍，核对和训练时的评估对得上。"""
    rows = [json.loads(l) for l in (HERE / "data" / "val.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    rows = rows[:n]
    stat = {t: [0, 0] for t in ("noul", "choice", "score")}
    noul_pred = [0, 0]
    for i, r in enumerate(rows, 1):
        if "choice" in r:
            o = [{"t": x["t"], "d": x.get("d", "")} for x in r["choice"]["options"]]
            p = engine.predict("choice", r["choice"]["question"], o, r["state"])
            stat["choice"][0] += p["pred"] == r["choice"]["y"]; stat["choice"][1] += 1
        if "score" in r:
            o = [{"t": lv, "d": ""} for lv in r["score"]["levels"]]
            p = engine.predict("score", r["score"]["question"], o, r["state"])
            stat["score"][0] += p["pred"] == r["score"]["y"]; stat["score"][1] += 1
        if "noul" in r:
            p = engine.predict("noul", r["noul"]["statement"], [], r["state"])
            pred = 1 if p["items"][0]["p"] >= 0.5 else 0
            noul_pred[pred] += 1
            stat["noul"][0] += pred == r["noul"]["y"]; stat["noul"][1] += 1
        if i % 10 == 0:
            print(f"  {i}/{len(rows)} " + "  ".join(
                f"{t}={c/max(1,m):.3f}({m})" for t, (c, m) in stat.items()))
    print("\n[selftest] " + "  ".join(f"acc_{t}={c/max(1,m):.4f} (n={m})" for t, (c, m) in stat.items()))
    print(f"[selftest] noul 预测分布：假={noul_pred[0]} 真={noul_pred[1]}  "
          f"（全押多数类的基线 = {max(sum(1 for r in rows if r.get('noul',{}).get('y')==k) for k in (0,1))/max(1,stat['noul'][1]):.4f}）")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(HERE / "runs" / "full" / "best.pt"))
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--device", default=None, help="mps / cpu / cuda，默认自动")
    ap.add_argument("--no-open", action="store_true")
    ap.add_argument("--selftest", type=int, nargs="?", const=40, default=None, metavar="N")
    a = ap.parse_args()

    ckpt = Path(a.ckpt)
    if not ckpt.is_absolute():
        ckpt = HERE / ckpt
    if not ckpt.exists():
        sys.exit(f"找不到 checkpoint：{ckpt}")
    engine = Engine(ckpt, a.device)

    if a.selftest is not None:
        return selftest(engine, a.selftest)

    Handler.engine = engine
    srv = ThreadingHTTPServer((a.host, a.port), Handler)
    url = f"http://{a.host}:{a.port}/"
    print(f"[serve] 打开 {url}   (Ctrl-C 停)")
    if not a.no_open:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n[serve] bye")


if __name__ == "__main__":
    main()
