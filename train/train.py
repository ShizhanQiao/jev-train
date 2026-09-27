#!/usr/bin/env python
"""Jev 式三题型打分器：xlm-roberta-base + 一个共享的 (768,1) 打分头。

用法
  python3 train.py --smoke            # 64 条小样本冒烟，几分钟
  python3 train.py                    # 全量
  python3 train.py --max-len 512 --bs 4 --accum 4 --run runs/l512   # 快 2.5 倍，但材料只能看到前 1/4

架构（一次前向出三题的答案）
  <s> 判断:陈述 </s> 选择:问题 [OPT]选项1 [OPT]选项2 … </s> 打分:问题 [LVL]档1 [LVL]档2 … </s> 材料 </s>
  - [OPT]/[LVL] 物理上都是 <mask> token，靠一个可训练的 slot-type embedding 区分（0 普通 / 1 选项 / 2 档位）
  - 共享打分头 scorer: Linear(768,1) 作用在 <s>（判断）、每个 [OPT]（选择）、每个 [LVL]（打分）的 hidden state 上
  - 选项/档位数逐条动态，padding 槽位 logit 填 -inf 再 softmax
  - 词向量冻结（250k 词表占了 2/3 参数，冻掉省 ~2.3GB 优化器内存），其余全训

损失
  L = λn·BCE(noul) + λc·CE(choice) + λs·[CE(score) + μ·|E[档]−y|/(K−1)] + λr·RLCD
  RLCD（仅合成数据、仅 pos 对 neg 错的干净对）: −log σ(β·(log p(pos) − log p(neg)))

日志
  runs/<run>/log.jsonl  每行一条，split=train 每 log_every 步一条、split=val 每次评估一条，
  含各分项 loss、各题型 acc、score MAE、ECE。live plot 直接读它。
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Sampler
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HOME", str(HERE.parent / "bench" / "hf_cache"))
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from transformers import AutoModel, AutoTokenizer, logging as hf_logging  # noqa: E402

hf_logging.set_verbosity_error()   # 关掉 "sequence length > 512" 之类的噪音（我们自己截断）

TASKS = ("noul", "choice", "score")
SLOT_TEXT, SLOT_OPT, SLOT_LVL = 0, 1, 2
OPT_TOK_MAX, LVL_TOK_MAX = 28, 14   # 单个选项 / 档位文本的 token 上限，保证问题块不吃光预算
STATE_MIN = 64                      # 材料至少留这么多 token；不够就压缩选项/档位文本


# ------------------------------------------------------------------ 数据

class JevDataset(Dataset):
    def __init__(self, path: Path, tok, max_len: int, limit: int | None = None):
        self.rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
        if limit:
            self.rows = self.rows[:limit]
        self.tok, self.max_len = tok, max_len

    def __len__(self):
        return len(self.rows)

    def _t(self, s: str, cap: int | None = None) -> list[int]:
        ids = self.tok(s, add_special_tokens=False)["input_ids"]
        return ids[:cap] if cap else ids

    def length_hint(self, i) -> int:
        """粗略长度（字符数），只用来给 BucketSampler 分组，不必精确。"""
        r = self.rows[i]
        return min(len(r["state"]), 4 * self.max_len) + sum(
            len(o["t"]) + len(o.get("d", "")) for o in r.get("choice", {}).get("options", []))

    def __getitem__(self, i):
        r, tok = self.rows[i], self.tok
        cls, sep, mask = tok.cls_token_id, tok.sep_token_id, tok.mask_token_id

        # 先把问题块的各段切好；若总长超过预算，就按比例压每个选项/档位的 token 上限
        opts = [self._t(f'{o["t"]}：{o["d"]}' if o.get("d") else o["t"], OPT_TOK_MAX)
                for o in r.get("choice", {}).get("options", [])]
        lvls = [self._t(lv, LVL_TOK_MAX) for lv in r.get("score", {}).get("levels", [])]
        heads = {
            "noul": self._t("判断：" + r["noul"]["statement"], 96) if "noul" in r else None,
            "choice": self._t("选择：" + r["choice"]["question"], 64) if "choice" in r else None,
            "score": self._t("打分：" + r["score"]["question"], 64) if "score" in r else None,
        }
        fixed = 1 + sum(len(h) + 1 for h in heads.values() if h is not None) + len(opts) + len(lvls)
        room = self.max_len - STATE_MIN - fixed
        if sum(map(len, opts + lvls)) > room:
            cap = max(2, room // max(1, len(opts) + len(lvls)))
            opts, lvls = [x[:cap] for x in opts], [x[:cap] for x in lvls]

        ids, st = [cls], [SLOT_TEXT]
        opt_pos, lvl_pos = [], []

        def push(x, slot=SLOT_TEXT):
            ids.extend(x); st.extend([slot] * len(x))

        if heads["noul"] is not None:
            push(heads["noul"]); push([sep])
        if heads["choice"] is not None:
            push(heads["choice"])
            for o in opts:
                opt_pos.append(len(ids)); push([mask], SLOT_OPT); push(o)
            push([sep])
        if heads["score"] is not None:
            push(heads["score"])
            for lv in lvls:
                lvl_pos.append(len(ids)); push([mask], SLOT_LVL); push(lv)
            push([sep])
        # 材料放最后，预算剩多少截多少（问题块永远完整，槽位不会被截掉）
        budget = self.max_len - len(ids) - 1
        push(self._t(r["state"])[:max(budget, 0)]); push([sep])
        assert len(ids) <= self.max_len, (len(ids), self.max_len)

        rl = r.get("rlcd", {})
        return {
            "ids": ids, "slot": st, "opt_pos": opt_pos, "lvl_pos": lvl_pos,
            "noul_y": r["noul"]["y"] if "noul" in r else -1,
            "choice_y": r["choice"]["y"] if "choice" in r else -1,
            "score_y": r["score"]["y"] if "score" in r else -1,
            **{f"rl_{t}_{s}": rl[t][s] if t in rl else -1 for t in TASKS for s in ("pos", "neg")},
            "source": r["source"],
        }


class BucketSampler(Sampler):
    """近似按长度分组成批：打乱 → 每 window 批一个窗口内按长度排序 → 切批 → 批序再打乱。
    合成数据 ~1000 token、公开数据集 ~50 token，混着随机成批要多算 28% 的 padding。"""

    def __init__(self, lengths: list[int], bs: int, window: int = 50):
        self.lengths, self.bs, self.window = lengths, bs, window

    def __iter__(self):
        idx = list(range(len(self.lengths))); random.shuffle(idx)
        batches, w = [], self.bs * self.window
        for k in range(0, len(idx), w):
            chunk = sorted(idx[k: k + w], key=self.lengths.__getitem__)
            batches += [chunk[j: j + self.bs] for j in range(0, len(chunk), self.bs)]
        random.shuffle(batches)
        return iter(batches)

    def __len__(self):
        return math.ceil(len(self.lengths) / self.bs)


def collate(batch, pad_id: int):
    B, L = len(batch), max(len(b["ids"]) for b in batch)
    NO, NL = max(1, max(len(b["opt_pos"]) for b in batch)), max(1, max(len(b["lvl_pos"]) for b in batch))
    out = {
        "ids": torch.full((B, L), pad_id), "am": torch.zeros(B, L, dtype=torch.long),
        "slot": torch.zeros(B, L, dtype=torch.long),
        "opt_idx": torch.zeros(B, NO, dtype=torch.long), "opt_mask": torch.zeros(B, NO, dtype=torch.bool),
        "lvl_idx": torch.zeros(B, NL, dtype=torch.long), "lvl_mask": torch.zeros(B, NL, dtype=torch.bool),
    }
    for i, b in enumerate(batch):
        n = len(b["ids"])
        out["ids"][i, :n] = torch.tensor(b["ids"]); out["am"][i, :n] = 1
        out["slot"][i, :n] = torch.tensor(b["slot"])
        if b["opt_pos"]:
            out["opt_idx"][i, :len(b["opt_pos"])] = torch.tensor(b["opt_pos"]); out["opt_mask"][i, :len(b["opt_pos"])] = True
        if b["lvl_pos"]:
            out["lvl_idx"][i, :len(b["lvl_pos"])] = torch.tensor(b["lvl_pos"]); out["lvl_mask"][i, :len(b["lvl_pos"])] = True
    for k in [k for k in batch[0] if k.endswith("_y") or k.startswith("rl_")]:
        out[k] = torch.tensor([b[k] for b in batch])
    out["source"] = [b["source"] for b in batch]
    return out


# ------------------------------------------------------------------ 模型

def extend_positions(backbone, max_len: int):
    """RoBERTa 的位置向量是学出来的 514 个；超过就把预训练的块平铺复制（Longformer 初始化法）。"""
    e = backbone.embeddings
    old = e.position_embeddings.weight.data
    off, blk, n = 2, old.size(0) - 2, max_len + 2
    if n <= old.size(0):
        return
    new = torch.zeros(n, old.size(1))
    new[:off] = old[:off]
    for i in range(0, n - off, blk):
        k = min(blk, n - off - i); new[off + i: off + i + k] = old[off: off + k]
    pe = nn.Embedding(n, old.size(1), padding_idx=e.position_embeddings.padding_idx)
    pe.weight.data = new
    e.position_embeddings = pe
    e.position_ids = nn.Buffer(torch.arange(n).expand((1, -1)), persistent=False)
    e.token_type_ids = nn.Buffer(torch.zeros((1, n), dtype=torch.long), persistent=False)
    backbone.config.max_position_embeddings = n


class JevLike(nn.Module):
    def __init__(self, name: str, max_len: int, freeze_word_emb: bool = True, grad_ckpt: bool = True):
        super().__init__()
        # attention dropout 必须关：MPS 上 SDPA 只要 dropout_p>0 就退化到慢路径，实测慢 9 倍（1.3s → 11.5s/step）。
        # hidden dropout 不受影响，照常保留。
        self.backbone = AutoModel.from_pretrained(name, add_pooling_layer=False, attention_probs_dropout_prob=0.0)
        extend_positions(self.backbone, max_len)
        H = self.backbone.config.hidden_size
        self.marker = nn.Embedding(3, H)          # slot-type embedding：区分普通 / [OPT] / [LVL]
        nn.init.normal_(self.marker.weight, std=0.02)
        self.scorer = nn.Linear(H, 1)             # 唯一的打分头，所有槽位共享
        if freeze_word_emb:
            self.backbone.embeddings.word_embeddings.weight.requires_grad_(False)
        if grad_ckpt:
            self.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    def forward(self, b):
        emb = self.backbone.embeddings.word_embeddings(b["ids"]) + self.marker(b["slot"])
        h = self.backbone(inputs_embeds=emb, attention_mask=b["am"]).last_hidden_state
        s = self.scorer(h).squeeze(-1)                                   # [B, L] 每个位置一个分
        noul = s[:, 0]                                                   # <s> 位 → 判断
        choice = s.gather(1, b["opt_idx"]).masked_fill(~b["opt_mask"], float("-inf"))
        score = s.gather(1, b["lvl_idx"]).masked_fill(~b["lvl_mask"], float("-inf"))
        return noul, choice, score


def load_checkpoint(path, dev: str = "cpu"):
    """推理用：from train import load_checkpoint; model, tok = load_checkpoint('runs/x/best.pt', 'mps')"""
    ck = torch.load(path, map_location="cpu")
    c = ck["cfg"]
    model = JevLike(c["model"], c["max_len"], not c["no_freeze"], grad_ckpt=False)
    model.load_state_dict(ck["model"], strict=False)
    return model.to(dev).eval(), AutoTokenizer.from_pretrained(c["model"])


# ------------------------------------------------------------------ 损失

def logp_noul(logit, ans):
    return F.logsigmoid(torch.where(ans == 1, logit, -logit))


def compute_losses(out, b, cfg) -> dict[str, torch.Tensor]:
    noul, choice, score = out
    L: dict[str, torch.Tensor] = {}

    m = b["noul_y"] >= 0
    if m.any():
        L["noul"] = F.binary_cross_entropy_with_logits(noul[m], b["noul_y"][m].float())
    m = b["choice_y"] >= 0
    if m.any():
        L["choice"] = F.cross_entropy(choice[m], b["choice_y"][m])
    m = b["score_y"] >= 0
    if m.any():
        lg, y = score[m], b["score_y"][m]
        K = b["lvl_mask"][m].sum(-1).clamp(min=2)
        ev = (lg.softmax(-1) * torch.arange(lg.size(1), device=lg.device)).sum(-1)   # 概率加权档位
        L["score"] = F.cross_entropy(lg, y) + cfg.mu * ((ev - y).abs() / (K - 1)).mean()

    # RLCD 对比：pos 的对数概率要高于 neg
    rl = []
    for t, lg in (("noul", noul), ("choice", choice), ("score", score)):
        m = b[f"rl_{t}_pos"] >= 0
        if not m.any():
            continue
        p, n = b[f"rl_{t}_pos"][m], b[f"rl_{t}_neg"][m]
        if t == "noul":
            lp, ln = logp_noul(lg[m], p), logp_noul(lg[m], n)
        else:
            lsm = lg[m].log_softmax(-1)
            lp, ln = lsm.gather(1, p[:, None]).squeeze(1), lsm.gather(1, n[:, None]).squeeze(1)
        rl.append(-F.logsigmoid(cfg.beta * (lp - ln)))
    if rl:
        L["rlcd"] = torch.cat(rl).mean()

    w = {"noul": cfg.l_noul, "choice": cfg.l_choice, "score": cfg.l_score, "rlcd": cfg.l_rlcd}
    L["total"] = sum(w[k] * v for k, v in L.items())
    return L


# ------------------------------------------------------------------ 评估

def ece(conf: torch.Tensor, hit: torch.Tensor, bins: int = 10) -> float:
    """Expected Calibration Error：置信度分桶后 |平均置信度 − 实际准确率| 的加权和。"""
    if conf.numel() == 0:
        return float("nan")
    edges = torch.linspace(0, 1, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.float().mean().item() * (conf[m].mean() - hit[m].float().mean()).abs().item()
    return e


@torch.no_grad()
def evaluate(model, loader, cfg, dev) -> dict:
    model.eval()
    sums: dict[str, float] = {}; cnt: dict[str, int] = {}
    conf = {t: [] for t in TASKS}; hit = {t: [] for t in TASKS}; src_hit = {}
    mae = []
    for b in loader:
        b = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in b.items()}
        out = model(b)
        L = compute_losses(out, b, cfg)
        for k, v in L.items():   # 各分项只在部分 batch 出现（rlcd 尤其），按各自出现次数平均
            sums[k] = sums.get(k, 0.0) + v.item(); cnt[k] = cnt.get(k, 0) + 1
        noul, choice, score = out
        preds = {
            "noul": (noul.sigmoid(), (noul > 0).long(), b["noul_y"]),
            "choice": (choice.softmax(-1).max(-1).values, choice.argmax(-1), b["choice_y"]),
            "score": (score.softmax(-1).max(-1).values, score.argmax(-1), b["score_y"]),
        }
        for t, (c, p, y) in preds.items():
            m = y >= 0
            if not m.any():
                continue
            c = torch.where(p[m] == 1, c[m], 1 - c[m]) if t == "noul" else c[m]
            h = (p[m] == y[m])
            conf[t].append(c.cpu()); hit[t].append(h.cpu())
            for s, ok in zip([b["source"][i] for i in m.nonzero().flatten().tolist()], h.tolist()):
                src_hit.setdefault(f"{t}/{s}", []).append(ok)
            if t == "score":
                mae.append((p[m] - y[m]).abs().float().cpu())
    model.train()
    r = {f"loss_{k}": v / cnt[k] for k, v in sums.items()}
    for t in TASKS:
        if conf[t]:
            c, h = torch.cat(conf[t]), torch.cat(hit[t])
            r[f"acc_{t}"] = h.float().mean().item()
            r[f"conf_{t}"] = c.mean().item()
            r[f"ece_{t}"] = ece(c, h)
    if mae:
        r["mae_score"] = torch.cat(mae).mean().item()
    r["acc_by_source"] = {k: sum(v) / len(v) for k, v in sorted(src_hit.items())}
    return r


# ------------------------------------------------------------------ 主流程

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="FacebookAI/xlm-roberta-base")
    ap.add_argument("--data", default=str(HERE / "data"))
    ap.add_argument("--run", default=None)
    ap.add_argument("--max-len", type=int, default=1024, help="M1 8GB 实测：512≈18分/epoch, 1024≈45分/epoch")
    ap.add_argument("--bs", type=int, default=2)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--warmup", type=float, default=0.06)
    ap.add_argument("--l-noul", type=float, default=1.0)
    ap.add_argument("--l-choice", type=float, default=1.0)
    ap.add_argument("--l-score", type=float, default=1.0)
    ap.add_argument("--l-rlcd", type=float, default=0.5)
    ap.add_argument("--mu", type=float, default=0.5, help="score 期望值项权重")
    ap.add_argument("--beta", type=float, default=2.0, help="RLCD 温度")
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--eval-every", type=int, default=200, help="按优化步计")
    ap.add_argument("--no-freeze", action="store_true")
    ap.add_argument("--no-ckpt", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="64/32 条小样本冒烟")
    ap.add_argument("--limit", type=int, default=None, help="只用前 N 条训练（验证取 N/4），做中等规模试跑")
    ap.add_argument("--seed", type=int, default=42)
    cfg = ap.parse_args()

    if cfg.smoke:
        cfg.run = cfg.run or "runs/smoke"; cfg.epochs, cfg.eval_every, cfg.log_every = 2, 8, 2
        cfg.max_len, cfg.bs, cfg.accum = min(cfg.max_len, 384), 4, 1
    cfg.run = cfg.run or f"runs/{time.strftime('%m%d-%H%M')}"
    run = HERE / cfg.run; run.mkdir(parents=True, exist_ok=True)
    (run / "config.json").write_text(json.dumps(vars(cfg), indent=1, ensure_ascii=False))
    log_f = (run / "log.jsonl").open("a", encoding="utf-8")

    def log(rec):
        log_f.write(json.dumps(rec, ensure_ascii=False) + "\n"); log_f.flush()

    random.seed(cfg.seed); torch.manual_seed(cfg.seed)
    dev = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

    tok = AutoTokenizer.from_pretrained(cfg.model)
    lim = (64, 32) if cfg.smoke else ((cfg.limit, max(8, cfg.limit // 4)) if cfg.limit else (None, None))
    tr = JevDataset(Path(cfg.data) / "train.jsonl", tok, cfg.max_len, lim[0])
    va = JevDataset(Path(cfg.data) / "val.jsonl", tok, cfg.max_len, lim[1])
    col = lambda b: collate(b, tok.pad_token_id)  # noqa: E731
    tr_dl = DataLoader(tr, batch_sampler=BucketSampler([tr.length_hint(i) for i in range(len(tr))], cfg.bs),
                       collate_fn=col)
    va_dl = DataLoader(va, cfg.bs, shuffle=False, collate_fn=col)

    model = JevLike(cfg.model, cfg.max_len, not cfg.no_freeze, not cfg.no_ckpt).to(dev)
    params = [p for p in model.parameters() if p.requires_grad]
    print(f"device={dev}  train={len(tr)} val={len(va)}  可训练参数 {sum(p.numel() for p in params)/1e6:.1f}M  "
          f"max_len={cfg.max_len} bs={cfg.bs}x{cfg.accum}  -> {run}")
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=0.01)
    total = math.ceil(len(tr_dl) / cfg.accum) * cfg.epochs
    warm = int(total * cfg.warmup)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: s / max(1, warm) if s < warm else max(0.0, (total - s) / max(1, total - warm)))

    step, best = 0, float("inf")
    acc: dict[str, float] = {}; acc_n: dict[str, int] = {}
    model.train()
    for ep in range(cfg.epochs):
        pbar = tqdm(tr_dl, desc=f"ep{ep}", dynamic_ncols=True)
        for i, b in enumerate(pbar):
            b = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in b.items()}
            L = compute_losses(model(b), b, cfg)
            (L["total"] / cfg.accum).backward()
            for k, v in L.items():
                acc[k] = acc.get(k, 0.0) + v.item(); acc_n[k] = acc_n.get(k, 0) + 1
            if (i + 1) % cfg.accum and (i + 1) != len(tr_dl):
                continue
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True); step += 1

            if step % cfg.log_every == 0:
                rec = {"split": "train", "step": step, "epoch": ep + i / len(tr_dl), "lr": sched.get_last_lr()[0],
                       **{f"loss_{k}": v / acc_n[k] for k, v in acc.items()}}
                log(rec); acc, acc_n = {}, {}
                pbar.set_postfix({k[5:]: f"{v:.3f}" for k, v in rec.items() if k.startswith("loss_")})
            if step % cfg.eval_every == 0 or step == total:
                r = evaluate(model, va_dl, cfg, dev)
                log({"split": "val", "step": step, "epoch": ep + i / len(tr_dl), **r})
                tqdm.write(f"[val@{step}] " + "  ".join(
                    f"{k}={v:.3f}" for k, v in r.items() if isinstance(v, float)))
                if r["loss_total"] < best:
                    best = r["loss_total"]
                    # 只存可训练部分（冻结的 250k 词表占 770MB，加载时从预训练权重取）
                    sd = {k: v for k, v in model.state_dict().items()
                          if k != "backbone.embeddings.word_embeddings.weight" or cfg.no_freeze}
                    torch.save({"model": sd, "cfg": vars(cfg)}, run / "best.pt")
    log_f.close()
    print(f"done. best val loss {best:.4f}  ->  {run}/best.pt   日志 {run}/log.jsonl")


if __name__ == "__main__":
    main()
