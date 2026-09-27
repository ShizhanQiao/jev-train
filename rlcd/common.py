"""RLCD 数据生成公共部件：豆包 Responses API、并发请求、断点续跑。"""
from __future__ import annotations

import asyncio
import json
import os
import random
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import httpx

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "合成数据"
RAW_DIR = OUT_DIR / "_raw"
API_URL = "https://ark.cn-beijing.volces.com/api/v3/responses"
MODEL = "doubao-seed-2-1-lite-260915"


def load_key() -> str:
    """从 .env 读 key。只返回值，绝不打印。"""
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k = k.strip()
            v = v.strip().strip("'\"")
            if k.lower() in ("doubao_api", "doubao_api_key", "ark_api_key"):
                return v
    v = os.environ.get("ARK_API_KEY", "") or os.environ.get("DOUBAO_API_KEY", "")
    if not v:
        raise SystemExit("找不到豆包 key：.env 里应有 doubao_api=...")
    return v


def build_payload(system: str, user: str, *, thinking: bool = False,
                  json_mode: bool = False, max_tokens: int = 2048,
                  temperature: float = 1.0) -> dict[str, Any]:
    text = system + "\n\n" + user
    return {"model": MODEL, "input": [{"role": "user", "content": [{"type": "input_text", "text": text}]}],
            "thinking": {"type": "enabled" if thinking else "disabled"},
            "stream": False, "max_output_tokens": max_tokens, "temperature": temperature}


class Runner:
    """固定并发 + 指数退避重试 + 边跑边落盘的任务池。"""

    def __init__(self, key: str, concurrency: int = 256, timeout: float = 180.0,
                 max_retries: int = 5):
        self.key = key
        self.concurrency = concurrency
        self.timeout = timeout
        self.max_retries = max_retries
        self.stats = {"ok": 0, "fail": 0, "retry": 0, "prompt_tok": 0, "completion_tok": 0}
        self._t0 = time.time()
        self.fatal: str | None = None   # 余额耗尽等不可恢复错误，置位后剩余请求直接跳过

    async def _one(self, client: httpx.AsyncClient, sem: asyncio.Semaphore,
                   payload: dict[str, Any]) -> tuple[str | None, dict[str, Any]]:
        last = ""
        async with sem:
            if self.fatal:                      # 熔断：别再往一个已经没钱的账号上砸请求
                return None, {"error": self.fatal, "skipped": True}
            for attempt in range(self.max_retries):
                try:
                    r = await client.post(API_URL, json=payload)
                    if r.status_code == 200:
                        d = r.json(); u = d.get("usage") or {}
                        self.stats["prompt_tok"] += u.get("input_tokens", 0)
                        self.stats["completion_tok"] += u.get("output_tokens", 0)
                        content = ""
                        for item in d.get("output", []):
                            for part in item.get("content", []):
                                if part.get("type") in ("output_text", "text"):
                                    content += part.get("text", "")
                        return content, {"usage": u, "finish_reason": d.get("status")}
                    last = f"HTTP {r.status_code}: {r.text[:300]}"
                    if r.status_code in (401, 402):
                        # 402 = 余额不足，401 = key 无效。整批都救不回来，直接熔断。
                        if not self.fatal:
                            self.fatal = last
                            print(f"\n  !! 熔断（{r.status_code}）：{r.text[:160]}\n"
                                  f"  !! 剩余请求全部跳过，已完成的部分已落盘，充值后重跑即可续。",
                                  flush=True)
                        break
                    if r.status_code in (400, 404):
                        break  # 单条不可重试
                except Exception as e:  # noqa: BLE001
                    last = f"{type(e).__name__}: {e}"
                self.stats["retry"] += 1
                await asyncio.sleep(min(2 ** attempt, 20) * (0.5 + random.random()))
        return None, {"error": last}

    async def run(self, tasks: Sequence[tuple[str, dict[str, Any]]],
                  on_result: Callable[[str, str | None, dict[str, Any]], None],
                  label: str = "") -> None:
        """tasks: [(task_id, payload)]。on_result 在每条完成时同步调用。"""
        sem = asyncio.Semaphore(self.concurrency)
        limits = httpx.Limits(max_connections=self.concurrency + 16,
                              max_keepalive_connections=self.concurrency + 16)
        headers = {"Authorization": f"Bearer {self.key}", "Content-Type": "application/json"}
        total = len(tasks)
        done = 0
        async with httpx.AsyncClient(timeout=self.timeout, limits=limits, headers=headers) as client:
            coros = [self._one(client, sem, p) for _, p in tasks]
            for fut in asyncio.as_completed(
                    [self._tag(tid, c) for (tid, _), c in zip(tasks, coros)]):
                tid, content, meta = await fut
                if content is None:
                    self.stats["fail"] += 1
                else:
                    self.stats["ok"] += 1
                # 单条脏数据不能掀翻整批：解析/校验异常一律记失败继续跑
                try:
                    on_result(tid, content, meta)
                except Exception as e:  # noqa: BLE001
                    self.stats["ok"] -= (content is not None)
                    self.stats["fail"] += 1
                    print(f"  [{label}] {tid} on_result 异常: {type(e).__name__}: {e}", flush=True)
                done += 1
                if done % max(1, total // 100) == 0 or done == total:
                    el = time.time() - self._t0
                    print(f"  [{label}] {done}/{total}  ok={self.stats['ok']} "
                          f"fail={self.stats['fail']} retry={self.stats['retry']} "
                          f"{el:.0f}s ({done/max(el,1e-9):.1f}/s)", flush=True)

    @staticmethod
    async def _tag(tid: str, coro):
        c, m = await coro
        return tid, c, m


def parse_json(text: str) -> Any:
    """容忍 ```json 围栏和前后噪声。"""
    t = (text or "").strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[1] if "\n" in t else t
        t = t.rsplit("```", 1)[0]
    t = t.strip()
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        pass
    for op, cl in (("{", "}"), ("[", "]")):
        i, j = t.find(op), t.rfind(cl)
        if i != -1 and j > i:
            try:
                return json.loads(t[i:j + 1])
            except json.JSONDecodeError:
                continue
    raise ValueError("无法解析 JSON")


def read_done(path: Path, key: str = "id") -> set[str]:
    """断点续跑：读已产出的 id 集合。"""
    if not path.exists():
        return set()
    s = set()
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                s.add(str(json.loads(line)[key]))
            except Exception:  # noqa: BLE001
                continue
    return s


class JsonlWriter:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.f = path.open("a", encoding="utf-8")

    def write(self, obj: Any) -> None:
        self.f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        self.f.flush()

    def close(self) -> None:
        self.f.close()
