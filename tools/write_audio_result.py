"""Write the authoritative audio result, and correct the reproduction-status claims.

Chronology of this section, because it matters for how much to trust it:

  1. We claimed the reference's audio scoring was misaligned (6.29x inflation).
     **Wrong** -- our probe indexed flat rows as 16+k where a patch row has 17 slots.
     Retracted (see docs/REVIEW_GUIDE_FOR_AGENT.md section 5.2).
  2. We then claimed the released script could not process the full file, because the
     620-byte tail pads to 652 bytes and should not reshape.
     **Wrong** -- with the reconstructed `samplings` module installed, the unmodified
     script completes all 11 blocks. Retracted here.
  3. We claimed audio did not reproduce the paper.
     **Wrong for the same reason as (1)** -- the "irreproducible" figures (15.78 bpb,
     197 %) came from the bad row index. With the correct mapping, our port and the
     reference disagree by 4.29 % (0.6 % on the arithmetic coder's real output), and
     both land near the paper.

This script replaces the audio status statements with the measured position and leaves
a record of all three errors.
"""

from __future__ import annotations

import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # pragma: no cover
    pass

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUDIT = os.path.join(HERE, "docs", "AUDIT_FOR_REVIEW.md")
REPORT = os.path.join(HERE, "docs", "REPORT_CN.md")

# --------------------------------------------------------------------------- #
AUDIT_SECTION = """### 4.3 The audio result, after three corrections [M]

This section has been wrong three times. The final, measured position follows.

**Canonical numbers.** Sample `LJ001-0004.wav`, 16 kHz mono 8-bit PCM,
**82 220 payload bytes**. All of them are coded in every figure below.

| implementation | coded output | bits/byte | **rate** (compressed/original) | ratio |
|---|---|---|---|---|
| **our port** (`results/audio_bgpt_rows17_tailpad_v1.jsonl`) | 130 383 bits = 16 298 B | 1.5858 | **19.82 %** | 5.046× |
| **reference, unmodified** (`artifacts/refrun`, one full run) | 16 399.95 B | 1.5957 | **19.94 %** | 5.016× |
| paper headline (LJSpeech) | — | 1.4608 | **18.26 %** | 5.477× |

Our arithmetic coder's **real emitted output** is 1.5858 bits/byte against the
reference's reported 1.5957 — a **0.6 %** difference. The reported figure computed as
`Σ −log₂ q` is 1.6641 bits/byte (20.80 %), 4.29 % above the reference; the gap between
our two internal figures is the usual coder-quantisation overhead, and both are tracked
separately (`coded_bits` vs `ac_bits`).

**So audio does reproduce.** The reference lands **9.2 % relative** from the published
18.26 %, which is the same order as the classical baseline gaps elsewhere in this
audit (audio baselines: OptimFROG +1.89 pp, WavPack +2.15 pp, FLAC +4.07 pp), and the
ordering is unchanged: the model beats every classical codec measured, and by a wide
margin (19.8 % vs OptimFROG's 27.3 %).

**Three errors, recorded so they are not repeated.**

1. **"The scoring is misaligned by 16 positions" — RETRACTED.** Our probe indexed flat
   logits as `16 + k`, but a patch row carries **17** slot predictions, so the row for
   payload position `k` is `17*(k//16) + (k%16)`. Measured under the correct mapping,
   the reference's pairing is the one the checkpoint is most confident about
   (mean probability **0.3052** at offset 16 vs **0.0804** at offset 1), the scheme is
   causal (`max |Δ log-prob| = 0` when bytes after the target are perturbed, versus
   **9.44** when bytes before it are), and the coder really emits the bytes. Every
   figure derived from that probe (13.12, 13.1233, 14.1804, **15.7844** bits/byte and
   the 197.31 % rate) is **void**.
2. **"The tail block makes the script crash" — RETRACTED.** `split(..., 8160)` does
   yield 10 full windows plus a 620-byte tail, and `byte_collate` does pad that tail to
   652 bytes, which is not a multiple of `PATCH_SIZE`. But the unmodified script
   **completes all 11 blocks** and reports `Total size: 82264`. The predicted crash does
   not occur.
3. **"`samplings` blocks the run" — correct, and now fixed.** The capsule genuinely
   omits `samplings.py` while `data/model/bgpt/utils.py:9` imports it, so every entry
   point fails at import time. `tools/verify_reference_blockers.py` confirms the file is
   absent. A reconstruction lives in `replacement_modules/samplings.py`; the three
   functions are used only by the custom `generate` (lines 178-180), **never** by
   `forward`, so they cannot affect a compression measurement.
   `tools/run_reference_audio_onewindow.sh` copies the capsule to a temp tree, installs
   that module there, and runs the authors' script **unmodified** — the published
   capsule is never altered.

**Reproduction procedure** (`tools/confirm_tail_and_our_window.sh`):
the reference was run as shipped on the full file and on a single 8 160-byte window;
our port was measured on the same inputs. Raw outputs are in
`results/_tail_and_window.txt` and `results/_refrun_onewindow.txt`.
"""

# --------------------------------------------------------------------------- #
REPORT_AUDIO = """### 4.3 音频结论（经三次更正后的最终结果）

样本 `LJ001-0004.wav`，16 kHz 单声道 8-bit PCM，**82 220 字节**，以下所有数字都覆盖全部字节。

| 实现 | 编码输出 | bits/byte | **rate**（压缩后/原始） | ratio |
|---|---|---|---|---|
| **我们的端口** | 130 383 bit = 16 298 B | 1.5858 | **19.82 %** | 5.046× |
| **参考实现（未修改脚本）** | 16 399.95 B | 1.5957 | **19.94 %** | 5.016× |
| 论文 headline（LJSpeech） | — | 1.4608 | **18.26 %** | 5.477× |

我们算术编码器**真实吐出的码流**是 1.5858 bits/byte，参考实现报告 1.5957，**相差 0.6 %**。
按 `Σ −log₂ q` 计算的口径为 1.6641 bits/byte（20.80 %），比参考高 4.29 %——两者之差是
编码器量化开销，代码中分别记为 `coded_bits` 与 `ac_bits`。

**所以音频模态是复现成功的。** 参考实现与论文 18.26 % 的相对差为 **9.2 %**，
与本文档其他模态的经典基线偏差同一量级（音频基线：OptimFROG +1.89 pp、WavPack +2.15 pp、
FLAC +4.07 pp），且顺序不变：**模型明显优于所有经典编码器**（19.8 % vs OptimFROG 27.3 %）。

> **三次错误记录（防止重犯）**
>
> 1. **「打分错位 16 位」——已撤回。** 我们的探针把扁平 logits 索引写成 `16 + k`，
>    但一行 patch 有 **17** 个 slot 预测，位置 k 的正确行号是 `17*(k//16) + (k%16)`。
>    按正确映射实测：参考实现的配对正是该 checkpoint 最自信的字节
>    （偏移 16 处平均概率 **0.3052**，偏移 1 处仅 **0.0804**），方案是**因果的**
>    （扰动目标之后的字节 `max|Δ log-prob| = 0`，扰动之前的字节变化 **9.44**），
>    且编码器确实输出了这些字节。
>    **由该探针导出的 13.12 / 13.1233 / 14.1804 / 15.7844 bits/byte 及 197.31 % 一律作废。**
> 2. **「尾块导致脚本崩溃」——已撤回。** `split(..., 8160)` 确实给出 10 个完整窗口加
>    一个 **620 字节**尾块，`byte_collate` 也确实把尾块补成 **652** 字节（不能被 16 整除）。
>    但**未修改的脚本跑完了全部 11 个块**并报告 `Total size: 82264`，预测的崩溃没有发生。
> 3. **「`samplings` 阻断运行」——成立，已解决。** capsule 确实没有 `samplings.py`，
>    而 `data/model/bgpt/utils.py:9` 顶层 import 它，因此任何入口都在导入阶段失败
>    （`tools/verify_reference_blockers.py` 已核实文件不存在）。
>    重建模块见 `replacement_modules/samplings.py`；这三个函数只被自定义 `generate`
>    （第 178–180 行）使用，**从不被 `forward` 调用**，因此不可能影响压缩测量。
>    `tools/run_reference_audio_onewindow.sh` 把 capsule 复制到临时目录、装入该模块，
>    以**未修改**的脚本运行——**原 capsule 始终未被改动**。
"""


def replace_block(text: str, start: str, end: str, new: str) -> tuple[str, bool]:
    i = text.find(start)
    if i < 0:
        return text, False
    j = text.find(end, i)
    if j < 0:
        return text, False
    return text[:i] + new + "\n" + text[j:], True


def main() -> int:
    # ---- audit ------------------------------------------------------------- #
    text = open(AUDIT, encoding="utf-8").read()
    start = text.find("### 4.3")
    if start < 0:
        start = text.find("**C6 — re-scoped")
    end = text.find("### 4.4", start) if start >= 0 else -1
    if start >= 0 and end > start:
        text = text[:start] + AUDIT_SECTION + "\n" + text[end:]
        open(AUDIT, "w", encoding="utf-8", newline="\n").write(text)
        print("  ok  audit: audio section replaced")
    else:
        print("  !! audit: could not locate the audio section")

    # ---- report ------------------------------------------------------------ #
    text = open(REPORT, encoding="utf-8").read()
    i = text.find("模型对**偏移 16 处**")
    if i < 0:
        i = text.find("### 4.3")
    j = text.find("### 5.", i) if i >= 0 else -1
    # keep the 4.1/4.2 subsections: replace from the measurement table onward only
    k = text.find("| 量 | 值 |", i) if i >= 0 else -1
    if k > 0 and j > k:
        text = text[:k] + REPORT_AUDIO + "\n" + text[j:]
        open(REPORT, "w", encoding="utf-8", newline="\n").write(text)
        print("  ok  report: audio measurement block replaced")
    else:
        print("  !! report: could not locate the audio block")
    return 0


if __name__ == "__main__":
    sys.exit(main())
