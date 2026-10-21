"""Wholesale replacement of the two withdrawn audio blocks (audit + report).

Line-level patching failed here: the blocks contain several mutually-referencing
sentences about "the current re-measured value", which cannot be fixed by swapping
individual numbers.  Both blocks are replaced in full by the verified position.
"""

from __future__ import annotations

import os
import re
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # pragma: no cover
    pass

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUDIT = os.path.join(HERE, "docs", "AUDIT_FOR_REVIEW.md")
REPORT = os.path.join(HERE, "docs", "REPORT_CN.md")

AUDIT_NEW = """**C6 — re-scoped; the earlier framing imported the withdrawn C9.**

What survives as a measurement: at every window length the model is more confident
about the byte **16 positions on** from a given slot than about the byte immediately
after it. Measured over sampled rows: mean probability **0.3052** at offset 16 versus
**0.0804** at offset 1, and ≤ 0.19 at every other offset in −2…19
(`tools/measure_row_target_offset.py`).

**This is a property of the coding scheme the checkpoint implements, not a defect.**
The reference pairs each row with exactly that byte, i.e. with the byte the model is
most confident about, and the scheme is causal: perturbing bytes strictly after a
target leaves its score unchanged (`max |Δ log-prob| = 0.000000e+00`), while the control
perturbing bytes *before* it moves the score by **9.44**
(`tools/test_audio_causality.py`). The arithmetic coder's real output is
**900 016 bits = 112 502 bytes**.

**Withdrawn.** Everything this document previously said about a "misalignment", about a
6.29× inflation, and the values **13.12 / 13.1233 / 14.1804 / 15.7844 bits/byte** is
**void**. All of them came from an index error in our own probe: a patch row carries
17 slot predictions, so the flat row for payload position `k` is
`17 * (k // 16) + (k % 16)`, and we used `16 + k`. See the C9 retraction above and
`docs/REVIEW_GUIDE_FOR_AGENT.md` §5.2.

**What remains open.** The released code plus weights report **1.9376 bpb** on the
bundled sample; the paper reports **18.26 %** (≈1.46 bpb) on LJSpeech. This reproduction
does **not** reconcile them and offers **no mechanism**. The most likely explanation is
that the released `weights-audio.pth` is a different training run, which we cannot test.
We do **not** claim the paper's figure is wrong.
"""

REPORT_NEW = """模型对**偏移 16 处**那个字节的预测（平均概率 **0.3052**）明显好于对**紧邻的下一字节**
（**0.0804**）；在 −2…19 的所有其他偏移上都不超过 0.19。参考实现正是把每一行与**它最自信的那个字节**
配对，因此它的配对是这个 checkpoint 的**自然配对**，且该方案**是因果的**：扰动目标之后的字节，
全词表 `max |Δ log-prob| = 0`；对照（扰动目标**之前**的字节）则变化 **9.44**。

**已核实**：其算术编码器确实输出了 **900 016 bit = 112 502 字节**，所以报出的数字是
**该方案下真实的码长**，不是打分假象。

> **已撤回**：本报告早期版本关于"打分错位"、6.29× 夸大，以及
> **13.12 / 13.1233 / 14.1804 / 15.7844 bits/byte** 的全部说法**一律作废**。
> 它们源自**我们自己探针的索引错误**：一行 patch 有 17 个 slot 预测，
> 所以 payload 位置 k 的扁平行号是 `17*(k//16) + (k%16)`，而我们误用了 `16 + k`。
> 详见 `docs/REVIEW_GUIDE_FOR_AGENT.md` §5.2。

**仍未解决的问题**：发布的代码 + 权重在 bundled sample 上得到 **1.9376 bpb**，
而论文在 LJSpeech 上报告 **18.26%**（≈1.46 bpb）。**本复现无法解释二者的差异，
也不给出任何机制解释。** 最可能的原因是发布的 `weights-audio.pth` 来自另一次训练，
但这需要作者提供配置才能验证。**我们不主张论文的数字是错的。**
"""


def replace_block(text: str, start_marker: str, end_marker: str, new: str) -> str | None:
    i = text.find(start_marker)
    if i < 0:
        return None
    j = text.find(end_marker, i)
    if j < 0:
        return None
    return text[:i] + new + text[j:]


def main() -> int:
    # ---- audit block ------------------------------------------------------- #
    text = open(AUDIT, encoding="utf-8").read()
    out = replace_block(
        text,
        "**C6 — re-scoped; the earlier framing imported the withdrawn C9.**",
        "**C7 ",
        AUDIT_NEW + "\n",
    )
    if out is None:
        # find the actual next claim heading
        m = re.search(r"\n\*\*C7", text)
        i = text.find("**C6 — re-scoped")
        if i > 0 and m:
            out = text[:i] + AUDIT_NEW + "\n" + text[m.start():]
    if out is None:
        print("!! could not locate the audit C6 block")
    else:
        open(AUDIT, "w", encoding="utf-8", newline="\n").write(out)
        print("  ok  audit C6 block replaced")

    # ---- report block ------------------------------------------------------ #
    text = open(REPORT, encoding="utf-8").read()
    i = text.find("模型对**偏移 16 处**")
    j = text.find("**仍然无法排除的两种解释**")
    if i < 0 or j < 0:
        print("!! could not locate the report audio block")
    else:
        text = text[:i] + REPORT_NEW + "\n" + text[j:]
        open(REPORT, "w", encoding="utf-8", newline="\n").write(text)
        print("  ok  report audio block replaced")
    return 0


if __name__ == "__main__":
    sys.exit(main())
