"""Propagate the C9 retraction into the Chinese report and the review guide.

The retracted claim appeared in three places in REPORT_CN (the one-line summary, the
audio section, and the status table) and throughout REVIEW_GUIDE_FOR_AGENT.  Leaving
any of them would mean the thesis cites a withdrawn finding.
"""

from __future__ import annotations

import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # pragma: no cover
    pass

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORT = os.path.join(HERE, "docs", "REPORT_CN.md")
AUDIT = os.path.join(HERE, "docs", "AUDIT_FOR_REVIEW.md")
GUIDE = os.path.join(HERE, "docs", "REVIEW_GUIDE_FOR_AGENT.md")

RETRACT_NOTE = """> **重要勘误（音频结论已撤回）**：早期版本声称参考实现的音频打分存在"错位"，
> 导致其压缩率被夸大 6.29×。**该结论是错的，已整体撤回。**
> 实测表明：参考实现把每一行预测与它**最自信的那个字节**配对
> （偏移 16 处平均概率 **0.3052**，而紧邻的下一字节只有 **0.0804**），
> 且该方案**是因果的**（扰动目标之后的字节，全词表 `max |Δ log-prob| = 0`；
> 对照扰动之前的字节则变化 9.44），其算术编码器确实输出了 **900 016 bit = 112 502 字节**。
> **错误根源在我们自己**：一行 patch 有 **17** 个 slot 预测，所以 payload 位置 k 对应的
> 扁平行号是 `17*(k//16) + (k%16)`，而我们早期探针误用了 `16 + k`，
> 于是把每行拿去和"约 p 个位置之外"的字节打分，凭空造出了错位。
> 因此本报告中**所有音频的"诚实 next-byte"数字（13.12 / 14.18 / 15.78 bpb）一律作废**。
> 音频唯一仍然成立的结论是：**用作者自己的方案，发布的权重在发布样本上得到 1.9376 bpb，
> 而复现不出论文 headline 的 18.26%**——原因未知，本文不给出机制解释。"""


def main() -> int:
    # ---- 1. REPORT_CN: the summary line ---------------------------------- #
    text = open(REPORT, encoding="utf-8").read()
    edits = [
        (
            "**音频**：发布的 bGPT-audio checkpoint + reference scoring **无法复现论文 headline 的 18.26%**，\n"
            "且直接测量表明该 checkpoint 在 bundled sample 上是在**重建当前字节**而非预测下一个字节。",
            "**音频**：发布的 bGPT-audio checkpoint + reference scoring **无法复现论文 headline 的 18.26%**\n"
            "（1.9376 bpb vs 论文 1.46 bpb）。**注意：早期版本曾把原因归为「打分错位」，"
            "该结论已撤回——详见 §4 开头的勘误。原因目前未知。**",
        ),
        (
            "| 音频 | bGPT-audio | **否** | 论文 headline 18.26%（= 1.46 bpb）；参考代码报 **1.9376 bpb**；"
            "**正确 next-byte = 15.7844 bpb**。机制已定量：错位配对 **2.2561** vs 正确 **14.1804 bpb**，"
            "**6.29× 全部来自索引错位** |",
            "| 音频 | bGPT-audio | **否** | 论文 headline 18.26%（= 1.46 bpb）；参考代码报 **1.9376 bpb**。"
            "**早期版本宣称的「打分错位 / 6.29×」已撤回**（那是我们自己的行索引错误）；"
            "不可复现的原因**未知** |",
        ),
    ]
    applied = 0
    for old, new in edits:
        if old in text:
            text = text.replace(old, new, 1)
            applied += 1
            print(f"  ok  REPORT_CN: {old[:52]!r}")
        else:
            print(f"  !! REPORT_CN no match: {old[:52]!r}")

    # insert the retraction notice at the head of section 4
    anchor = "## 4. 音频模态：发布的 checkpoint + reference scoring 无法复现论文 headline\n"
    if anchor in text and "重要勘误（音频结论已撤回）" not in text:
        text = text.replace(anchor, anchor + "\n" + RETRACT_NOTE + "\n", 1)
        applied += 1
        print("  ok  REPORT_CN: retraction notice inserted at head of section 4")

    with open(REPORT, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)

    # ---- 2. AUDIT: C6 wording (different phrasing than expected) --------- #
    atext = open(AUDIT, encoding="utf-8").read()
    old_c6 = ("**C6 [M].** At every window length the checkpoint matches the *current* byte")
    if old_c6 in atext:
        atext = atext.replace(
            old_c6,
            "**C6 — re-scoped; the earlier framing imported the withdrawn C9.** "
            "At every window length the checkpoint matches the byte at offset 16",
            1)
        with open(AUDIT, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(atext)
        applied += 1
        print("  ok  AUDIT: C6 re-scoped")

    print(f"\n{applied} edit(s) applied")
    print("\nREVIEW_GUIDE_FOR_AGENT.md still describes C9 as a live claim and must be "
          "rewritten by hand: its whole section 5 / Claim 2 is now a retraction.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
