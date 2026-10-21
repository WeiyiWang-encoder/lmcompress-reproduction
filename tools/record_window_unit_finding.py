"""Record the window-unit discovery: our "context 4096" was never the paper's "4096".

Found while building the paired Colab notebook, and it changes how every text number in
this audit must be compared with the paper.

The reference implementation windows the text corpus as a **string**:

    def split(s, max_length):
        return [s[i : i + max_length] for i in range(0, len(s), max_length)]
    ...
    split(self.data[0], context_window_size)     # self.data[0] is text_str (a str)

so `--context_window 8192` means 8192 **characters**. Our port tokenises first and then
windows by **tokens**. On MeDAL one character is about 0.18 Llama-3 tokens, so the paper's
three settings correspond to roughly 739 / 1129 / 1476 tokens per window, while our
"context 4096" is 3836 tokens per window -- 2.6x more context than the paper's largest.
"""

from __future__ import annotations

import io
import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # pragma: no cover
    pass

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUDIT = os.path.join(HERE, "docs", "AUDIT_FOR_REVIEW.md")
REPORT = os.path.join(HERE, "docs", "REPORT_CN.md")

AUDIT_SECTION = """### 5.5 The two "context lengths" were never the same quantity [M]

Found while building the paired notebook, and it changes how every text number here may
be compared with the paper.

**The reference windows by characters.** `compress_with_bgpt_or_llama.py` builds the text
corpus as a string and splits it:

```python
def split(s, max_length):
    return [s[i : i + max_length] for i in range(0, len(s), max_length)]
...
split(self.data[0], context_window_size)      # self.data[0] is text_str, a str
```

`--context_window 8192` therefore means **8192 characters**, not 8192 tokens.

**Our port windows by tokens.** `uic.data.text_units` tokenises the whole text, and the
predictor chunks the resulting id list at `--context-window`. Measured on the same
corpus, the two axes are:

| setting | window unit | windows | mean tokens per window |
|---|---|---|---|
| paper 4096 | characters | 78 | **739** |
| paper 6144 | characters | 51 | **1129** |
| paper 8192 | characters | 39 | **1476** |
| our 4096 | tokens | 15 | **3836** |
| our 8192 | tokens | 9 | **6394** |

**Our "context 4096" carried 2.6x more context than the paper's largest setting.**

**What this invalidates and what it does not.**

* It does **not** invalidate the measurement: base 9.0410 % and +LoRA 9.3529 % are correct
  for what they actually are, a 4096-token window.
* It **invalidates the comparison to the paper's 9.36 % row**. That row is a 739-token
  context; ours was 3836. Comparing them was apples to oranges, and the direction of the
  error is knowable: more context compresses better, so our 9.0410 % should sit *below*
  the 739-token figure, and it does.
* It also **explains an otherwise suspicious coincidence**: our 9.0410 % landed near the
  paper's 8192 row (9.06 %) rather than its 4096 row (9.36 %), which looked like a good
  reproduction but was really a longer context happening to land in the same place.
* The adapter penalty (+0.312 pp at our settings) is unaffected as a measurement, but its
  comparability to the paper's LoRA claim is now doubly limited: different window unit and
  different effective context.

**A second discrepancy this reopens.** The published main text says the text context is
**1024**, while SI Table 9 reports 8192 / 6144 / 4096. Under character units a 1024-char
window is only ~190 tokens, which would compress far worse than 9.06 %; so "1024" cannot
be a character window consistent with Table 9. It is either a token count, a different
experiment, or an error. This audit cannot resolve it and does not guess, but it now has
the unit information needed to state the question precisely.

**The fix** is `colab/Text_Paired_Context_Colab.ipynb`, which measures **both axes** --
character windows (4096/6144/8192, directly comparable with Table 9) and token windows
(4096/8192, comparable with our port) -- crossed with both arms. Ten cells, and the
character axis carries its own independent check: the authors' unmodified harness at 8192
characters reported base **8.9253 %** / +LoRA **9.5154 %** on this corpus.
"""


REPORT_SECTION = """### 5.5 两个"上下文长度"根本不是同一个量

这是在写配对 notebook 时发现的，它改变了本文所有文本数字与论文的比较方式。

**参考实现按字符切窗。** 它把语料存成字符串，然后

```python
def split(s, max_length):
    return [s[i : i + max_length] for i in range(0, len(s), max_length)]
split(self.data[0], context_window_size)      # self.data[0] 是 text_str，str
```

所以 `--context_window 8192` 是 **8192 个字符**，不是 8192 个 token。

**我们端口按 token 切窗**（`text_units` 先分词，predictor 再按 `--context-window` 切 id 列表）。
同一语料上两条轴实测：

| 设置 | 窗口单位 | 窗口数 | 平均 token/窗口 |
|---|---|---|---|
| 论文 4096 | 字符 | 78 | **739** |
| 论文 6144 | 字符 | 51 | **1129** |
| 论文 8192 | 字符 | 39 | **1476** |
| 我们 4096 | token | 15 | **3836** |
| 我们 8192 | token | 9 | **6394** |

**我们的"上下文 4096"带的上下文比论文最大设置还多 2.6 倍。**

**这个发现作废了什么、没作废什么**

* **没作废**测量本身：base 9.0410 % 与 +LoRA 9.3529 % 对"4096 token 窗口"而言是正确的。
* **作废了对论文 9.36 % 那一行的比较**。那行是 739 token 的上下文，我们是 3836，比较是无效的。
  而且误差方向可知：上下文越长压得越好，所以我们的 9.0410 % 应当**低于** 739 token 的数，事实正是如此。
* **也解释了一个本来很可疑的巧合**：我们的 9.0410 % 落在论文 8192 行（9.06 %）附近而不是 4096 行（9.36 %），
  看着像复现得很好，其实是更长的上下文恰好落到了同一个位置。
* 适配器惩罚（+0.312 pp）作为一个测量不受影响，但它与论文 LoRA 结论的可比性现在有**双重限制**：
  窗口单位不同、有效上下文也不同。

**这也重新打开了一个矛盾**：正文说文本上下文是 **1024**，而 SI Table 9 报的是 8192 / 6144 / 4096。
按字符算，1024 字符窗口只有约 190 token，压缩率会远差于 9.06 %，
所以"1024"不可能是与 Table 9 自洽的字符窗口。它要么是 token 数、要么是另一个实验、要么是笔误。
本报告不猜，但现在有了说明这个问题所需要的单位信息。

**补救**是 `colab/Text_Paired_Context_Colab.ipynb`：**两条轴都测** ——
字符窗口（4096/6144/8192，可直接对 Table 9）与 token 窗口（4096/8192，可与我们的端口对照），
并与双臂交叉，共 10 个单元格。字符轴自带独立校验：作者的原始 harness 在 8192 字符下
给出 base **8.9253 %** / +LoRA **9.5154 %**。
"""


def main() -> int:
    a = io.open(AUDIT, encoding="utf-8").read()
    if "5.5 The two" not in a:
        anchor = "### 5.4 Why the text arms"
        if anchor in a:
            a = a.replace(anchor, AUDIT_SECTION + "\n" + anchor, 1)
        else:
            a = a.rstrip() + "\n\n" + AUDIT_SECTION + "\n"
        io.open(AUDIT, "w", encoding="utf-8", newline="\n").write(a)
        print("  ok  audit: added the window-unit section")
    else:
        print("  --  audit already records it")

    r = io.open(REPORT, encoding="utf-8").read()
    if "两个\"上下文长度\"根本不是同一个量" not in r:
        anchor = "### 关于\"为什么当初不在 8192 下测\""
        if anchor in r:
            r = r.replace(anchor, REPORT_SECTION + "\n" + anchor, 1)
        else:
            r = r.rstrip() + "\n\n" + REPORT_SECTION + "\n"
        io.open(REPORT, "w", encoding="utf-8", newline="\n").write(r)
        print("  ok  report: added the window-unit section")
    else:
        print("  --  report already records it")

    # The earlier note said ten 4096-windows become three at 8192; the measured
    # counts are 15 -> 9 across the three files.
    a = io.open(AUDIT, encoding="utf-8").read()
    old = ("so ten windows at 4096 become three at 8192 with the same token count")
    new = ("so the 15 windows at 4096 tokens become 9 at 8192 with the same token count")
    if old in a:
        a = a.replace(old, new, 1)
        io.open(AUDIT, "w", encoding="utf-8", newline="\n").write(a)
        print("  ok  audit: corrected the window counts in 5.4")
    return 0


if __name__ == "__main__":
    sys.exit(main())
