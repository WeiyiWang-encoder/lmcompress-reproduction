"""Record why the text arms were measured at context 4096, and correct a false excuse.

An external reviewer asked the obvious question: the report names context length as the
largest confound in the fine-tuning result, and the paper's own Supplementary Table 9
quantifies it at ~0.3 pp -- the same magnitude as the effect being reported. So why was
the experiment not run at 8192 in the first place?

The honest answer is that there was no good reason, and two plausible excuses are both
false:

  * "4096 was all the 12 GB card could hold."  FALSE. The paper reports peak GPU memory
    as 15.49 GB for **all three** context lengths -- 8192, 6144 and 4096 -- so context
    length does not change the footprint. Our port exposes `--context-window` with a
    default of 4096, and the text runs simply used the default.
  * "a longer context is slower, and the budget was tight."  FALSE, and backwards. The
    paper's own timings fall as context grows: 7323 s at 4096, 6855 s at 6144, 3072 s
    at 8192, for the same corpus.

What actually happened: the port was written with a 4096 default, the text arms were
measured with that default, and the confound was only recognised later when Table 9 was
read for a different reason. The cost of the omission is that the report cannot separate
the adapter effect from the context effect, and has to carry that caveat throughout.

This is recorded rather than quietly patched because a reader is entitled to know whether
a confound was an unavoidable constraint or an oversight. It was an oversight.
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

AUDIT_SECTION = """### 5.4 Why the text arms were measured at context 4096 [M]

A reviewer asked the obvious question: this audit names context length as the largest
confound in the fine-tuning result, the paper's own Supplementary Table 9 quantifies it
at about **0.3 pp** -- the same magnitude as the effect being reported -- so why was the
experiment not run at 8192 from the start?

**There was no good reason, and the two plausible excuses are both false.**

| excuse | verdict |
|---|---|
| "4096 was all the 12 GB card could hold" | **false.** The paper reports peak GPU memory of **15.49 GB for all three** context lengths (8192, 6144, 4096). Context length does not change the footprint. Our port exposes `--context-window`; its default is 4096, and the text runs used the default. |
| "a longer context is slower, and the budget was tight" | **false, and backwards.** The paper's own timings *fall* as context grows: **7323 s at 4096, 6855 s at 6144, 3072 s at 8192** for the same corpus. |

What actually happened: the port was written with a 4096 default, the text arms were
measured with that default, and the confound was only recognised later, when Table 9 was
read for an unrelated reason. The consequence is that this reproduction **cannot separate
the adapter effect from the context effect**, and must carry that caveat wherever the
penalty is quoted.

The fix is a paired run at 8192, and it is cheap: our three test files total only
**57 547 tokens**, so ten windows at 4096 become three at 8192 with the same token count.
The cost is hours, not days.

Recorded rather than quietly patched because a reader is entitled to know whether a
confound was an unavoidable constraint or an oversight. It was an oversight.
"""


REPORT_SECTION = """### 关于"为什么当初不在 8192 下测"

**没有正当理由。而且我原先可能给出的两种解释都是错的。**

| 可能的解释 | 实际情况 |
|---|---|
| "12 GB 显存只够 4096" | **错。** 论文报告 8192 / 6144 / 4096 三档的峰值显存**都是 15.49 GB** —— 上下文长度不改变显存占用。我们端口有 `--context-window` 参数，**默认值就是 4096**，文本实验直接用了默认值。 |
| "上下文越长越慢，时间预算不够" | **错，而且方向相反。** 论文自己的耗时随上下文变长而**下降**：4096 用 7323 s，6144 用 6855 s，**8192 只用 3072 s**（同一语料）。 |

实际情况是：端口写的时候把默认值定成 4096，文本双臂就用这个默认跑了；而上下文这个混淆项是在后来为了别的原因去读 Table 9 时才意识到的。

**代价是这份复现无法把"适配器效应"与"上下文效应"分开**，因此凡引用 +0.312 pp 都必须带上该限定。

**补救很便宜**：我们的三个测试文件合计只有 **57 547 tokens**，4096 下的 10 个窗口在 8192 下会变成 3 个窗口，**总 token 数不变**。成本是小时级，不是天级。

之所以记录而不是悄悄改掉，是因为读者有权知道一个混淆项究竟是"客观限制"还是"疏忽"。**这次是疏忽。**
"""


def main() -> int:
    a = io.open(AUDIT, encoding="utf-8").read()
    if "Why the text arms were measured at context 4096" not in a:
        anchor = "### 5.3"
        if anchor in a:
            a = a.replace(anchor, AUDIT_SECTION + "\n" + anchor, 1)
        else:
            a = a.rstrip() + "\n\n" + AUDIT_SECTION + "\n"
        io.open(AUDIT, "w", encoding="utf-8", newline="\n").write(a)
        print("  ok  audit: added section on the 4096 choice")
    else:
        print("  --  audit already records it")

    r = io.open(REPORT, encoding="utf-8").read()
    if "为什么当初不在 8192 下测" not in r:
        anchor = "### 6.3 那么\"差在哪里\"？"
        if anchor in r:
            r = r.replace(anchor, REPORT_SECTION + "\n" + anchor, 1)
            io.open(REPORT, "w", encoding="utf-8", newline="\n").write(r)
            print("  ok  report: added the explanation")
        else:
            print("  --  report anchor not found")
    else:
        print("  --  report already records it")
    return 0


if __name__ == "__main__":
    sys.exit(main())
