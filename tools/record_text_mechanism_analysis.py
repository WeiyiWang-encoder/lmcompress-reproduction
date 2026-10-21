"""Analysis of the text level discrepancy: which mechanism, and what is eliminated.

A reader asked what "the paper's figure is pessimistic" actually means -- a convention
difference, a formula difference, or a code difference. That was an imprecise phrase on my
part and it needs replacing with an enumerated, testable analysis.

Four candidate mechanisms:

  A. convention  -- a different definition of what is counted (denominator, which
                    distribution codes which token, what is excluded)
  B. formula     -- the same convention evaluated by a different expression
  C. code        -- an implementation defect in one of the two code paths
  D. corpus      -- a different MeDAL subset, which genuinely compresses differently

Eliminations, each by measurement:

  A is eliminated for the shift: the reference's ``forward_next`` uses
    ``shifted_logits = logits[:, :-1]`` against ``shifted_input_ids = input_ids[:, 1:]``,
    which is the correct causal alignment, and an independent gather over the same window
    reproduces ``output.loss * (L-1)`` to 5.6e-08.
  A is eliminated for the denominator: container vs payload differ by 0.39 % and
    characters vs bytes by 0.14 % -- 5-20x too small to matter.
  D is eliminated by tools/test_corpus_subset.py: across all 100 MeDAL sample files the
    classical compressor rate spans only 30.26-32.99 % at the 8192-byte window, and our
    three files sit on the HARDER side (+0.31 pp against the corpus mean), which would
    push the model's rate UP, not down. A corpus change therefore cannot produce a
    2.73 pp gap in the required direction.

That leaves B and C, and it bounds them: both the reference harness (8.9253 %) and our
port (9.0410 %) report about **1.48x** the code length implied by the model's own loss on
identical weights, while a direct forward pass reports 1.00x. Two independently written
pipelines landing on the same inflation factor is the signature of a shared environmental
cause, not of two coincident bugs -- and the only thing that can raise cross-entropy by
1.3 bits/token without changing the formula is the model itself being in a different state.

The single most likely candidate is therefore **precision or checkpoint state**, not
arithmetic: the paper states its base model is loaded in 8-bit, this machine cannot do
8-bit at all, and a quantised 8B model evaluated on out-of-domain medical text can easily
lose 1.3 bits/token. This is cheap to settle on an A100 by running the same window in
bf16 / 8-bit / 4-bit and seeing which reproduces 3.87 bits/token.
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

AUDIT_SECTION = """### 5.9 What kind of difference is the text level? An enumerated analysis [M]

An earlier draft of this audit called the paper's text figure "pessimistic". That was too
loose to be useful, and a reader was right to ask whether the difference is one of
**convention**, **formula** or **code**. This section replaces the adjective with four
enumerated candidates and eliminates three of them by measurement.

| # | candidate | what it would mean |
|---|---|---|
| A | **convention** | a different definition of what is counted: the denominator, which distribution codes which token, what is excluded at window edges |
| B | **formula** | the same convention evaluated by a different expression (a mean where a sum is needed, a logarithm base, a mask applied at the wrong point) |
| C | **code** | an implementation defect in one of the two code paths |
| D | **corpus** | a different MeDAL subset, which genuinely compresses differently |

#### A is eliminated for the shift

The reference's `forward_next` uses

```python
shifted_logits     = logits[:, :-1, :]
shifted_input_ids  = input_ids[:, 1:]
```

which is the correct causal alignment -- position *j* scores token *j+1*. And an
independent gather over one window reproduces the model's own loss,
`output.loss * (L-1) = 10 516.776` bits against a computed `10 516.775`, a relative
difference of **5.6e-08**. So neither side is mis-shifted.

#### A is eliminated for the denominator

Container vs extracted payload differ by **0.39 %** (313 272 vs 312 053 bytes); payload
characters vs payload bytes differ by **0.14 %**. A 2.73 pp gap cannot come from either.

#### D is eliminated by measuring the corpus distribution

`tools/test_corpus_subset.py` runs brotli q11 and lzma 9e at the paper's 8192-byte window
over **all 100** CSVs in the sample:

| codec, window | corpus min | corpus mean | corpus max | our three | our mean |
|---|---|---|---|---|---|
| brotli, 8192 | 30.26 | 31.53 | 32.99 | 31.73 / 31.77 / 32.01 | **31.84** |
| brotli, inf | 26.96 | 28.47 | 29.56 | 28.61 / 28.78 / 29.02 | **28.80** |
| lzma, 8192 | 38.72 | 40.68 | 42.40 | 41.09 / 41.36 / 41.86 | **41.44** |
| lzma, inf | 29.55 | 31.25 | 32.41 | 31.39 / 31.51 / 32.15 | **31.68** |

The whole corpus spans only **2.7 pp** for a classical compressor, and our three files are
**harder** than the corpus mean by 0.31-0.76 pp, not easier. A harder subset would push the
model's rate **up**, whereas the gap to be explained has our rate **below** the paper's.
**Corpus choice is therefore eliminated as an explanation of the 2.73 pp gap.**

#### That leaves B and C, and bounds them

Both the reference harness (**8.9253 %**) and our port (**9.0410 %**) report roughly
**1.48x** the code length implied by the model's own loss on identical weights -- 3.87 and
3.80 bits/token against 2.5682 -- while a direct forward pass reports 1.00x.

Two independently written pipelines landing on the same inflation factor is the signature
of a **shared environmental cause**, not of two coincident bugs in two different codebases.
And the only thing that can raise cross-entropy by 1.3 bits/token *without changing the
formula* is the model being in a different state.

The leading candidate is therefore **precision**, not arithmetic: the paper states its base
model is loaded in **8-bit**, this machine cannot do 8-bit at all (bitsandbytes ships no
CUDA binary here), and a quantised 8B model on out-of-domain medical text plausibly loses
on the order of a bit per token. That single hypothesis explains the 1.48x without
requiring either implementation to contain a bug.

**The test that discriminates it is cheap on an A100**: one window, three precisions
(bf16 / 8-bit / 4-bit), and see which reproduces 3.87 bits/token. If 8-bit lands near 3.87,
the difference is a genuine precision effect and *both* the paper's figure and our port's
retracted figure are honest measurements of a quantised model; if 8-bit stays near 2.6, then
both harnesses share a computation defect and locating it becomes mandatory.

**What may not be claimed until that test is run**: neither "the paper is pessimistic" nor
"our port had a bug in the text path". What may be claimed is the elimination above --
shift, denominator and corpus are all excluded by measurement, which already narrows the
question to precision or shared computation.
"""


REPORT_SECTION = """### 5.9 文本量级差到底是什么性质：四选一的分析

我在前面把论文的文本数说成"偏悲观"，这个说法太含糊。你的追问是对的 ——
需要区分是**口径**、**计算**还是**代码**。下面把形容词替换成四个候选机制，并用测量排掉三个。

| # | 候选 | 含义 |
|---|---|---|
| A | **口径** | 计什么不同：分母、哪个分布编哪个 token、窗口边缘排除什么 |
| B | **计算** | 同一口径用不同表达式（该用和却用了均值、对数底、掩码位置） |
| C | **代码** | 某条路径的实现缺陷 |
| D | **语料** | MeDAL 的不同子集，本身压缩率就不同 |

#### A（口径）在位移上被排除

参考实现的 `forward_next` 用的是

```python
shifted_logits     = logits[:, :-1, :]
shifted_input_ids  = input_ids[:, 1:]
```

即位置 *j* 打分 token *j+1*，这是**正确的因果对齐**。而独立 gather 复现模型自身 loss：
算出 `10516.775` bits，`output.loss*(L-1)` 为 `10516.776`，相对差 **5.6e-08**。
**两侧都没有位移错误。**

#### A（口径）在分母上被排除

容器 vs 抽出载荷差 **0.39 %**（313 272 vs 312 053 字节）；载荷字符 vs 载荷字节差 **0.14 %**。
2.73 pp 的差距不可能来自这两个。

#### D（语料）被实测排除

`tools/test_corpus_subset.py` 在论文的 8192 字节窗口下，对样本里**全部 100 个** CSV 跑了 brotli 与 lzma：

| 编码器, 窗口 | 全库最小 | 全库均值 | 全库最大 | 我们三个 | 我们均值 |
|---|---|---|---|---|---|
| brotli, 8192 | 30.26 | 31.53 | 32.99 | 31.73 / 31.77 / 32.01 | **31.84** |
| brotli, 无界 | 26.96 | 28.47 | 29.56 | 28.61 / 28.78 / 29.02 | **28.80** |
| lzma, 8192 | 38.72 | 40.68 | 42.40 | 41.09 / 41.36 / 41.86 | **41.44** |
| lzma, 无界 | 29.55 | 31.25 | 32.41 | 31.39 / 31.51 / 32.15 | **31.68** |

**整个语料库对经典压缩器只跨 2.7 pp**，而我们的三个文件比全库均值**更难压** 0.31–0.76 pp。
更难压的子集会让模型码率**上升**，而要解释的差距是我们**低于**论文。
**所以语料选择不能解释这 2.73 pp。**

#### 剩下 B 与 C，而且可以把范围收窄

参考 harness（**8.9253 %**）与我们端口（**9.0410 %**）在**相同权重**上都报告约
**1.48 倍**于模型自身 loss 隐含的码长 —— 3.87 与 3.80 bits/token 对 **2.5682** ——
而直接前向是 1.00 倍。

**两条彼此独立编写的管线落在同一个膨胀倍数上**，这是"共享环境原因"的特征，
而不是两个不同代码库里恰好同时出现的两个 bug。而唯一能在**不改公式**的前提下
把交叉熵抬高 1.3 bits/token 的东西，就是**模型本身处于不同状态**。

因此首要候选是**精度**，不是算术：论文自述基座按 **8-bit** 加载，而本机**根本做不了 8-bit**
（bitsandbytes 无 CUDA 二进制）。一个量化后的 8B 模型在领域外的医学文本上，损失一个 bit/token 量级是合理的。
**这一个假设就能解释 1.48 倍，而不需要任何一方存在 bug。**

**在 A100 上判别它很便宜**：同一个窗口跑三种精度（bf16 / 8-bit / 4-bit），看哪个复现 3.87 bits/token。
若 8-bit 落在 3.87 附近，则差异是真实的精度效应，**论文的数与我们已撤回的端口数都是对量化模型的诚实测量**；
若 8-bit 仍在 2.6 附近，则两条管线共享一处计算缺陷，定位它就成为必须。

**在该测试完成前不能声称的**：既不能说"论文偏悲观"，也不能说"我们端口文本路径有 bug"。
可以声称的是上面的排除结论 —— 位移、分母、语料三者都由测量排除，
范围已经收窄到"精度或共享计算"。
"""


def main() -> int:
    a = io.open(AUDIT, encoding="utf-8").read()
    if "5.9 What kind of difference" not in a:
        anchor = "### 5.8 The training-amount ablation"
        if anchor in a:
            a = a.replace(anchor, AUDIT_SECTION + "\n" + anchor, 1)
            io.open(AUDIT, "w", encoding="utf-8", newline="\n").write(a)
            print("  ok  audit: added 5.9")
        else:
            print("  !! audit anchor missing")
    else:
        print("  --  audit already has 5.9")

    r = io.open(REPORT, encoding="utf-8").read()
    if "5.9 文本量级差" not in r:
        anchor = "### 5.8 【撤回】"
        if anchor in r:
            r = r.replace(anchor, REPORT_SECTION + "\n" + anchor, 1)
            io.open(REPORT, "w", encoding="utf-8", newline="\n").write(r)
            print("  ok  report: added 5.9")
        else:
            print("  !! report anchor missing")
    else:
        print("  --  report already has 5.9")
    return 0


if __name__ == "__main__":
    sys.exit(main())
