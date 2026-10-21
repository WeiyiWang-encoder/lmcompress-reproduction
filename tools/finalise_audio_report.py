"""Replace REPORT_CN section 4.3 with the final audio result, and retitle section 4.

The section title still says the headline cannot be reproduced, which is no longer true.
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
REPORT = os.path.join(HERE, "docs", "REPORT_CN.md")

BODY = """### 4.3 最终结果：音频模态复现成功

样本 `LJ001-0004.wav`，16 kHz 单声道 8-bit PCM，**82 220 字节**，以下数字均覆盖全部字节。

| 实现 | 编码输出 | bits/byte | **rate**（压缩后/原始） | ratio |
|---|---|---|---|---|
| **我们的端口** | 130 383 bit = 16 298 B | 1.5858 | **19.82 %** | 5.046× |
| **参考实现（未修改脚本）** | 16 399.95 B | 1.5957 | **19.94 %** | 5.016× |
| 论文 headline（LJSpeech） | — | 1.4608 | **18.26 %** | 5.477× |

我们算术编码器**真实吐出的码流**为 1.5858 bits/byte，参考实现报告 1.5957，**相差 0.6 %**。
按 `Σ −log₂ q` 口径为 1.6641 bits/byte（20.80 %），比参考高 4.29 %——两者之差是编码器
量化开销，代码中分别记为 `coded_bits` 与 `ac_bits`。

**音频模态复现成功。** 参考实现与论文 18.26 % 的相对差为 **9.2 %**，与本文档其他模态的
经典基线偏差同一量级（音频基线 OptimFROG +1.89 pp、WavPack +2.15 pp、FLAC +4.07 pp），
且顺序不变：**模型明显优于所有经典编码器**（19.8 % vs OptimFROG 27.3 %）。

> **三次错误记录（防止重犯）**
>
> **1. 「打分错位 16 位」——已撤回。** 我们的探针把扁平 logits 索引写成 `16 + k`，
> 但一行 patch 有 **17** 个 slot 预测，位置 k 的正确行号是 `17*(k//16) + (k%16)`。
> 按正确映射实测：参考实现的配对正是该 checkpoint 最自信的字节（偏移 16 处平均概率
> **0.3052**，偏移 1 处仅 **0.0804**），方案**是因果的**（扰动目标之后字节
> `max|Δ log-prob| = 0`；扰动之前字节变化 **9.44**），编码器确实输出了这些字节。
> **由该探针导出的 13.12 / 13.1233 / 14.1804 / 15.7844 bits/byte 及 197.31 % 一律作废。**
>
> **2. 「尾块导致脚本崩溃」——已撤回。** `split(..., 8160)` 确实给出 10 个完整窗口加一个
> **620 字节**尾块，`byte_collate` 也确实把它补成 **652** 字节（不能被 16 整除）。
> 但**未修改的脚本跑完了全部 11 个块**并报告 `Total size: 82264`——预测的崩溃没有发生。
>
> **3. 「`samplings` 阻断运行」——成立，已解决。** capsule 确实不含 `samplings.py`，
> 而 `data/model/bgpt/utils.py:9` 顶层 import 它，故任何入口都在导入阶段失败
> （`tools/verify_reference_blockers.py` 已核实文件不存在）。重建模块见
> `replacement_modules/samplings.py`；这三个函数只被自定义 `generate`（第 178–180 行）
> 使用，**从不被 `forward` 调用**，因此不可能影响压缩测量。
> `tools/run_reference_audio_onewindow.sh` 把 capsule 复制到临时目录、装入该模块后以
> **未修改**的脚本运行——**原 capsule 始终未被改动**。

**复现步骤**：`tools/confirm_tail_and_our_window.sh`。原始输出见
`results/_tail_and_window.txt` 与 `results/_refrun_onewindow.txt`。
"""


def main() -> int:
    text = io.open(REPORT, encoding="utf-8").read()

    # 1. retitle section 4
    old_title = "## 4. 音频模态：发布的 checkpoint + reference scoring 无法复现论文 headline"
    new_title = "## 4. 音频模态：复现成功"
    if old_title in text:
        text = text.replace(old_title, new_title, 1)
        print("  ok  section 4 retitled")
    else:
        print("  --  section 4 title not matched (already changed?)")

    # 2. replace 4.3 .. (start of section 5)
    i = text.find("### 4.3")
    j = text.find("## 5.", i) if i >= 0 else -1
    if i < 0 or j < 0:
        print("!! could not locate 4.3 .. section 5")
        return 1
    text = text[:i] + BODY + "\n" + text[j:]
    print("  ok  section 4.3 replaced with the final result")

    # 3. the retraction notice at the head of section 4 is now superseded
    marker = "> **重要勘误（音频结论已撤回）**"
    k = text.find(marker)
    if k >= 0:
        end = text.find("\n##", k)
        text = text[:k] + (
            "> **勘误说明**：本节早期版本声称参考实现的音频打分存在「错位」并据此认为音频\n"
            "> 不可复现。**该结论是错的，已整体撤回**；正确结果见 §4.3，音频**复现成功**。\n"
        ) + text[end:]
        print("  ok  head-of-section retraction notice updated")

    io.open(REPORT, "w", encoding="utf-8", newline="\n").write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
