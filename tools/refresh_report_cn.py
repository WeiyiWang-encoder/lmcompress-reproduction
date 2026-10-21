"""Rewrite the stale sections of REPORT_CN.md to match the canonical facts.

Systematic upgrade rather than piecemeal number substitution, because three review
rounds established that cross-document residue is this project's dominant error
mode.  Each section is replaced wholesale with text derived from
``AUDIT_FOR_REVIEW.md``, then the whole file is checked for forbidden tokens.
"""

from __future__ import annotations

import argparse
import os
import re

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORT = os.path.join(HERE, "docs", "REPORT_CN.md")

# Strings that must NOT survive anywhere in the refreshed report (outside of
# explicit retraction notes).  Checked at the end.
FORBIDDEN = [
    "9.033%", "9.346%", "0.313",
    "0.017%", "67.43", "330 字节",
    "1912 倍", "1 957 388", "5 872 164", "原始 RGB 像素",
    "13.1 bits/byte", "0.51×",
    "四种模态的经典基线也复现成功",
]

SECTIONS = {
    # ---- header summary -------------------------------------------------- #
    "SUMMARY": """**文本模态复现成功且高度吻合**（原始模型 9.041% vs 论文 LMCompress\\* 的 9.08%，
差 **0.04 pp**；微调 9.353% vs 论文 9.06%）；
**已测三种模态（文本/音频/图像）的经典基线也复现成功**（音频 OptimFROG/WavPack/FLAC 的
顺序与量级都吻合；视频**未测**）；
但**发布的 bGPT-audio checkpoint + reference scoring 无法复现论文 headline 的 18.26%**，
且证据表明它在 bundled sample 上给出的 1.94 bits/byte 不是"预测下一个字节"的码长（§4）。
**图像侧**：官方路径的 accounting 口径与论文量级接近（15.51% vs 14.62%），
但该路径**对源图像不是无损**——已用穷举证明（§5）。

**对进阶任务（文本微调）的关键结论**：论文在 MeDAL 上的微调增益本来就只有 0.02 pp
（9.08 → 9.06），而我实测微调后**反而变差 0.312 pp**（9.041% → 9.353%，三个文件无一例外）。
原因不在训练量不足，而在**发布的 checkpoint 过拟合了**——`trainer_state.json` 显示
eval loss 在第 1020 步（约 2 epoch）最优 2.4708，到第 5120 步升到 2.6728。
**但必须同时声明一个同量级的混淆项**：论文自己的 Supplementary Table 9 显示 ctx 8192 / 6144 / 4096
在 MeDAL 上是 9.06 / 9.16 / 9.36%，跨度约 **0.3 pp —— 和我测到的惩罚同量级**。
所以这 0.312 pp 有多少来自适配器、多少来自我用的较短上下文（4096 vs 8192），
**目前无法分离**，必须在 Colab 上以 ctx 8192 重跑双臂才能定论（§7）。
因此"只要提高训练量效果即可提升"这一写法**与论文自己的训练日志相矛盾**，报告里不能这么写。""",

    # ---- §3 text table --------------------------------------------------- #
    "TEXT_TABLE": """| 方法 | 实测 rate | 实测 ratio | 论文 rate | 差 |
|---|---|---|---|---|
| LMCompress\\*（原始 Llama-3-8B，不微调） | **9.041 %** | 11.061× | 9.08 % | **−0.04 pp** |
| LMCompress（+ capsule 提供的 LoRA） | **9.353 %** | 10.692× | 9.06 % | **+0.29 pp** |
| brotli q11 | 28.805 %（容器口径）/ **28.918 %**（payload 口径） | 3.46× | 27.93 % | +0.99 |
| lzma preset 9e（等价 7z -mx9） | 31.726 % / **31.850 %** | 3.14× | 30.25 % | +1.60 |
| zstd --ultra -22 | 31.994 % / **32.119 %** | 3.11× | 30.45 % | +1.67 |
| zlib 9 | 35.018 % / **35.155 %** | 2.84× | — | — |

**关于两个分母**：经典基线压的是整份 CSV 容器（313 272 字节），而模型编的是抽出来的
`TEXT` 载荷（312 053 字节，`pandas` 去掉了引号），差 0.13%。**真正 like-for-like 的对比
应当用 payload 口径那一列**（粗体），即 brotli **28.918%** 对模型 **9.041%**，
模型好 **3.20×**。
""",

    # ---- §4 audio -------------------------------------------------------- #
    "AUDIO": """## 4. 音频模态：发布的 checkpoint + reference scoring 无法复现论文 headline

**先把三件事分清楚**（早期版本把它们混为一谈，是错的）：

| 量 | 值 | 它是什么 |
|---|---|---|
| 论文 headline（LJSpeech） | 18.26 % | 已发表结果；**我们不主张它是错的** |
| reference code 在 bundled sample 上报告的 | 1.9376 bpb ≈ 4.13× | 发布代码*报告*的数字 |
| 诚实 next-byte 打分，同一 sample | **15.7844 bpb = 197.31 % = 0.5068×** | checkpoint 实际达到的水平 |

样本 `LJ001-0004.wav`：16 kHz 单声道 **8-bit PCM**，82 220 字节。

### 4.1 order-1 熵的对比 —— 只是旁证，不是证明

| 量 | 数值 |
|---|---|
| order-0 熵 | 4.8337 bits/byte |
| order-1 熵 | 3.1748 bits/byte |
| reference 报告 | **1.9376 bits/byte** |

**审阅意见已采纳**：早期版本把 order-1 经验熵称为"valid lower bound"、把差异称为
"proof"，两者都不成立。真正的下界是**给定模型上下文后的信源熵**，一个长上下文的好模型
完全可以低于 order-1 经验估计；而且 82 KB 单文件的经验熵本身带噪声。
所以这个对比**只是提示性的旁证**，用来指引我们去找打分缺陷。
**承重证据是 §4.3 的直接测量。**

### 4.2 harness 对照（证明测量装置可用）

| 方法 | 实测 | 论文 | 差 |
|---|---|---|---|
| OptimFROG --preset max | 27.311 % | 25.42 % | +1.89 |
| WavPack -hh -x3 | 35.303 % | 33.15 % | +2.15 |
| FLAC -compression_level 12 | 45.334 % | 41.26 % | +4.07 |
| zlib 9 | 62.627 % | — | — |

顺序（OptimFROG < WavPack < FLAC）与量级都吻合，说明装置可用，
因此下面 bGPT 那一行的反差可信。

### 4.3 承重证据：checkpoint 到底在预测什么（直接测量）

| 窗口 | argmax 命中「同 slot 字节」 | 命中「下一个字节」 |
|---|---|---|
| 512 B | **0.756** | 0.551 |
| 1 024 B | **0.599** | 0.398 |
| 1 536 B | **0.549** | 0.272 |
| 2 048 B | **0.543** | 0.211 |
| 4 096 B | **0.471** | 0.132 |
| 8 160 B | **0.415** | 0.111 |

每个窗口长度上，模型对**同位字节**的预测都明显好于对**下一个字节**——即它在**重建**。
而原实现的索引运算 `logits[:-1,:-1,:]`（砍掉每行最后一格再丢 16 行）
把它实际评分的网格位置定在 1…8160，而 payload 位于 16…8175，
于是开头 15 与结尾 15 字节从未被评分，且每个分布都配到了相隔一个 patch 的目标。

按**真正的 next-byte 打分**（当前代码实测）：

| 量 | 值 |
|---|---|
| coded bits | 1 297 795.82 |
| 分母（8-bit PCM） | 82 220 字节 |
| **bits per original byte** | **15.7844** |
| **compression rate** | **197.31 %** |
| **compression ratio** | **0.5068×**（膨胀） |

> **勘误**：早期版本文档写 "13.12 bits/byte（197.3%）"，这两者根本不是同一个测量——
> 13.1233 bpb 对应 164.0% / 0.610×。13.1233 来自 §3.2 的 offset 扫描探针，
> 那是在对齐修复**之前**跑的，之后被误当成 headline。当前值为 **15.7844 bpb**，
> 可用 `tools/remeasure_audio_nextbyte.py` 复现。

**仍然无法排除的两种解释**：(a) 论文 headline 来自**另一次 bGPT-audio 训练**，
与 capsule 里发布的 checkpoint 不同；(b) reference 的索引算术本身就把数字抬高了。
区分需要作者提供训练配置或更多权重。**我们不主张 18.26% 错，只主张
"shipped code + shipped checkpoint 复现不出它"，以及它在 bundled sample 上报告的
1.94 bpb 不是那些字节的 next-byte 码长。**
""",
}

# Placeholder markers the file uses so the script can find each region.
MARKERS = {
    "SUMMARY": ("<!-- SECTION:SUMMARY:START -->", "<!-- SECTION:SUMMARY:END -->"),
    "TEXT_TABLE": ("<!-- SECTION:TEXT_TABLE:START -->", "<!-- SECTION:TEXT_TABLE:END -->"),
    "AUDIO": ("<!-- SECTION:AUDIO:START -->", "<!-- SECTION:AUDIO:END -->"),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    text = open(REPORT, encoding="utf-8").read()
    replaced = 0
    for key, body in SECTIONS.items():
        start, end = MARKERS[key]
        if start not in text or end not in text:
            print(f"  !! marker missing for {key}; skipping")
            continue
        pattern = re.compile(re.escape(start) + r".*?" + re.escape(end), re.S)
        text = pattern.sub(start + "\n" + body.rstrip() + "\n" + end, text)
        replaced += 1
        print(f"  ok  replaced section {key}")

    if not args.check:
        with open(REPORT, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
        print(f"  -> wrote REPORT_CN.md ({replaced} sections)")

    print("\n=== forbidden-token scan ===")
    bad = 0
    for line_no, line in enumerate(text.splitlines(), 1):
        for tok in FORBIDDEN:
            if tok in line:
                # allow inside explicit retraction context
                if any(w in line for w in ("勘误", "superseded", "撤回", "withdrawn", "早期版本", "早期版本文档")):
                    continue
                print(f"  !! line {line_no}: {tok!r} -> {line.strip()[:90]}")
                bad += 1
    print(f"  {bad} offending line(s)")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
