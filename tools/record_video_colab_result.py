"""Record the Colab channel-wise video result, replacing the withdrawn RGB numbers.

Colab output (A100, 10 frames of Netflix_Aerial_4096x2160_60fps_8bit_420.y4m):

  weighted rate        18.1508 %      published high-res 15.31 %   (+2.84 pp)
  per-frame median     18.1499 %
  frame-to-frame range 0.0219 pp      (std 0.00635)
  per frame            25 920 units / 26 542 080 samples / 478 s model + 18 s tokenise

Baselines on the SAME frames:

  H.265 crf 0   26.467 %   (published 21.69, +4.78)
  H.264 crf 0   34.087 %   (published 37.42, -3.33)
  FFV1 level 3  40.114 %   (published 40.16, -0.05)
  zstd -22      46.292 %   (published 51.04, -4.75)

So the model beats the best baseline by 1.46x and the ordering reproduces exactly.

Also records a presentation bug in the notebook: each JSONL row wrote the aggregate
coded_bytes (sum over all frames) rather than that frame's own bytes, making every row
look identical. The rates were always computed from bits and are unaffected.
"""

from __future__ import annotations

import io
import json
import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # pragma: no cover
    pass

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUDIT = os.path.join(HERE, "docs", "AUDIT_FOR_REVIEW.md")
REPORT = os.path.join(HERE, "docs", "REPORT_CN.md")

AUDIT_SECTION = """### 4.5 Video [M]

Source: **`Netflix_Aerial_4096x2160_60fps_8bit_420.y4m`**, from the Xiph.org collection
the paper draws on -- 4096x2160 @60 fps, aerial footage, **10 frames** streamed over HTTP
Range (133 MB of a 15.9 GB file). Measured on an A100 via
`colab/Video_ChannelWise_Colab.ipynb`.

Frame count and geometry were verified by parsing the Y4M header and frame markers
directly, and the channel-wise unit count (8640 per channel, **25 920 per frame**) was
checked against the official loader's arithmetic.

**The paper's own method, channel-wise.** The paper deliberately uses no inter-frame
information, so iGPT codes each frame independently while the classical codecs below may
exploit temporal redundancy.

| quantity | value |
|---|---|
| frames | **10** |
| units per frame | **25 920** (8 640 chunks x 3 channels) |
| samples per frame | **26 542 080** |
| **weighted rate (official convention)** | **18.1508 %** |
| per-frame median | 18.1499 % |
| frame-to-frame range | **0.0219 pp** (std 0.00635) |
| published, SI Table 4 high-res | 15.31 % |
| **delta** | **+2.84 pp** |
| cost per frame | 478 s model + 18 s tokenise |

The frame-to-frame spread is **0.02 pp**, so ten frames pin the mean to about
0.007 pp and the +2.84 pp gap is not sampling noise on this clip.

**Classical baselines on the identical frames**, run through the same harness:

| codec | ours (%) | published (%) | delta |
|---|---|---|---|
| H.265 crf 0 (x265 veryslow) | **26.467** | 21.69 | +4.78 |
| H.264 crf 0 (x264 veryslow) | **34.087** | 37.42 | -3.33 |
| FFV1 level 3 | **40.114** | 40.16 | **-0.05** |
| zstd --ultra -22 over raw YUV | **46.292** | 51.04 | -4.75 |

Two results of note. First, **the ordering reproduces exactly**
(H.265 < H.264 < FFV1 < zstd), which is the check that the harness is measuring what the
paper measured. Second, **the model beats the best codec by 1.46x** (18.15 % against
26.47 %), which is the paper's headline qualitative claim for video and it holds here
independently of the corpus.

**What the +2.84 pp gap is, and is not.** The baselines themselves move by up to 4.8 pp
against their published values on this clip (H.265 +4.78, zstd -4.75), so clip-to-clip
variation of several points is established by our own controls. Our local clip -- a
different aerial shot -- measured **16.7162 %** on its first frame, 1.4 pp closer to the
published value than this one. The honest reading is that the model lands in the
published regime and that the residual is dominated by which clip is used, which
`n = 10` frames of one clip cannot resolve.

**Withdrawn numbers.** The previously reported **0.0042 %** and an exact-reconstruction
completion of **73.05 %** came from tokenising whole RGB frames rather than channels, a
defect recorded in section 4.4.4. They are void; the corrected figure differs by three
orders of magnitude.

**A presentation bug worth recording.** Each row of the Colab JSONL wrote the *aggregate*
`coded_bytes` rather than that frame's own bytes, so all ten rows appear identical in
that column while their `coded_bits` differ. Every rate was computed from `coded_bits`
and is correct; only the redundant column is wrong. Fixed in the notebook and noted here
so a reader comparing rows is not misled into thinking frames were duplicated.
"""


REPORT_SECTION = """### 5.3 视频（Colab A100，10 帧，已定稿）

**素材**：`Netflix_Aerial_4096x2160_60fps_8bit_420.y4m`，来自论文所用的 Xiph.org 集合，
4096×2160 @60fps 航拍，**10 帧**经 HTTP Range 流式取得（15.9 GB 中的 133 MB）。
在 A100 上跑 `colab/Video_ChannelWise_Colab.ipynb`。

**论文自己的方法（逐通道）**：论文刻意不用帧间信息，所以 iGPT 逐帧独立编码，
而下面经典编码器可以利用时间冗余。

| 量 | 值 |
|---|---|
| 帧数 | **10** |
| 每帧单元 | **25 920**（8640 chunk × 3 通道） |
| 每帧样本 | **26 542 080** |
| **加权 rate（官方口径）** | **18.1508 %** |
| 每帧中位数 | 18.1499 % |
| 帧间极差 | **0.0219 pp**（标准差 0.00635） |
| 论文 SI Table 4 高分辨率 | 15.31 % |
| 差 | **+2.84 pp** |
| 单帧成本 | 478 s 模型 + 18 s 分词 |

**帧间波动只有 0.02 pp**，所以 10 帧把均值钉到约 0.007 pp，**+2.84 pp 的差距不是这份片段的采样噪声**。

**同一批帧上的经典基线**

| 编码器 | 我们 (%) | 论文 (%) | 差 |
|---|---|---|---|
| H.265 crf 0 (x265 veryslow) | **26.467** | 21.69 | +4.78 |
| H.264 crf 0 (x264 veryslow) | **34.087** | 37.42 | −3.33 |
| FFV1 level 3 | **40.114** | 40.16 | **−0.05** |
| zstd --ultra -22 (raw YUV) | **46.292** | 51.04 | −4.75 |

两个要点。第一，**顺序完全复现**（H.265 < H.264 < FFV1 < zstd）—— 这是"装置在测论文所测之物"的检验。
第二，**模型优于最好编码器 1.46 倍**（18.15 % vs 26.47 %），这是论文对视频的核心定性主张，
在这里**独立于语料地成立**。

**+2.84 pp 是什么、不是什么**：基线自身在这份片段上最多偏离论文值 **4.8 pp**
（H.265 +4.78，zstd −4.75），所以"片段间差异可达数个百分点"已由我们自己的对照确立。
我们本地那份不同的航拍片段首帧测得 **16.7162 %**，比这份更接近论文值 1.4 pp。
**诚实的解读是：模型落在论文量级内，残差主要由片段决定，而 10 帧单片段无法再分辨。**

**已作废的数字**：此前的 **0.0042 %** 与精确补全 **73.05 %** 来自整帧 RGB token 化
（缺陷记录在 `AUDIT_FOR_REVIEW.md` §4.4.4），与更正后的值差三个数量级，已作废。

**一个记录在案的呈现 bug**：Colab 输出的 JSONL 每一行写的是**汇总后**的 `coded_bytes`
而不是该帧自己的字节数，所以十行的该列看起来相同，而 `coded_bits` 是不同的。
**所有 rate 都由 `coded_bits` 算出，均正确**；只有那一列冗余字段是错的。
已在 notebook 修正并记录于此，避免读者误以为帧被重复计算。
"""


def main() -> int:
    d = json.load(open(os.path.join(HERE, "results", "video_channelwise_summary.json"),
                       encoding="utf-8"))
    print(f"  rate {d['rate_pct_official']:.4f} %, frames {d['frames']}")

    a = io.open(AUDIT, encoding="utf-8").read()
    start = a.find("### 4.5 Video")
    if start < 0:
        print("  !! audit video section not found")
        return 1
    end = a.find("\n### ", start + 5)
    if end < 0:
        end = a.find("\n## ", start + 5)
    if end < 0:
        end = len(a)
    a = a[:start] + AUDIT_SECTION + a[end:]
    io.open(AUDIT, "w", encoding="utf-8", newline="\n").write(a)
    print("  ok  audit video section replaced")

    r = io.open(REPORT, encoding="utf-8").read()
    start = r.find("### 5.3")
    if start < 0:
        print("  !! report video section not found")
        return 0
    end = r.find("\n## ", start + 5)
    if end < 0:
        end = len(r)
    r = r[:start] + REPORT_SECTION + "\n" + r[end:]
    io.open(REPORT, "w", encoding="utf-8", newline="\n").write(r)
    print("  ok  report video section replaced")

    # status-table row
    r = io.open(REPORT, encoding="utf-8").read()
    old = ("| 视频 | ImageGPT（iGPT 逐帧，逐通道） | **部分（1/60 帧）** | 第 0 帧 **16.7162 %**"
           "（每帧 25 920 单元 / 26 542 080 样本），论文 15.31 % —— 已落在论文量级；"
           "其余帧后台测量中。原 0.0042% 因 RGB/逐通道 token 化错误已作废（§4.4.4） |")
    new = ("| 视频 | ImageGPT（iGPT 逐帧，逐通道） | **是（10 帧，Colab A100）** | "
           "**18.1508 %**（帧间极差仅 0.0219 pp），论文 15.31 %；同帧基线最佳为 H.265 26.467 %，"
           "**模型优 1.46 倍**，且基线顺序完全复现。原 0.0042 % 已作废（§4.4.4） |")
    if old in r:
        r = r.replace(old, new, 1)
        io.open(REPORT, "w", encoding="utf-8", newline="\n").write(r)
        print("  ok  report status row updated")
    else:
        print("  --  report status row not matched")
    return 0


if __name__ == "__main__":
    sys.exit(main())
