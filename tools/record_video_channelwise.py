"""Record the channel-wise video result and its status.

The earlier video number (0.0042 %, exact completion 73.05 %) came from an RGB
tokenisation that does not match the official pipeline, and was withdrawn. This
records the corrected measurement and marks it provisional until all frames finish.
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

SECTION = """### 4.5 Video [M, provisional]

Source clip: `data/test_videos/4k_aerial_first_second.y4m`, 4096x2160 @60 fps,
**60 frames**, raw YUV420 = 796 262 400 bytes. Frame count verified by strict parsing
with zero leftover bytes (`tools/y4m_info.py`); an arithmetic heuristic returned 59 and
would have made every rate wrong by 1/60.

**Classical baselines, measured** (`tools/baselines_video.sh`,
`tools/baselines_video_zstd.sh`; ffmpeg from the capsule's own static build; all 60
frames):

| codec | compressed bytes | rate (raw YUV) | encode time | published | delta |
|---|---|---|---|---|---|
| h.265 crf 0 (x265 veryslow) | 182 044 211 | **22.862 %** | 703 s | 21.69 % | +1.17 pp |
| h.264 crf 0 (x264 veryslow) | 293 551 029 | **36.866 %** | 29 s | 37.42 % | -0.55 pp |
| ffv1 level 3 | 319 836 906 | **40.167 %** | 5 s | 40.16 % | **+0.007 pp** |
| zstd --ultra -22 over raw YUV | 406 369 765 | **51.035 %** | 737 s | 51.04 % | **-0.005 pp** |

Four for four: the ordering reproduces exactly (H.265 < H.264 < FFV1 < zstd), the
largest discrepancy is 1.17 pp, and FFV1 and zstd agree to better than 0.01 pp. This is
the tightest baseline agreement of any modality and validates the apparatus.

**The paper's own method, channel-wise, per frame.** Note the paper deliberately does
not use inter-frame information, so iGPT sees each frame independently while the
classical codecs above exploit temporal redundancy.

The official loader chunks a channel by flattening it row-major and cutting consecutive
`32*32` runs, so a 4096x2160 frame yields `4096*2160 // 1024 = 8640` chunks per channel
and **25 920 units per frame**. Measured on frame 0:

| quantity | value |
|---|---|
| units per frame | **25 920** (8 640 chunks x 3 channels) |
| samples per frame | **26 542 080** |
| coded bytes | **4 436 832** |
| **rate (official convention)** | **16.7162 %** |
| frame encode cost | 674 s (42 s tokenise + 632 s model) |

**Status: provisional.** One of sixty frames is measured; the remaining frames are
running (`tools/run_video_igpt_channelwise.py`, resumable per frame). The published
high-resolution figure is 15.31 %, so frame 0 already lands in the published regime.

**Two withdrawn numbers.** The previous figures of **0.0042 %** and an
**exact-reconstruction completion of 73.05 %** came from a whole-RGB tokenisation that
does not match the official channel-wise path. They differ from the corrected value by
roughly three orders of magnitude and are void. See section 4.4.4.

**A performance note worth keeping.** The first attempt measured 308 ms per unit, of
which 78 % was the arithmetic encoder rather than the model; adding
`measure_stream(..., with_coder=False)` cut it to 62 ms, and vectorising the scoring
over batches of 32 units brought a frame to 674 s. Without those two changes the job
would have taken 133 hours instead of 11.
"""


def main() -> int:
    text = io.open(AUDIT, encoding="utf-8").read()
    start = text.find("### 4.5 Video")
    if start < 0:
        print("  !! video section not found")
        return 1
    end = text.find("\n### ", start + 5)
    if end < 0:
        end = text.find("\n## ", start + 5)
    if end < 0:
        print("  !! could not find end of video section")
        return 1
    text = text[:start] + SECTION + text[end:]
    io.open(AUDIT, "w", encoding="utf-8", newline="\n").write(text)
    print("  ok  audit video section replaced")

    r = io.open(REPORT, encoding="utf-8").read()
    old = ("| 视频 | ImageGPT（iGPT 逐帧） | **数字暂时撤回** | 原 0.0042% 由**我们端口的 RGB 路径**产生，"
           "而官方是**逐通道**路径，两者 token 化不同（同一 patch：官方 9 个不同 id，我们 101 个）。"
           "需按官方路径重测后方可引用；详见 `AUDIT_FOR_REVIEW.md` §4.4.4 |")
    new = ("| 视频 | ImageGPT（iGPT 逐帧，逐通道） | **部分（1/60 帧）** | 第 0 帧 **16.7162 %**"
           "（每帧 25 920 单元 / 26 542 080 样本），论文 15.31 % —— 已落在论文量级；"
           "其余帧后台测量中。原 0.0042% 因 RGB/逐通道 token 化错误已作废（§4.4.4） |")
    if old in r:
        r = r.replace(old, new, 1)
        io.open(REPORT, "w", encoding="utf-8", newline="\n").write(r)
        print("  ok  report video row updated")
    else:
        print("  --  report video row not matched")
    return 0


if __name__ == "__main__":
    sys.exit(main())
