# 复现结果整理（当前状态）

**论文**：Li et al., *Lossless data compression by large models*,
Nature Machine Intelligence **7**:794–799 (2025)；预印本 *Understanding is Compression*
([arXiv:2407.07723](https://arxiv.org/abs/2407.07723))。Code Ocean capsule 4957991。

**指标口径**：论文用 **compression ratio = 原始/压缩后（越大越好）**。
本项目主要报 **rate = 压缩后/原始 %（越小越好）**，两者互为倒数。
凡涉及论文原文数字，来源为 arXiv 正文与其 NMI 补充材料（**capsule 内没有论文 PDF**）。

> **本文件只收录已核实数字。** 每个数字都标注了来源文件或脚本。
> 已撤回的结论单列在 §7，不与结论混排。

---

## 0. 一句话现状

**三个模型族、四种模态全部测过；四种模态的经典基线全部复现。
文本与音频复现成功；图像和视频的问题在"评测口径"，不在模型。**

---

## 1. 文本 —— 复现成功（最强的一环）

**设置**：MeDAL 抽样 3 文件，312 053 载荷字节 / 57 547 BPE tokens；
原始 Llama-3-8B ± 作者随 capsule 发布的 LoRA 适配器。
来源：`results/text_base_v2.jsonl`、`results/text_lora_v2.jsonl`。

| 方法 | rate | ratio | 论文 | 差 |
|---|---|---|---|---|
| LMCompress\*（原始 Llama-3-8B） | **9.0410 %** | 11.0607× | 9.08 % | **−0.04 pp** |
| LMCompress（+ 作者发布的 LoRA） | **9.3529 %** | 10.6919× | 9.06 % | +0.29 pp |
| brotli q11（payload 口径） | 28.6187 % | 3.46× | 27.93 % | +0.99 |
| lzma preset 9e（≈7z -mx9） | 31.850 % | 3.14× | 30.25 % | +1.60 |
| zstd --ultra -22 | 32.119 % | 3.11× | 30.45 % | +1.67 |
| zlib 9 | 35.155 % | 2.84× | — | — |

**要点**

* 原始模型与论文 LMCompress\* 差 **0.04 pp** —— 移植忠实。
* 模型比最好的经典基线好 **3.20×**，与论文"约为经典方法三分之一"同档。
* 基线顺序完全一致（brotli 最优、zstd ≈ 7z、zlib 最差）。

### 1.1 适配器：两个独立 harness 都测出"变差"

| harness | base rate | +LoRA rate | Δ |
|---|---|---|---|
| **我们端口**（ctx 4096） | 8.8631 % | 9.1269 % | **+0.2638 pp**（单文件） |
| **作者未修改的脚本**（ctx 8192） | **8.9253 %** | **9.5154 %** | **+0.5901 pp** |
| 论文 | 9.08 % | 9.06 % | −0.02 pp（称略好） |

来源：`results/_task1_ref_ablation.txt`、`results/_task1_lora_arm.txt`。

* **base 上两个独立实现差 0.0622 pp** → 端口忠实。
* **两个 harness 都显示适配器让压缩变差**，只是幅度不同（配置不同：我们 4096、参考 8192）。
* 论文自己的代码 + 论文自己的适配器，跑不出论文报告的方向。

**为什么适配器无益，证据链**

| 证据 | 内容 |
|---|---|
| 训练日志 | `trainer_state.json`：eval loss 最优 **2.4708 @ step 1020（≈2 epoch）**，退化到 **2.6728 @ step 5120**；发布的是 step 5120 |
| 合并无副作用 | 合并失真仅 **+0.003213 bits/token = 报告惩罚的 2.84%**（`results/_merge_cost_gpu.txt`） |
| 未合并更差 | 忠实（未合并）路径上适配器效应 **+0.207068 bits/token**，比合并路径的 +0.210281 更小 → 合并**不是**惩罚来源 |

**必须随结论声明的限定**：惩罚在 ctx 4096 测得，而论文自己的
Supplementary Table 9 显示 MeDAL 在 ctx 8192/6144/4096 为 9.06/9.16/9.36 %，
即 **4096↔8192 约 0.3 pp，与惩罚同量级**。所以"惩罚存在"有两个 harness 支撑，
但**归因于适配器而非上下文长度**仍需 ctx 8192 那一跑。

---

## 2. 音频 —— 复现成功

**设置**：`LJ001-0004.wav`，16 kHz 单声道 8-bit PCM，**82 220 载荷字节**，全部计入。
来源：`results/audio_bgpt_rows17_tailpad_v1.jsonl`、`results/_task1_ref_ablation.txt` 同批运行。

| 实现 | 编码输出 | bits/byte | rate | ratio |
|---|---|---|---|---|
| **我们端口** | 130 383 bit = 16 298 B | 1.5858（真实码流） | **19.82 %** | 5.046× |
| **参考实现（未修改脚本）** | 16 399.95 B | 1.5957 | **19.94 %** | 5.016× |
| 论文 LJSpeech | — | 1.4608 | **18.26 %** | 5.477× |

**要点**

* 我们编码器**真实吐出的码流**与参考实现差 **0.6 %**。
* 参考实现与论文的相对差 **9.2 %**，与经典基线的偏差同量级。
* **模型明显优于所有经典编码器**（19.8 % vs OptimFROG 27.3 %）。

### 2.1 经典基线

| 方法 | rate | 论文 | 差 |
|---|---|---|---|
| OptimFROG --preset max | 27.311 % | 25.42 % | +1.89 |
| WavPack -hh -x3 | 35.303 % | 33.15 % | +2.15 |
| FLAC -compression_level 12 | 45.334 % | 41.26 % | +4.07 |
| zlib 9 | 62.627 % | — | — |

顺序（OptimFROG < WavPack < FLAC）与量级均吻合。

---

## 3. 图像 —— 口径接近，但**不是对源图像无损**

**设置**：1 张 1714×1142 PNG，直接驱动官方代码
（`compress_with_igpt.py` → `language_model.compress`）。
来源：`results/official_image_accounting.json`。

| 量 | 值 |
|---|---|
| chunks | **5 733** |
| processed samples（分母） | **5 870 592** |
| 码长 | **910 788 字节** |
| **rate** | **15.5144 %** |
| **ratio** | **6.4456×** |
| 论文 CLIC2019 | ratio **6.32** |
| 源 RGB 通道样本 | 5 872 164 |
| 被 loader 整除丢弃 | **1 572**（每通道 524） |

**分母的确切含义**（`tools/explain_official_denominator.py` 核实）：
`compress_with_igpt.py:93` 的 `raw_length += h * w` 数的是
**源图像全分辨率、按通道计**的样本数（`process_img` 不缩放）；
而分子编的是 **32×32、512 色的量化网格**（每 chunk 1 024 token）。
**分子是缩略图，分母是原图** —— 这是 15.51 % 的真正含义。

### 3.1 为什么它不是无损的 —— 穷举证明

官方路径把每帧按通道切开，`language_model.compress` 收到 `(32,32,1)`，
再 `np.repeat(...,3)` → 喂给 processor 的每个输入都是单个 8-bit 通道值构成的 `(v,v,v)`。
**域是 256**，穷举结果（`results/igpt_grayscale_collisions.json`）：

| 量 | 值 |
|---|---|
| 域大小 | 256 |
| **像大小（不同 token id）** | **48** |
| 单射 | **False** |
| 碰撞值对 | **585** |
| 最大碰撞类 | token 391 ← 7 个值（78…84） |

例：`token 71 ← [0,1,2,3]`。真实 patch 上同样碰撞（patch #116：110 个源值 → 32 个 token）。
coder 只发 token id、无残差 → **不存在能唯一恢复源值的解码器**。
即**对 token 网格无损、对图像有损**，且分母是自洽的 —— 这是 **tokenisation 缺陷**，
不是 denominator mismatch。

### 3.2 经典基线

| 方法 | rate | 论文 CLIC2019 |
|---|---|---|
| webp -lossless -q100 | 48.279 % | 35.56 % |
| jpeg-2000 lossless | 53.252 % | 48.64 % |
| png (ffmpeg cl100) | 84.381 % | 47.08 % |

顺序复现。绝对值偏差大是因为这张测试图是 4K 航拍细节图，
比 CLIC 的专业摄影图难压得多（PNG 84.4 % vs 47.1 % 就是信号）。
**要公平比较必须换 CLIC2019 / ILSVRC 的图。**

---

## 4. 视频 —— 基线吻合度最高的一栏

**素材**：`4k_aerial_first_second.y4m`，4096×2160 @60fps，**60 帧**，
raw YUV420 = 796 262 400 字节。帧数用严格解析核对（无残留字节）。
来源：`results/baselines_video_wsl.jsonl`、`results/baselines_video_zstd.jsonl`、`results/video_igpt.jsonl`。

### 4.1 经典基线（全部 60 帧，lossless）

| 编码器 | 压缩后 | rate（raw YUV） | 耗时 | 论文 high-res | 差 |
|---|---|---|---|---|---|
| h.265 crf 0 (x265 veryslow) | 182 044 211 | **22.862 %** | 703 s | 21.69 % | +1.17 pp |
| h.264 crf 0 (x264 veryslow) | 293 551 029 | **36.866 %** | 29 s | 37.42 % | −0.55 pp |
| ffv1 level 3 | 319 836 906 | **40.167 %** | 5 s | 40.16 % | **+0.007 pp** |
| zstd --ultra -22（raw YUV） | 406 369 765 | **51.035 %** | 737 s | 51.04 % | **−0.005 pp** |

**四个全部吻合，顺序完全一致**（H.265 < H.264 < FFV1 < zstd），
最大偏差 1.17 pp，FFV1 与 zstd 吻合好于 **0.01 pp**。

### 4.2 论文自己的方法（iGPT 逐帧，刻意不用帧间信息）

| 量 | 值 |
|---|---|
| 每帧 token 数 | **1 024**（processor 恒定 32×32） |
| 每帧用到的调色板 id | 102–109 |
| 码长 | 22 293.9 字节总计，**≈371 字节/帧** |
| rate（论文口径） | **0.004200 %** |
| rate（÷ RGB 字节） | 0.001400 % |
| 论文 high-res | 15.31 % |
| **精确重建补全** | **73.0546 %**（残差 1 163 389 756 B） |

每帧 4096×2160 → 1024 token = **8 640× 缩减**。
371 字节只占一帧 RGB 字节的 **0.0014 %** —— token 流几乎不携带这一帧；
精确重建需 **≈73 %**，**比所有经典编码器都差**。
所以 iGPT 那一行应报为「**一个 32×32、512 色代理对象的码率**」，不是视频压缩率。

---

## 5. ImageGPT 量化器的性质（任务二答案）

**固定的、独立的预处理，不是压缩方法，不可训练。**

配置见 `data/model/imagegpt-small/preprocessor_config.json`：
512 个 RGB 聚类中心 + `do_resize 32×32` + `do_color_quantize true`。

| 问题 | 答案 |
|---|---|
| 只是对比试验用？ | 不是，**每条**图像/视频路径都用它 |
| 是论文的主要压缩方法？ | **不是**。方法是 iGPT + 算术编码；量化器是两者都需要的固定分词器 |
| 独立固定量化，还是语言模型？ | **独立固定**。iGPT checkpoint 自带的 k-means 调色板，无训练、无梯度、不属 LM 参数 |

论文 §2.1 也把它描述为 tokenisation（"transforming … into a sequence"），与代码一致。
**损失性正出在这里**（见 §3.1）。

---

## 6. 我们自己发现并修掉的真 bug

| # | bug | 后果 | 状态 |
|---|---|---|---|
| 1 | **自研算术编码器区间塌陷** | `p=9.06e-11` 时 `high<low`，**静默产生错误码流** | 已修（最小宽度 + 断言），`tests/test_coder.py` 8 项全过 |
| 2 | **文本目标切片晚一位** | 每个 token 在用**前一个 token** 的分布打分 | 已修 + 逐 token 断言；9.0333→9.0410、9.3459→9.3529 |
| 3 | `bits2bytes` 不存在之谜 | 真因是 `from_pretrained()` 未传 `torch_dtype` → fp32 8B ≈ 32 GB → OOM | 已查明 |

**第 2 条的方式值得记住**：我在自己代码里犯了与参考实现同一类的错误，
且"跑通 + 数字合理"完全掩盖了它 —— 是被审阅者质疑我那句错误解释时才查出来的。

### 6.1 环境问题（待修）

```
bitsandbytes 0.45.2 : "compiled without GPU support"；缺 libbitsandbytes_cuda128.dll
accelerate   0.27.2 : 已装
transformers 5.5.3  : 要求 accelerate >= 1.1.0
```

* **4-bit 量化在本机不可用**；请求即 `ImportError`。端口 loader 现在直接报错。
* 文档里所有"4-bit NF4"的说法**都是错的**，需更正。
* 时间线：文本结果产生于 **2026-10-05**，而 peft/accelerate/bitsandbytes 装于 **2026-10-07**
  → **那些结果不可能用过 bitsandbytes**，两臂用同一 loader，故 dtype 问题不单独制造 LoRA 惩罚。

---

## 7. 已撤回的结论（勿引用）

| 曾主张 | 状态 | 原因 |
|---|---|---|
| 参考实现音频打分"错位 16 位"，夸大 **6.29×** | **撤回** | **我们自己的探针索引错误**：一行 patch 有 **17** 个 slot，payload 位置 k 的正确行号是 `17*(k//16)+(k%16)`，我们用了 `16+k` |
| 由此导出的 13.12 / 13.1233 / 14.1804 / **15.7844** bits/byte 及 **197.31 %** | **作废** | 同上 |
| 音频不可复现论文 headline | **撤回** | 音频**复现成功**（§2） |
| 参考脚本因尾块崩溃、因缺 `samplings` 无法运行 | **撤回** | `samplings` 是已安装的 PyPI 包；未修改脚本跑完全部 11 块，报 `Total size: 82264` |
| 图像 330 字节 / 0.017 % / **67.43 %** "真正无损率" | **撤回** | 来自我们自己的 whole-image 实验，非官方路径 |
| 24 bits/pixel → 9 bits 的基数论证 | **撤回** | 域是 256、值域是 512，基数论证不成立；已换成穷举 |
| "20 bits 高于 byte alphabet 的 8 bits 上限"作为 `w_min` 理由 | **撤回** | 256 个符号不限制真字节的 surprisal；正确理由是报告值不消费 coder 量化概率 |
| order-1 经验熵是 "valid lower bound" | **降级** | 只是旁证；承重证据是直接测量 |

**一个跨文档教训**：本项目的主导错误模式不是单文档推理错，而是
**跨文档/跨版本的残留**。为此建了 `tools/check_doc_consistency.py`
（把 canonical facts 编码一次，扫描五份文档的旧值/缺失值，退出码可作提交门禁，当前 **exit=0**）。

---

## 8. 未完成 / 待办

| 项 | 说明 |
|---|---|
| **你自己的微调** | **零进度**。已测的是**作者发布的**适配器，不是你的训练 |
| ctx 8192 双臂对照 | 分离"适配器效应"与"上下文长度效应"（当前最大混淆项） |
| 训练量消融 | epochs ∈ {1,2,5,10} —— 论文从未做过，决定 MeDAL 结果是否为训练量效应 |
| 换模型（任务三） | GPT-2 124M 本地可训；ByT5-base / Llama-3-8B 需 Colab。**BERT 不适合**（masked-LM，非因果）。瓶颈是时间不是显存：64 MB 训练集 ≈ 16 M tokens，本地单 epoch 数天 |
| 图像换 CLIC2019/ILSVRC | 做同分布比较 |
| 音频多文件复验 | 目前 1 个样本 |
| 修依赖 | `accelerate>=1.1.0`（或降 transformers），并决定是否修 bitsandbytes CUDA 二进制 |
| 结果里记录真实 dtype + 峰值显存 | 让"4-bit"这类说法不可能再漂移 |

---

## 9. 交付物索引

| 路径 | 内容 |
|---|---|
| `src/uic/` | 自研实现（编码器/引擎/三模型/四模态/CLI） |
| `tests/test_coder.py` | 8 项编码器测试，全过 |
| `docs/AUDIT_FOR_REVIEW.md` | **主报告**：结论、推导、证据、可攻击点 |
| `docs/REVIEW_GUIDE_FOR_AGENT.md` | 给审阅 agent 的对抗式指南（含撤回记录） |
| `docs/THREE_TASKS_FINDINGS.md` | 三项调查的完整结论 + v6 notebook 反馈 |
| `docs/REVIEW_RESPONSE.md` | 三轮审阅逐条回应与勘误 |
| `docs/REPORT_CN.md` | 中文报告 |
| `docs/REPRODUCTION_NOTES.md` | 技术笔记 |
| `results/comparison_tables.md`、`video_tables.md` | 跨模态对比表 |
| `tools/review_harness.py` | 10 项机器可校验检查（10/10 通过） |
| `tools/check_doc_consistency.py` | 跨文档一致性门禁（exit=0） |
| `colab/LMCompress_Colab_H100.ipynb` + `lmcompress_colab_bundle.zip` | 待跑 |

---

## 10. 一段话总结

**文本复现成功**（原始模型差 0.04 pp；适配器无益，两个独立 harness 一致）。
**音频复现成功**（我们 19.82 % / 参考 19.94 % / 论文 18.26 %，真实码流相差 0.6 %）。
**图像与视频的模型数字不是源数据的无损码率** —— 那是一条把 4096×2160 帧压成
32×32、512 色缩略图再编码的路径，分子是缩略图、分母是原图；同一套预处理的
**非单射性已用穷举证明**（256→48，585 碰撞对）。
**四种模态的经典基线全部复现**，视频吻合最好（最大偏差 1.17 pp，FFV1/zstd <0.01 pp）。
**微调反了方向**：发布的适配器让压缩变差，且其训练日志显示最优在 ≈2 epoch。
剩余的核心工作是你自己的微调与 ctx 对照，两者都要上 Colab。
