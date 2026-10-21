# 验证报告：作者发布的 LoRA 方向 + ImageGPT 量化路径

**验证者**：本次会话（独立于产出原结论的那次会话）
**日期**：2026-10-11
**方法**：不采信既有文档，重新读源码 + 重新跑作者原始脚本 + 动态插桩

---

## 0. 结论速览

| # | 待验证命题 | 结论 | 关键证据 |
|---|---|---|---|
| 1 | 作者发布的 LoRA 在**作者自己的源代码**上确实让文本压缩变差 | **成立，且逐位复现** | 两次独立运行：base `11.204097`、+LoRA `10.509252`，两次结果完全相同 |
| 2 | 源代码中**图片**经过 ImageGPT processor 量化 | **成立** | 动态插桩：`--dataset imagenet` 路径 40 次 processor 调用，`do_color_quantize=True` |
| 3 | 源代码中**视频**经过 ImageGPT processor 量化 | **成立，与图片同一代码路径** | `--dataset raw_video_frames` 路径同样 40 次调用 |
| 4 | 该量化有损 | **成立** | 256 个源灰度值 → 48 个 token，585 个碰撞对 |
| 5 | 文档中"每条图像/视频路径都用 iGPT" | **需要修正** | 仓库里另有两个图像/视频入口**不经过** iGPT |
| 6 | 文档中"视频每帧 1024 token / 8640× 缩减" | **需要修正** | 作者 loader 实测每帧 **25 920** 个 chunk |

原始输出：`results/_verify_refrun_base.txt`、`results/_verify_refrun_lora.txt`、
`results/_verify_igpt_paths.txt` / `.json`、`results/_verify_video_rate.txt`。
脚本：`tools/verify_lora_reference_run.py`、`tools/verify_igpt_paths.py`、
`tools/verify_video_rate.py`、`tools/verify_lora_direction.py`。

---

## 1. 命题一：发布的 LoRA 在源代码上更差 —— 成立

### 1.1 为什么不沿用原证据

原结论来自两次**分离的进程**（`results/_task1_ref_ablation.txt` 16:51 只跑了 base，
`results/_task1_lora_arm.txt` 17:11 单独跑 LoRA）。"同进程对照"是更强的控制条件，
所以先尝试在一个进程内依次跑两臂 ——

```
第二个 8B 模型加载时: OSError: 页面文件太小，无法完成操作。(os error 1455)
```

本机 11.94 GiB 显存 + 15 GiB 权重，一个进程装不下两个 8B 模型。
**这同时解释了原 agent 为什么分两次跑** —— 不是疏忽，是硬件限制。

### 1.2 那么分进程会不会污染结论？不会，因为它可复现

用作者**未修改**的 `compress_with_bgpt_or_llama.compress_with_llm`，
同一文件 `medal_102285.csv`，`context_window=8192`、`batch_size=1`，
只改 `lora_path`：

| 臂 | 本次验证 | 原 agent | 一致？ |
|---|---|---|---|
| base（无适配器） | **11.204097** | 11.204097 | 逐位相同 |
| + 作者发布的 LoRA | **10.509252** | 10.509252 | 逐位相同 |

两侧都是确定性数值、两次运行完全相同，**所以进程身份不影响结果**，
原结论的证据强度不因"分两次跑"而降低。

| 臂 | ratio | rate = 100/ratio | 压缩后字节 | 分母 |
|---|---|---|---|---|
| base | 11.204097 | **8.9253 %** | 9 226.0 | 103 369 |
| + LoRA | 10.509252 | **9.5154 %** | 9 836.0 | 103 369 |
| 差 | −0.694845 | **+0.5901 pp（更差）** | +610 B | 相同 |

分母完全相同（103 369 字符）、只换了适配器，**方向明确：更差**。

### 1.2b 逐窗口剖面：惩罚是均匀的，不是被个别窗口带偏

用本项目的独立实现（重建作者 csv 路径的 tokenize / shift / mask / NLL / ratio 公式）
在**同一窗口集合** {0, 4, 8, 12}（跨整个文件，5 595 token）上对照：

| 窗口 | tokens | base bits | +LoRA bits | Δ bits | Δ bits/token | 适配器 |
|---|---|---|---|---|---|---|
| 0 | 1 495 | 5 856 | 6 112 | **+256** | +0.171 | 更差 |
| 4 | 1 646 | 6 272 | 6 400 | **+128** | +0.078 | 更差 |
| 8 | 1 613 | 5 920 | 6 176 | **+256** | +0.159 | 更差 |
| 12 | 841 | 3 728 | 3 936 | **+208** | +0.247 | 更差 |
| 合计 | 5 595 | 21 776 | 22 624 | **+848** | **+0.1515** | 更差 |

> bits 的粒度是 32（bf16 尾数，sum≈6000 时 ULP=2^13−8=32），所以 128/256/208 都是
> 整数个 ULP，不是巧合。

**4 个窗口全部同号**，跨度覆盖整个文件，说明总体的 +0.59 pp 不是个别窗口造成的。
（注：脚本里 λ 是相对 peft 原生幅度 `alpha/r = 4.0` 定义的 —— 我第一版脚本把
`scaling` 直接写成 1.0，等于只用了 1/4 强度，那一版才会看到"适配器更好"。
已修正，见 §1.5。）

### 1.3 机制：发布的适配器正是它自己那一轮里最过拟合的点

我独立读取了 `data/model/finetuned_llama/Llama-3-8B_lora/checkpoint-5120/trainer_state.json`：

| 量 | 值 |
|---|---|
| `num_train_epochs` | **10** |
| 最后一 step | 5 120（epoch 10.0） |
| eval loss 最低 | **2.4708 @ step 1020（epoch 1.99）** |
| eval loss @ step 5120 | **2.6728** |
| 相对最优点退化 | **+0.202 nats（+8.2 %）** |
| `best_model_checkpoint` | **None**（没有 `load_best_model_at_end`） |

曲线形态是教科书式过拟合：前 2 个 epoch 单调下降，之后分段抬升
（epoch 2→3、3→4、4→5…每段开头都跳高再缓降）。

**一个重要的澄清（本次新增，修正原文档的措辞）**：原文档把"换 checkpoint"
列为可能解释之一。但论文补充材料 S1.4 明确写
*"The model was trained for 10 epochs … The base model was loaded in 8 bits
precision and trained under bfloat16 precision."* —— 所以 **step-5120 就是论文配方
应当产出的那个模型**，不是发错了 checkpoint。这不是对作者的辩解，反而把问题
钉得更死：**论文自己的训练配方把适配器训过头了**。

### 1.4 必须随结论一起声明的限定（不能省）

论文补充材料 Table 6（MeDAL，CR%，越小越好）：

| | 论文 SI | 本次在作者代码上实测 |
|---|---|---|
| base（LMCompress\*，未微调） | 9.08 % | **8.9253 %**（比论文好 0.15 pp） |
| + LoRA（LMCompress） | 9.06 % | **9.5154 %**（比论文差 0.46 pp） |
| 论文的方向 | 好 **0.02 pp** | 差 **0.5901 pp** |

三点必须讲清楚：

1. **不是整体偏移**。我们的 base 比论文的 base 更好，而 LoRA 比论文的 LoRA 更差。
   如果只是"测试集更容易"，两臂应当同时变好。
2. **测试集规模差约 3 个数量级**。已发表版（NMI 正文）写 MeDAL 取
   *"the first approximately **180 megabytes**"*，去掉前 64 MB 训练 + 16 MB 验证
   → 测试集 ≈ **100 MB**（预印本写的是 1104 MB → ≈1 GB，两版差 6 倍，
   见 §5）。我们用的是 capsule 里的**一个 104 KB 文件**（19 338 token）。
3. 因此**不能**说"论文错了"。能说的只有：**论文自己发布的代码 + 自己发布的适配器，
   在论文自己发布的样例上，跑不出论文报告的方向**。单文件结论的泛化性未验证
   （见 §3 未完成项）。

另：论文正文说"Compared to raw LLaMA3-8B, LMCompress improves the compression ratio
by **8.5 %** on MeDAL"，而它自己的 SI Table 6 只显示 9.08 → 9.06（≈0.2 %）。
论文内部这两处本身就不一致。

---

### 1.5 适配器的"方向"是好的，坏在"幅度"（本次新增，解释性最强的一条）
peft 的原生 LoRA 幅度是 `lora_alpha / r = 32 / 8 = **4.0**`（脚本已实测打印）。
把适配器增量按 λ 倍原生幅度缩放（λ=1.0 即作者的原始设置），
在同样的 4 个窗口上：

| λ（× 原生 4.0） | W0 | W4 | W8 | W12 | 合计 bits | rate | 相对 base |
|---|---|---|---|---|---|---|---|
| **0.00**（= base） | 5 856 | 6 272 | 5 920 | 3 728 | 21 776 | 9.1832 % | — |
| **1.00**（作者设置） | 6 112 | 6 400 | 6 176 | 3 936 | 22 624 | 9.5408 % | **+848 bits = +0.3576 pp（更差）** |
| **0.25** | 5 664 | 6 016 | 5 664 | 3 600 | **20 944** | **8.8324 %** | **−832 bits = −0.3508 pp（更好）** |

逐窗口符号完全一致：λ=1.0 时 **4/4 窗口更差**，λ=0.25 时 **4/4 窗口更好**。
（原始输出：`results/_verify_lora_profile.txt`）

即：**同一个适配器，缩小到 1/4 幅度就从"伤害"变成"帮助"，且幅度几乎对称
（+848 / −832 bits）**。这与 §1.3 的训练日志完全吻合 —— eval loss 在 epoch 2 最低、
epoch 10 高出 8.2 %，说明权重被推得过头了。论文报的 +0.02 pp "微弱改善"，
量级上更像一个**幅度接近 0** 的适配器应有的效果；而发布出来的是训满 10 epoch 的那一个。

这条同时解释了为什么论文的 SI（9.08 → 9.06）与本文实测（8.93 → 9.52）方向相反：
**不是 harness 分歧，而是同一适配器在幅度轴上位于增益/损害的两侧。**
它也是本次验证里唯一一条**可操作**的结论：要用这个适配器，应先降幅度（λ≈0.25），
或在 epoch≈2 处早停重训。



### 1.6 与论文逐项对表（回应"我们的 base 是否等于论文的微调版"）

论文原文（已发表 SI，`lit/nmi_supp.txt`）：

* Table 6：`LMCompress* [11,17]` = MeDAL **9.08** / Pile of Law **5.86**；
  `LMCompress` = MeDAL **9.06** / Pile of Law **4.90**。
  表注写明 LMCompress\* *"utilizes the raw LLaMA3-8B model without fine-tuning"*。
* Table 9 的引言（SI S3）：*"we … evaluate **LMCompress** using three different
  context lengths"* → Table 9 测的是**已微调**的 LMCompress，不是 base。

| 测什么 | 文件 | ctx | base（无 LoRA） | +LoRA |
|---|---|---|---|---|
| 论文 SI Table 6 | ~1 GB 测试尾段 | — | 9.08 % | **9.06 %** |
| 论文 SI Table 9（已微调） | MeDAL | **8192** | — | 9.06 % |
| 论文 SI Table 9（已微调） | MeDAL | 6144 | — | 9.16 % |
| 论文 SI Table 9（已微调） | MeDAL | 4096 | — | 9.36 % |
| **本次：作者原始脚本** | 1 文件 (medal_102285) | **8192** | **8.9253 %** | 9.5154 % |
| 本项目端口（前次） | 3 文件 | 4096 | **9.0411 %** | 9.3528 % |

**"我们的 base ≈ 论文的微调版"这个读法是对的，而且还要更强一点：**

| 比较 | 差 |
|---|---|
| 我们 base @8192 **8.9253 %** vs 论文 **base** 9.08 % | 我们好 **0.15 pp** |
| 我们 base @8192 **8.9253 %** vs 论文 **微调** 9.06 % | 我们好 **0.13 pp** |
| 我们 base @4096 **9.0411 %** vs 论文 **微调** @4096 9.36 % | 我们好 **0.32 pp** |
| 我们 base @4096 **9.0411 %** vs 论文 **微调** @8192 9.06 % | 我们好 **0.02 pp** |

**关键点：论文自己宣称的"微调收益"只有 0.02 pp（9.08→9.06），
比"我们的 base 与论文 base 之间的差距（0.15 pp）"还小。**
也就是说这个收益在现有测量精度下**根本分辨不出来**，而我们的数据在相反方向上更大。

必须同时声明的两点：

1. **测试集不同**（论文 ~1 GB 尾部 vs 我们 1–3 个 104 KB 抽样文件）。
   我们 base 更好，最可能的原因是抽样文件比语料尾部更容易，
   **不能**据此说我们的实现更强。
2. 但"论文的微调在 MeDAL 上有帮助"这一条**不可复现**，且方向在我们这里反了。

一个值得注意的旁证：论文自己的两个数据集差异极大 ——
Pile of Law 微调收益 **0.96 pp（相对 16 %）**，MeDAL 只有 **0.02 pp（相对 0.2 %）**。
所以论文自己的数据已经显示"**在 MeDAL 上微调几乎没用**"。
分歧其实很窄：只是那 0.02 pp 的**符号**，以及我们测到的 −0.59 pp。

### 1.7 "第 2 轮最好（8.83 %）"在论文里提到了吗？——没有；而且 8.83 % 不是 2 轮的结果

**先纠正一个标签。** `8.8324 %` 的准确身份是：

| 是 | 不是 |
|---|---|
| 发布的 **step-5120（第 10 轮）** 适配器 | ❌ 不是训到第 2 轮的 checkpoint |
| 把它的增量**整体乘以 0.25**（λ=0.25） | ❌ 不是早停重训出来的模型 |
| 在 **1 个文件的 4 个窗口**（29 641 字符 / 5 595 token）上测得 | ❌ 不是全文件 |
| rate = 8.8324 % | ❌ 不能直接和论文 9.06 % 比 |

**为什么这个区分重要**：LoRA 的 B 初始化为 0，训练过程中**方向和幅度同时变化**。
把最终增量整体缩放，只改了幅度、没改方向。所以 λ=0.25 是
"**同一个适配器、更弱的力度**"，不是"**更早的适配器**"，二者不等价。

**真正的"第 2 轮"是什么**：`trainer_state.json` 显示 eval loss 最低在
step 1020（epoch 1.99）= **2.4708**，到 step 5120 退化到 2.6728。但：

* capsule **只发布了 step-5120 一个 checkpoint**（`checkpoint-5120/` 是唯一的），
  step 1020 的权重不存在 → **测不了**；
* eval loss 是在论文的验证集（前 64 MB 之后的 16 MB）上算的，不是我们的测试文件；
* 对这条流水线，"压缩率"本质就是模型在被压数据上的交叉熵，所以
  "eval loss 更低 → 压缩更好"是合理预期，但**本次并未实测**。

**论文里有没有？没有。** 已对已发表 SI 全文核对：

| 检索项 | 结果 |
|---|---|
| 出现 "epoch" | **仅 1 次**，即超参句：*"The model was trained for 10 epochs with a initial learning rate of 10⁻⁴, while the first 10% of the steps were warm up steps."* |
| per-epoch / 逐轮结果表 | **无** |
| 训练量消融（1/2/5/10 轮） | **无** |
| 早停 / 选最优 checkpoint | **无**（`best_model_checkpoint = None`） |
| 论文唯一的微调对比 | 仅 Table 6 两行：MeDAL 9.08→9.06；Pile of Law 5.86→4.90 |

**推论**：论文的配方就是训满 10 轮且不选最优 checkpoint，所以论文报的 9.06
**就是第 10 轮的数**。"2 轮更好"这件事论文既没报、也没测 ——
它是我们从 capsule 里的 `trainer_state.json` 反推出来的，属于**新发现**，不是论文内容。

要真正坐实，需要拿到（或重训出）epoch≈2 的 checkpoint，在**同一测试集**上测压缩率 ——
这正是原文档 §8 "训练量消融 epochs ∈ {1,2,5,10}" 那条待办，需要在 Colab 上做。

---

## 2. 命题二：图片和视频都过 ImageGPT 量化 —— 成立

### 2.1 静态路径（读源码）

```
compress_with_igpt.py  main()
  └─ args.compressor 被强制为 "language_model"            (第 136 行)
  └─ data_loaders.GET_DATA_GENERATOR_FN_DICT[dataset]      (第 147 行)
       ├─ "imagenet"          → get_imagenet_iterator       (图片)
       └─ "raw_video_frames"  → get_raw_video_double_frame_iterator (视频)
  └─ compressor.COMPRESS_FN_DICT["language_model"] = language_model.compress
       └─ language_model.py:175/180  image_processor(sequence_array, ...)
            └─ ImageGPTImageProcessor, do_color_quantize=True
```

两条 loader 都 `is_channel_wised=True`，都把每个通道拆成 `(32,32,1)` 的 chunk；
`language_model.compress` 第 173 行看到 `shape[-1] != 3`，走第 178-180 行的
`np.repeat(..., 3, axis=2)` 造出 `(v,v,v)` 灰度三元组，再送进 processor。

`utils/preprocess_video_dataset.py:y4m2png` 正是为 `raw_video_frames` 准备输入
（`.y4m` → 逐帧 PNG），说明视频这条路是设计好的。

### 2.2 动态证明（实际插桩运行，不是推理）

用 spy 包住 `ImageGPTImageProcessor.__call__`，**原样运行作者的 `main()`**：

| 数据集 | processor 调用次数 | 观测到的输入 | 观测到的输出 | `do_color_quantize` |
|---|---|---|---|---|
| `imagenet` | **40** | `(32,32,3)` uint8 | `(1,1024)` int64 token | **True** |
| `raw_video_frames` | **40** | `(32,32,3)` uint8 | `(1,1024)` int64 token | **True** |

两条路径**都**实测到达 processor，输入形状都是 `(32,32,3)`（即已经过单通道复制），
输出都是 1024 个 token id。processor 配置：`do_color_quantize=True`、
`do_resize=True`、`size=32×32`、512 个聚类中心。

### 2.3 该量化是有损的（穷举）

对 256 个可能的 8-bit 灰度值（构造成作者那样的 `(v,v,v)`）穷举：

| 量 | 本次验证 | 原文档 | 一致？ |
|---|---|---|---|
| 域大小 | 256 | 256 | ✓ |
| 不同 token 数（像大小） | **48** | 48 | ✓ |
| 碰撞值对 ΣC(n,2) | **585** | 585 | ✓ |
| 最大碰撞类 | token 391 ← 78…84（7 个值） | 同 | ✓ |
| 示例 | token 71 ← [0,1,2,3] | 同 | ✓ |

> 说明：原文档的"585 碰撞对"是 ΣC(n,2)（无序值对）；
> 等价的另一种说法是"256 个值里只有 48 个能存活，即 208 个值是冗余的"。
> 两者不矛盾，我两种都算过。

coder 只发 token id、不带残差 → **不存在能唯一恢复源值的解码器**。
即"对 token 网格无损、对图像有损"。

### 2.4 需要修正的两处（本次新发现）

**(a) "每条图像/视频路径都用它" —— 不准确。** 仓库里共有三个入口：

| 入口 | 模态 | 是否过 iGPT 量化 |
|---|---|---|
| `compress_with_igpt.py --dataset imagenet` | 图片 | **是** |
| `compress_with_igpt.py --dataset raw_video_frames` | 视频（PNG 帧） | **是** |
| `compress_with_bgpt_or_llama.py --data_type bmp\|video\|wav` | 图片/视频/音频 | **否** —— bGPT 逐字节建模，无 processor |
| `compress_with_baseline.py --mode text`（扩展名白名单含 `.y4m`） | 视频基线 | **否** —— zstd/zpaq/7z/brotli |

准确说法应是：**论文方法这条路（iGPT）上，图片和视频都过量化；
但仓库里另有 bGPT 逐字节路径和经典编码器路径不过量化。**
而且 `run.sh` 只演示了图片走 iGPT，视频只演示了经典基线 ——
只看 `run.sh` 会误以为视频不过 iGPT。

**(b) 文档里"视频每帧 1024 token / 8640× 缩减 / rate 0.0042 %"是另一套构造，不是作者代码。**
`tools/run_video_igpt.py` 用的是本项目自己的 `uic.data.image_units`
（整帧 resize 到 32×32 → 1024 token/帧）。作者的 loader 不是这样：

| | 本项目旧构造 | **作者真实 loader** |
|---|---|---|
| 每帧 chunk 数 | 1 | **25 920**（3 通道 × 8 640） |
| 每帧 token 数 | 1 024 | **26 542 080** |
| 分母（每帧样本） | 8 847 360 像素 | 26 542 080 |
| 依据 | 整帧缩略图 | `_extract_image_sequence` 对**每个通道的全分辨率光栅**切 1024 像素段 |

作者代码里**没有** 8640× 缩减：它把每个 1024 像素段当成一个 32×32 量化单元，
所以 token 数 ≈ 源样本数（每像素每通道 1 个 token）。
我按作者真实路径做了有界测量（前 150 chunk，仅示意）：

| 路径 | bytes/chunk | rate（论文口径 = bytes/(chunks×1024)） | 论文值 |
|---|---|---|---|
| 图片（本机原图，全跑） | 158.87 → 15.5144 % | 15.5144 % | 14.62 %（SI Table 3） |
| 图片（前 150 chunk） | 203.76 | 19.90 % | —— |
| 视频（前 150 chunk） | **87.15** | **8.51 %** | 15.31 %（SI Table 4） |

> 视频这 150 chunk 只占整个片段 1 555 200 chunk 的 **0.0096 %**，且集中在第 1 帧
> 的一小块空间区域，**只能当量级示意，不能当测量值**。全跑按论文的 144 h 量级不现实。

---

## 3. 未完成 / 仍然存在的风险

| 项 | 说明 |
|---|---|
| LoRA 结论的泛化性 | 只测了 1 个 104 KB 文件（19 338 token），论文测试集 ≈ 1 GB。需要多文件复验才能说"该适配器普遍更差" |
| ctx 混淆 | **已消除**：两臂都在 ctx 8192、同一 harness 内测得（论文 SI Table 9 显示 4096↔8192 约 0.3 pp）|
| 视频真实 rate | 未全跑；25 920 chunk/帧 的结构性结论是确定的，rate 只是示意 |
| 图像公平比较 | 仍应换 CLIC2019/ILSVRC 的同分布图片 |
| 4-bit 路径 | 本机 bitsandbytes 无 CUDA 二进制，`load_in_4bit` 报 ImportError；与本次两个结论无关（两臂同路径）|

---

## 4. 追加：论文的图片/视频方法**不是**无损的，以及正确的无损对比怎么做

### 4.1 直接答案：不是无损

论文 NMI 版标题是 *Lossless data compression by large models*，其中
**文本**（Llama token）与**音频**（bGPT 直接对 8-bit 字节建模）确实按构造无损 ——
它们的字母表就是源字母表。**但图片和视频的 iGPT 路径不是**，证据是一条闭链：

1. **送进编码器的符号是 token id，不是像素值。** `language_model.py` 里
   `encoder.encode(normalize_pdf_for_arithmetic_coding(pdf), symbol)`，
   而 `symbol` 来自 `sequence_array = image_processor(...)['input_ids']`（第 175/180 行）。
   码流里**没有任何残差或边信息**。
2. **量化器非单射**（§2.3）：256 个源灰度值 → 48 个 token，585 个碰撞对。
   非单射的映射不存在逆映射。
3. ⇒ **不存在能精确恢复源像素的解码器**。它只是对"32×32、512 色 token 网格"无损。

一个必须说清的落差：**论文正文和补充材料从未提到颜色量化**（已核对 arXiv v3 全文
与已发表 SI，"quantiz*" 只出现在有损视频那节）。论文 §2.1/§3.1 写的是
*"yielding next-pixel probability for each pixel"* 和 *"every 1024-pixel segment of
which is fed into iGPT"* —— **字面读是无损的**。有损性来自论文没写的那一步。

### 4.2 能不能"绕开 iGPT 的量化"？—— 不能保留 iGPT，但作者自己提供了无损路径

iGPT 的输入**就是**离散 token，没有连续输入变体，所以"关掉量化还用 iGPT"不成立。
三条出路：

| 路线 | 做法 | 本次是否实测 |
|---|---|---|
| **R1 换模型**：用作者自己的 bGPT 逐字节路径 | `compress_with_bgpt_or_llama.py --data_type bmp --model_path data/model/bgpt/weights-image.pth` | **已实测** ✅ |
| R2 补边信息：保留 iGPT 码流，再无损编码残差 | iGPT 码 + 残差信道 | 已算**下界** ✅ |
| R3 换字母表：对像素字节建自回归模型 | 即 R1 的一般化 | 未做 |

**重要发现**：本仓库里 `--data_type bmp` 这条无损路径**此前从未被跑过** ——
项目旧 smoke test 报 `Found 0 files`（`data/test_images` 里是 `.png`，而
`run.sh` 第 5 行建议的 `--data_type png` 代码里根本不支持，只会静默返回 0 个文件）。
我把测试图转成 BMP 后跑通了。
（`tools/lossless_image_bgpt_vs_codecs.py`，38 s，模型 110 084 352 参数）

### 4.3 实测结果：作者自己的无损路径 ≈ PNG，不优于经典基线

同一张 1714×1142 图（raw RGB = 5 872 164 B），同一分母：

| 方法 | 字节 | rate（÷raw RGB） | ratio |
|---|---|---|---|
| WebP lossless（PIL method 6） | 2 734 500 | **46.5672 %** | 2.147× |
| JPEG-2000 lossless（PIL） | 2 949 996 | 50.2369 % | 1.991× |
| PNG（PIL optimize, level 9） | 3 058 683 | 52.0878 % | 1.920× |
| **LMCompress-bgpt（作者，无损）** | **3 058 787.8** | **52.0896 %** | **1.9198×** |
| brotli q11（raw RGB） | 3 899 758 | 66.4109 % | 1.506× |
| TIFF deflate | 4 985 826 | 84.9061 % | 1.178× |

bGPT 与 PNG **相差 105 字节（0.003 %）** —— 实际上打平，且两者都输给 WebP lossless。
（bGPT 的分母含 BMP 的 54 B 头 + 2 284 B 行填充，计 0.04 % 的不利，可忽略。）

> **顺带修正项目里的一个数字**：旧基线表把 `png (ffmpeg cl100)` 记成 **84.3809 %**，
> 并据此推断"这张图比 CLIC 难得多"。我实测 ffmpeg 的 PNG 编码器确实给出
> 4 954 982 B（84.38 %），但 PIL/libpng level 9 只有 3 058 683 B（**52.09 %**），
> 与论文 CLIC2019 的 PNG 47.08 % 相当接近。**旧推断是 ffmpeg PNG 编码器弱造成的，
> 不是图像难度。** 后续引用该图对比时请用 52.09 %。

### 4.4 R2：把量化丢掉的信息补回来要多少钱 —— 下界 46.12 %

因为作者喂给 processor 的是单通道值构成的灰度三元组 `(v,v,v)`，而颜色量化是逐像素的，
映射 `t: {0..255} → 48 个 token` **只依赖 v**。我穷举得到 `t(v)`，并用**真实 chunk**
验证（1024 个位置，0 处不符），于是全图归约为该映射 + 值直方图。
边信息的精确下界是条件熵 `H(V|T)`：

| 量 | 值 |
|---|---|
| H(V\|T) | **2.4485 bits/sample** |
| 边信息下界 | 1 797 216 B = **30.6057 %** of raw RGB |
| iGPT 码流（全跑实测） | 910 788 B = 15.5103 % |
| **无损总计（下界）** | **2 708 004 B = 46.1159 %**（ratio 2.168×） |

对比：**WebP lossless 46.5672 %**、JPEG-2000 50.2369 %、PNG 52.0878 %。

**结论：把 iGPT 丢掉的那 30.6 % 信息补回来之后，它相对于经典无损编码器的优势
基本消失（46.12 % 下界 vs WebP 46.57 %），而 46.12 % 对 iGPT 还是乐观值**
（下界假设解码器已知精确条件分布；真实残差编码只会更贵）。
换句话说，论文 6.32× 的头条数字，很大一部分来自**扔掉了 30.6 % 的信息**。

### 4.5 视频同理

同一映射、同样逐像素，对 3 帧真实 4096×2160 帧计算：

| 帧 | 样本 | H(V\|T) | 边信息（÷raw RGB） |
|---|---|---|---|
| frame 0 | 26 542 080 | 2.5252 | 31.5649 % |
| frame 1 | 26 542 080 | 2.5248 | 31.5600 % |
| frame 2 | 26 542 080 | 2.5250 | 31.5623 % |
| 均值 | | 2.5250 | **31.5624 %** |

统一到 raw RGB 口径（1 592 524 800 B = 全片 60 帧）后，项目已复现的无损基线：

| 编码器 | 字节 | ÷raw RGB | 论文 high-res |
|---|---|---|---|
| h.265 crf 0 | 182 044 211 | **11.43 %** | 21.69 % |
| h.264 crf 0 | 293 551 029 | 18.43 % | 37.42 % |
| FFV1 | 319 836 906 | 20.08 % | 40.16 % |
| zstd -22 | 406 369 765 | 25.52 % | 51.04 % |
| **iGPT 边信息单独一项** | —— | **≥ 31.56 %** | 论文报 15.31 % |

**仅边信息一项（31.56 %）就已经超过全部四个经典无损编码器**，
所以视频这条路做成无损后没有任何竞争力。另外作者**没有提供**无损视频模型
（没有 `weights-video.pth`；`--data_type video` 只能借 image 或 audio 权重），
所以视频的可比无损基准就是上面这四个经典编码器。

### 4.6 可复现的实验设计（命令）

```bash
# R1 作者自己的无损图片路径（bmp 是唯一被支持的图片容器）
#    先把 PNG 转 BMP:  PIL.Image.open(png).convert('RGB').save('test_image.bmp')
python compress_with_bgpt_or_llama.py \
    --data_path repro_understanding_is_compression/artifacts/lossless_img/test_image.bmp \
    --data_type bmp \
    --model_path data/model/bgpt/weights-image.pth \
    --context_window 8160 --batch_size 1
# 判据：total compression ratio（越大越好），分母 = BMP 文件字节

# 经典无损基线（同一像素，同一分母 = W*H*3）
python repro_understanding_is_compression/tools/lossless_image_bgpt_vs_codecs.py

# R2 边信息下界
python repro_understanding_is_compression/tools/lossless_sideinfo_bound.py
```

**实验设计要点（避免重蹈口径覆辙）**

1. **分母统一到 raw RGB = W×H×3**，不要一边用 BMP 文件大小、一边用原始字节。
2. **iGPT 行不得进入无损对比表** —— 它与 PNG/WebP/JPEG-2000/FFV1/x265 不同类。
   要放就必须标注 LOSSY，或换成 R2 的"码流 + 边信息下界"。
3. **换同分布图像**：本 capsule 只有 1 张 4K 航拍图；论文用 CLIC2019/ILSVRC。
   正确做法是在 CLIC2019 上重跑 R1，与论文 SI Table 3 的 LMCompress-bgpt 28.57 %
   对齐（本机 52.09 % vs PNG 52.09 %，而论文 28.57 % vs PNG 47.08 %，
   这个**相对关系都还没对上**，值得单独查）。
4. **bGPT 路径报告的是理论码长**（`encoder_decoder(...)` 在源码里被注释掉了），
   严格说"无损"是按构造成立，尚未做往返验证。要坐实需接上算术编码器跑一次 round-trip。

---

## 5. 版本关系与本轮新增的发表版核对

### 5.1 时间线（先有预印本，后有发表版）

| 版本 | 时间 | 标识 |
|---|---|---|
| arXiv 预印本 *Understanding is Compression* | **2024-07** | arXiv:2407.07723 |
| Research Square 预印本 | **2024-09-06** | DOI 10.21203/rs.3.rs-4735292/v1 |
| **Nature Machine Intelligence 发表版**<br>*Lossless data compression by large models* | **2025-05-01** | Vol **7**, pp. **794–799**，DOI 10.1038/s42256-025-01033-7 |

**发表版比预印本晚约 8 个月**（比 arXiv 晚约 10 个月）。之前的核对只能用预印本 +
已发表 SI，现在补齐了发表版正文（`lit/nmi_published.pdf` / `.txt`，6 页）。

### 5.2 代码链接确认在发表版里

NMI 正文 Code availability：

> *"Our code is available via Code Ocean at https://doi.org/10.24433/CO.**9735997**.v1. (ref. 28)"*

⚠️ **待核对**：这个 DOI 的后缀是 **9735997**，而本项目的 capsule 目录名是
`capsule-**4957991**-code`。两者不一致。Code Ocean 对自动抓取返回 403，
本次无法打开确认。可能只是"capsule 内部 ID ≠ 发布 DOI 编号"，也可能发表版指向的是
**另一个/更新过的 capsule**。建议手工打开该 DOI 对照内容。

### 5.3 发表版与预印本的实质差异（影响此前的对比口径）

| 项 | 预印本（2024-09） | **发表版（2025-05）** |
|---|---|---|
| MeDAL 用量 | 前 **1104 MB** | 前约 **180 MB** |
| 文本分段 / 上下文 | 段长 **2048 bytes** | **1024** |
| 微调收益表述 | 明确给出 *"improves … by **8.5 %** on MeDAL and 38.4 % on Pile of Law"* | **删去数字**，只留定性说法"highlighting the effectiveness of domain-specific fine-tuning" |
| iGPT token 的性质 | 未提及 | 新增：*"**the tokens of iGPT are greyscale values**"* |

两点修正：

* **"正文 8.5 % vs SI 0.2 % 自相矛盾"这条只适用于预印本** —— 发表版把 8.5 % 删掉了。
  引用时请注明版本。
* **发表版内部仍有矛盾**：正文说文本 *"context length was set to **1,024** to fit the
  context window of LLaMA3-8B"*，而已发表 SI Table 9 用的是 MeDAL
  **8192 / 6144 / 4096**。同一篇发表物内部对不上。

### 5.4 最重要的收获：发表版给出了有损的**根因**，而且那个前提是错的

发表版 §Image compression（`lit/nmi_published.txt` 第 352–360 行）：

> *"**Because the tokens of iGPT are greyscale values**, we regard an image as three
> greyscale images, one for each RGB channel, and compress each channel independently.
> … During decompression, each channel is decompressed independently, and the three
> channels are then merged to **reconstruct the original image**."*

也就是论文的推理链是：
**"iGPT 的 token 是灰度值" → 所以按通道拆开逐通道压缩是精确的 → 三通道合并即可还原原图。**

这个前提**是错的**。我直接检查了 capsule 里 iGPT 的词表
（`data/model/imagegpt-small/preprocessor_config.json` 的 `clusters`）：

| 检查项 | 实测 |
|---|---|
| `clusters` 形状 | **(512, 3)** —— 512 个 **RGB 三元组** |
| 值域 | −0.989 … 0.997（归一化 RGB） |
| **恰好是灰度（R=G=B）的 cluster 数** | **0 / 512** |
| 接近灰度（\|R−G\|,\|G−B\| < 0.05）的 cluster 数 | 61 / 512 |
| 256 个灰度值实际命中的不同 token 数 | **48** |
| 那 48 个 token 是否都是近灰度 cluster | 是（48/48） |

结论：**iGPT 的 token 是"颜色"聚类中心，不是灰度值。** 代码把每条通道的 8-bit 值
拼成灰度三元组 `(v,v,v)` 送进这个 **512 色 RGB 调色板**，而 RGB 空间的"灰度对角线"
只被其中约 48–61 个 cluster 稀疏散布地覆盖 —— 这正是 256 个灰度值塌缩成 48 个 token
（585 个碰撞对）的**根本原因**。

**所以有损性不是"代码与论文不符"，而是论文自己的推理前提错误直接导致的。**
论文还明确声称解压后能 *"reconstruct the original image"* —— 在非单射映射下不可能。

> 顺带说明：如果 iGPT 的 token 真是"灰度值"且量化到 512 级，那么 256 个灰度值
> 是可以被单射表示的，论文的推理就成立了。**分歧只在于 token 到底是什么**，
> 而这一点用 3 行代码就能查实。


