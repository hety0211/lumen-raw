# 模型来源与转换

## 2.1.0 AI 移除、风光语义蒙版与点选物体

三个功能各用一个模型，都随包离线运行、只用 CPU（测试机 Ryzen 7 9800X3D 上比 DirectML 更快，单张图也不必等显卡编译）。文件、固定版本与 SHA-256 见 `assets/editing-assets.json`（`tools/fetch_assets.py` 按它下载），许可文本见 `assets/LaMa-LICENSE.txt` 与 `assets/EfficientViT-LICENSE.txt`。打包时权重转为 float16（`tools/compact_models.py`），四个文件由 410 MB 降到约 199 MB。

| 用途 | 模型与许可 | 文件 | 输入及输出 |
|---|---|---|---|
| AI 移除 | LaMa big-lama（Suvorov 等，WACV 2022；advimman/lama，代码与权重 Apache-2.0）。ONNX 取自 Hugging Face `smile-forge/big-lama-onnx`（Apache-2.0，固定版本 `4211d2c`）：傅里叶单元的 FFT 写成与余弦／正弦矩阵的乘积，权重未改动，与原模型相差约 8×10⁻⁵ | `inpaint-lama.onnx`，208,851,664 字节 | `rgb` [1, 3, H, W]（0–1，H、W 为 8 的倍数，64–4096）与 `mask` [1, 1, H, W]（1 为要补全处）；输出补全后的图像 |
| 点选物体 | EfficientViT-SAM L0（MIT Han Lab，Apache-2.0；建立在 Meta 的 Segment Anything 上，Apache-2.0），官方 ONNX，Hugging Face `mit-han-lab/efficientvit-sam`（固定版本 `a2f0c59`） | `object-encoder.onnx` 123,112,917 字节、`object-decoder.onnx` 16,470,362 字节 | 编码器：长边缩到 512 像素、补零到 512 × 512、按 ImageNet 均值方差（0–255 尺度）归一化；解码器：点击坐标（换算到长边 1024 的画面）与标签（1 加选、0 减选），输出 256 × 256 的蒙版 logits，对应补零后的 1024 画面；每个减选点再以单个加选点解码一次，得到的物体不含加选点时从结果中扣掉（只给点击时模型会在被减的物体上留下残片）。未补 Segment Anything 的“占位点”（标签 −1）：在面包上它去掉了残片，但让小猫爪子、希腊灯塔堤岸的减选失效，单击的选取范围也有变化 |
| 风光语义蒙版 | EfficientViT-Seg B2，ADE20K 150 类（MIT Han Lab，Apache-2.0），权重 `efficientvit_seg_b2_ade20k.pt`（Hugging Face `han-cai/efficientvit-seg` 固定版本 `cf3ccaf`，SHA-256 `fabfa716…b510`），用 `tools/convert_landscape.py` 导出（PyTorch 2.14.1 旧导出器，opset 17，动态宽高，与 PyTorch 最大相差 4.3×10⁻⁶）。上游没有 ONNX 版，导出的文件发布在 LUMEN RAW 自己的 GitHub Release | `landscape-seg.onnx`，61,391,343 字节 | 短边 512、长边按比例取 32 的倍数（最多 1536）的 RGB，ImageNet 归一化；输出 1/8 分辨率上五组的概率：150 类 softmax 后按组求和 |

- **AI 移除的做法（`lumen/remove.py`）：** 在原图（镜头校正后、应用了之前的修复）上取区域四周各留出与区域同样大小（至少 96 个工作像素）的范围，按 0.3% 高光取一个尺度，乘白平衡并编码为 γ 2.2 后缩到最长 1024 像素；要补全的区域向外扩 3 个工作像素，右下镜像补到 8 的倍数后推理。结果解码回白平衡之前的线性值，只保留区域的外接矩形，以预乘 alpha 的 16 位 RGBA PNG 存入配方（alpha 即覆盖：涂抹为向内羽化的笔划，点选为略外扩、边缘柔和的物体蒙版）。渲染时按画面大小重采样图块（缩小时先按面积平均）；放大使用时补回颗粒：添加时测得周围原图每通道的噪声强度（1 像素高斯残差的稳健标准差 ÷ √亮度），以及 LaMa 在工作尺寸上保留了多少噪声，缺少的部分以按照片固定种子的白噪声补上，因此整幅与任意分块得到同样的像素。
- **选择理由与对比：** 在 PlayRaw 样片（Skagen 灯塔与旗帜、希腊灯塔堤上的行人、面包、小猫、台北 101）上比较了 LaMa 与 MI-GAN（MIT，28 MB）：MI-GAN 在小猫处补出了红色的团块、旗杆处留下条纹，LaMa 在各处都补出了连续的天空、云、草和墙面，因此虽大 7 倍仍选 LaMa；LaMa 在 512 与 1024 工作尺寸上的结果相近，取 1024 以便大区域保留更多细节。点选模型比较了 EfficientViT-SAM L0、SAM 2.1 hiera-tiny（onnx-community）与 MobileSAM：在 6 个物体上 EfficientViT-SAM 的轮廓最完整（含小猫抬起的爪子、整座灯塔），SAM 2.1 tiny 漏掉了 Skagen 灯塔，MobileSAM 漏掉爪子且在 DirectML 上输出 NaN。风光分割比较了 EfficientViT-Seg B2 与 L1：在 13 张风光样片上 B2 同样准确（L1 把黑森林的晨雾当成水、在喜马拉雅湖面的倒影中标出建筑），而大小只有四分之一、速度快一倍。SegFormer（NVIDIA 非商业许可）与 MaskFormer（CC BY-NC）因许可未考虑。
- **验证：** Skagen 灯塔（2400 万像素 NEF，点选）去除后天空与云层连续，100% 查看没有接缝；补全区域的噪声（1 像素高斯残差的稳健标准差 ÷ √亮度）不补颗粒时只有周围的 6–10%，补回后与周围之比为 1.0（测试中的合成图要求在 0.6–1.6 之间）；分块渲染与整幅渲染逐像素相同。五组语义在 Beach（海、沙滩）、Taipei101（建筑）、Waterfall（水）等样片上按预期分区；点选在合成图上与真实物体的交并比大于 0.9。见 `tests/test_v210.py`。

## 2.0.0 选片提示：YuNet 人脸检测与 MediaPipe Face Mesh

选片的闭眼提示用两个小模型（合计约 2.7 MB，打包后），随包离线运行，推理代码在 `lumen/culling.py`，文件、固定版本与 SHA-256 见 `assets/culling-assets.json`（`tools/fetch_assets.py` 按它下载）。

| 用途 | 模型与许可 | 输入及输出 |
|---|---|---|
| 找人脸 | YuNet（Shiqi Yu，MIT），OpenCV Zoo 的 `face_detection_yunet_2023mar.onnx`，232,589 字节，由 OpenCV 的 `FaceDetectorYN` 运行（打包时保持 float32，不做 float16 转换） | 相机预览缩到长边 1280 像素；人脸框、5 个关键点（双眼、鼻尖、嘴角）和置信度（阈值 0.7） |
| 眼睛开合 | Google MediaPipe Face Mesh V2（Apache-2.0），`face_landmarker.task`（float16 v1，SHA-256 `64184e22…`）中的 `face_landmarks_detector.tflite`，由 naklitechie 用 tf2onnx 转为 ONNX（opset 17，权重未改动），4,920,995 字节；打包时权重转 float16（约 2.5 MB） | 以人脸为中心、按双眼连线转正的 256 × 256 RGB 裁切（0–1）；478 个关键点与人脸存在概率。先用 YuNet 的框，再按第一次的关键点重新裁切一次 |

- **闭眼判断：** 每只眼的张开程度 = 上下眼睑三对关键点的平均距离 ÷ 内外眼角距离（Face Mesh 索引 33/133、159/145、158/153、160/144，左眼对应 263/362 等）。张得较大的那只眼也低于 0.16 时判为闭眼。只统计长边 3% 以上、存在概率 0.5 以上的人脸，因此风光照片中的小人物不参与。
- **验证：** 在 7 张闭眼人像（Wikimedia Commons）和 11 张睁眼人像上：睁眼的张开程度为 0.19–0.54，没有被误判为闭眼；正面闭眼的 4 张（0.06–0.15）全部识别；侧躺、大角度仰头被树叶遮挡的 2 张没有检出人脸，低头的小脸（56 像素）判为睁眼。这是提示，不是判定：侧脸、大角度和遮挡的脸常会漏掉。曾试过 Intel Open Model Zoo 的 `open-closed-eye-0001` 小模型直接分类眼部裁切，它在普通照片上把一半闭眼判成睁眼（训练数据为红外近距离眼部图像），未采用。
- **清晰度不用模型：** 在 1600 像素灰度图的 32 像素方块中，比较四个方向上二阶导数与一阶导数的能量，取结构最多的四分之一方块中第 90 百分位、四个方向中最低的一个，换算为 0–100。这一比值随模糊下降而与反差无关，雾天低反差的照片不会被当成模糊；取最低方向能发现单方向的运动模糊。

## 1.5.0 语音识别：SenseVoice-Small

1.8.0 起模型不再随包，第一次使用语音时下载（约 240 MB，`lumen/speech.py`）；2.0.0 起除 Hugging Face 与 hf-mirror.com 外，同样的文件也放在 LUMEN RAW 自己的 GitHub Release（`assets/speech-assets.json` 的 `releases`），上游仓库失效时仍可下载。

“自然语言输入”的语音由 **SenseVoice-Small**（FunAudioLLM / 阿里巴巴通义实验室，`iic/SenseVoiceSmall`）离线识别，支持中文、英文与中英混说。文件取自 k2-fsa / sherpa-onnx 的 ONNX 导出（Hugging Face `csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17`，固定版本 `2365baea`），未做修改：`model.int8.onnx`（239,233,841 字节，MatMul 权重 QUInt8 动态量化）与 `tokens.txt`（25,055 个 SentencePiece 词元）。校验值见 `assets/speech-assets.json`，许可为 FunASR 模型开源协议 1.1（`assets/SenseVoice-LICENSE.txt`，要求注明出处并保留模型名称）。

- **前端（`lumen/speech.py`，NumPy）：** 16 kHz 单声道、int16 幅度；Kaldi fbank：25 ms 帧长、10 ms 帧移、去直流、预加重 0.97、Hamming 窗、512 点 FFT 功率谱、20 Hz–8 kHz 共 80 个 Mel 三角滤波（1127·ln(1+f/700) 刻度，不含奈奎斯特频点）、取自然对数（下限 float32 eps），边缘截断。之后按模型元数据做 LFR（7 帧堆叠、步长 6，得到 560 维）和 CMVN（`neg_mean` / `inv_stddev`）。
- **推理：** 输入 `x`、`x_length`、`language`（默认自动，可在设置中固定中文或英文）、`text_norm`（`with_itn`，输出带标点与阿拉伯数字）。输出前 4 帧为语言、情感、事件与规范化标记，其后 CTC 贪心解码，去掉空白与 `<|…|>` 标记。ONNX Runtime CPU 执行设备，4 线程。
- **验证：** 模型自带的中文样例识别为“开放时间早上9点至下午5点。”、英文样例与参考文本一致；Windows 语音合成的中文“把天空调得更蓝一点，整体稍微暖一些”和英文 “Make the sky bluer and lift the shadows a little” 逐字正确（`tests/test_v15.py` 在 Windows 上自动合成测试语音）。6 秒语音约 0.08 秒（本机 CPU），首次加载约 0.8 秒。

语言模型不随包分发：自然语言修图使用用户自己选择的本地运行器（Ollama、LM Studio、llama.cpp 等）或云端 API。

## 1.3.1 macOS：Metal 内核与 Core ML

Mac 版不新增、也不重新转换任何模型，九个 ONNX 文件及校验值与 Windows 版完全相同。

- **逐像素显影：** `lumen/metal.py` 用 Metal Shading Language 实现与 `gpu_graphs` 相同的 `tonal`、`color`、`fused` 三个阶段，直接使用 `gpu_graphs.tonal_inputs` / `color_inputs` 生成的参数与查找表（1° HSL 表、4097 点曲线表），中性的色彩混合、恒等曲线和零强度分级与 CPU 路径一样跳过。着色器在运行时编译，关闭快速数学并对 `pow` / `exp2` / `sin` 使用精确版本；首次使用时以随机图与 NumPy 参考比较，最大误差超过 1×10⁻⁴ 即停用 Metal。像素按每段约 48 MiB 分条进入共享缓冲（统一内存，无 PCIe 传输），因此 4 亿像素图也只占用固定大小的 GPU 缓冲。
- **神经网络：** 通过 ONNX Runtime 1.30 的 Core ML 执行设备（MLProgram 格式，默认 `CPUAndGPU`，即 Apple GPU 上的 Metal）运行。超分与去杂色模型的宽高是动态的；Core ML 以动态形状编译时无法为 DRUNet 生成执行计划（“Error in building plan”），NAFNet 被拆成 80 段、比 CPU 更慢。因此 `compute.ShapedSessions` 在 Core ML 上按实际分块尺寸固定 `height` / `width` 后编译，每个模型保留最近使用的 4 种尺寸。输入不做额外补边，结果与动态模型一致。
- **一致性（M1 Pro，与 ONNX Runtime CPU 同一输入比较的最大绝对误差）：** Real-ESRGAN x4plus 1.2×10⁻⁶、realesr-general-x4v3 2.3×10⁻⁶、DRUNet 7.5×10⁻⁷、NAFNet 8.3×10⁻⁶、FFDNet 2.0×10⁻⁶、SkySeg 1.8×10⁻⁷、人物 6.1×10⁻⁸、U2NetP 1.9×10⁻⁶；MiDaS 相对深度 2.3×10⁻³（输出为未归一化的逆深度，之后按分位数归一化）。

## 1.3.0 GPU 逐像素图

1.3.0 不新增神经网络模型，九个 ONNX 模型及其校验值保持不变。`lumen/gpu_graphs.py` 在运行时生成三张不含权重的逐像素计算图（`tonal`、`color`、`fused`），用于 DirectML 显影；`lumen/onnx_graph.py` 直接写出 ONNX protobuf，因此运行和构建都不需要 `onnx` 包。图采用 opset 13、通道在后（1 × H × W × 3）的 float32 输入。

- `tonal`：白平衡增益、曝光、阴影／高光／黑色／白色与对比度，公式与 NumPy 版一致。
- `color`：饱和度／自然饱和度、八色 HSL（1° 查找表线性插值，节点均在整数角度上，因此与 `numpy.interp` 相同）、RGB 与单通道曲线（4097 点查找表，与平滑曲线的采样轴相同；折线模式重采样误差约 2×10⁻⁵）、黑白与三段色彩分级。
- `fused`：`tonal` 后接 `color`，未启用空间细节工具时一次完成。

CPU 路径仍是参考实现；`tests/test_v13_pipeline.py` 以 ONNX Runtime CPU 执行这些图并与 NumPy 结果比较。`python tools/build_tonal_dml.py 文件夹` 可导出三张图供 Netron 等工具查看。

## 1.2.1 照片合成

三种合成均使用已随软件打包的 OpenCV，不下载额外神经网络权重。HDR 使用 `createMergeMertens(1, 1, 1)`，配合曝光匹配和基于参考帧的运动区域替换；景深合成使用多尺度拉普拉斯清晰度与边界羽化；全景使用 SIFT、OpenCV detail 相机估计与优化、自行实现的分块球面投影和曝光补偿。依赖许可沿用 `THIRD_PARTY.md` 及 OpenCV 许可文件。

这些是 CPU 图像算法，不能称为 Adobe HDR / AI 景深算法。本版保留 1.2 的九个 ONNX 模型及其校验值，未更改 AI 超分和去杂色模型。

## 1.2 高画质增强

新增模型记录在 `assets/models/v12-models.json`，全部来源于上游官方权重。转换代码见 `tools/restoration_arch.py` 与 `tools/convert_restoration.py`；原始权重先验证 SHA-256，以 `torch.load(weights_only=True)` 读取并严格匹配网络。PyTorch / ONNX 仅开发转换使用，不打包进运行版。

| 用途 | 模型 | 参数数量 | 许可 |
|---|---|---:|---|
| 默认高画质超分 | Real-ESRGAN x4plus，23 RRDB | 16,697,987 | 权重 BSD-3-Clause；BasicSR 结构 Apache-2.0 |
| 默认高画质去杂色 | DRUNet color，带噪声条件的残差 U-Net | 32,640,960 | MIT，Kai Zhang |
| 可选真实噪声去杂色 | NAFNet SIDD width32 | 29,159,715 | MIT，Megvii |

官方权重：
- https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth
- https://github.com/cszn/KAIR/releases/download/v1.0/drunet_color.pth
- NAFNet 官方 README 所列 Google Drive ID `1lsByk21Xw-6aW7epCwOQxvm6HYCQZPHZ`，https://github.com/megvii-research/NAFNet

ONNX opset 17，动态尺寸。对 32×48、35×43 输入与原始 PyTorch 比较，RRDB 最大绝对误差 7.75e-7、DRUNet 3.58e-7、NAFNet 3.10e-6。DRUNet / NAFNet 内部动态补边后裁回输入尺寸。

RRDB 采用 128px 分块、32px 重叠和 32px 上下文；降噪为 256px 分块、64px 重叠、32px 上下文。重叠处余弦羽化并归一化，输出大于 256MiB 时使用磁盘映射。高画质局部预览只计算覆盖中央区域的块，网格、上下文和噪声参考均与整图一致。有限上下文仍不能保证与无限制整图神经推理完全相同。

DRUNet 使用原图九区 Haar 噪声估计，强度 35 为基准，sigma 上限 50/255。NAFNet 为 SIDD 真实照片噪声训练的固定模型，使用噪声参考调节结果与原图混合，默认强度 70。旧 FFDNet 和 compact 超分仍可选。算法并非对所有照片都更优；测试中 DRUNet 与 FFDNet 的合成噪声指标相近，NAFNet 在该样例残留噪点更多，详见测试报告。

所有模型处理显影后的 RGB；不等同 Adobe RAW 域增强。推理线程上限 32，实际取本机逻辑核心数与 32 的较小值；单一重图像任务运行，ONNX 跨算子线程 1。CPU 推理已验证，CUDA provider 接口保留但本轮没有 GPU 实测。

## 1.1 近景与去杂色

新增模型校验值见 `assets/models/v11-models.json`，均可离线运行：

- **MiDaS v2.1 small**：Intel ISL，MIT；官方 ONNX https://github.com/isl-org/MiDaS/releases/download/v2_1/model-small.onnx 。256×256 RGB，ImageNet 归一化，输出相对逆深度（越大越近）。用 10%–95% 分位归一化，平滑阈值选近处，再用引导滤波对齐边缘。不是绝对距离测量。
- **FFDNet color**：KAIR / Kai Zhang，MIT；官方权重 https://github.com/cszn/KAIR/releases/download/v1.0/ffdnet_color.pth 。12 层卷积、96 个特征通道，PixelUnshuffle / PixelShuffle，RGB 和噪声标准差两个输入。`tools/convert_denoise.py` 使用 PyTorch 2.14.0 导出 opset 17 动态尺寸 ONNX，权重加载使用 `weights_only=True`，转换最大绝对误差约 1.01e-6。

去杂色使用 256px 分块与 32px 上下文、偶数坐标对齐；输入奇数宽高时复制补边。噪声参考取全图九个原始像素区块的 Haar 高频 MAD 估计，预览和整图使用同一参考，35 为基准强度。最终噪声标准差上限 75/255。该模型针对 RGB 高斯型噪声训练，真实复杂噪声和纹理需要人工检查，不是 RAW 马赛克域降噪。

主体复用 U2NetP 前景概率，背景使用其反选，无额外下载。1.1 包含六个 ONNX 文件，1.2 增加三个，共九个；PyTorch 仅用于转换，不进入运行包。


## 1.0 自动蒙版

三个选区模型随包离线运行，推理代码在 `lumen/selection.py`，文件 SHA-256 见 `assets/models/selection-models.json`。

| 用途 | 模型与许可 | 输入及输出 |
|---|---|---|
| 天空 | Sky-Segmentation-and-Post-processing / SkySeg，MIT；ONNX https://huggingface.co/JianyuanWang/skyseg/resolve/main/skyseg.onnx | RGB 320×320，取首个天空概率输出 |
| 人物 | TorchVision DeepLabV3 MobileNetV3 Large，BSD-3-Clause，COCO_WITH_VOC_LABELS_V1 权重 | RGB 384×384，21 类 softmax 的 person 类（15） |
| 背景 | U2NetP 显著前景模型，Apache-2.0；ONNX https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2netp.onnx | RGB 320×320，显著前景概率取反 |

天空和 U2NetP 使用官方预处理方式：按最大输入强度归一化，再使用 ImageNet 均值／标准差；人物模型直接使用 ImageNet 归一化。概率图回到预览尺寸后使用引导滤波细化边缘，保存为 PNG 软蒙版，导出时映射到全尺寸。AI 选区可人工增减，不保证发丝和半透明边缘完全准确。

人物权重来自 https://download.pytorch.org/models/deeplabv3_mobilenet_v3_large-fc3c493d.pth 。用 `tools/convert_person.py` 转换为 opset 17 固定尺寸 ONNX。模型转换需要 PyTorch 2.14.0、TorchVision 0.29.0、ONNX 1.23.0 和 ONNX Runtime 1.30.0，仅开发者转换时需要，应用运行不包含 PyTorch。

示例：`python tools/convert_person.py C:\Models\person-deeplab.onnx`。脚本从 TorchVision 官方源下载权重，并比较 PyTorch / ONNX 概率输出。背景是显著前景的反选，不是通用实例分割；多人图的人物蒙版包含所有识别到的人，不逐个人物编号。

## 超分

运行 LUMEN RAW 不需要安装 PyTorch，也不需要另外下载权重。

随包提供的 `assets/models/realesr-general-x4v3.onnx` 来自 Real-ESRGAN 官方 v0.2.5.0 发布的 `realesr-general-x4v3.pth`，使用 BSD-3-Clause 许可。网络结构为 BasicSR SRVGGNetCompact：64 个特征通道、32 个中间卷积、PReLU、4× PixelShuffle，加最近邻放大的残差。

- 官方权重：https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth
- 原权重 SHA-256：`8dc7edb9ac80ccdc30c3a5dca6616509367f05fbc184ad95b731f05bece96292`
- ONNX SHA-256：`abd38ff81ca7cd9e03e213394267c50d393c8021b64f2f57b818f1ec82ad2369`
- 转换：PyTorch 2.14.0 的 legacy ONNX exporter，opset 17，动态宽高，RGB NCHW float32 输入，范围 0–1，原生 4× 输出。
- 对同一固定输入比较原始 PyTorch 模型与 ONNX CPU 输出，最大绝对误差 `8.9407e-7`。
- 2× 模式在每个 4× 输出块上做面积降采样；分块输入为 192px，上下文为 40px。自动测试对比整图和分块真实推理，误差小于 `2e-5`。

网络生成的高频纹理不一定是真实细节。软件的增强处理发生在 RAW 显影后的 RGB 阶段，不能代替 RAW 传感器级细节重建，也不是 Adobe 增强算法。

开发者可使用 `tools/convert_model.py` 重新转换。先在单独环境安装 `torch==2.14.0`、`onnx==1.23.0`、`onnxruntime==1.30.0`，从上述官方地址下载权重，再运行：

```powershell
python tools/convert_model.py C:\Models\realesr-general-x4v3.pth C:\Models\converted.onnx
```

转换脚本验证输入权重的 SHA-256，并使用 `torch.load(weights_only=True)`。此脚本不是应用运行时的一部分。Real-ESRGAN、BasicSR 的许可证已放在 `assets` 目录中。
