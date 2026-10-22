# 第三方依赖

LUMEN RAW 的源码许可不替代依赖许可。便携包保留动态库，Qt / PySide 可替换；应用源码和构建脚本同时交付。

| 组件 | 主要许可 / 来源 |
|---|---|
| Python | PSF，https://www.python.org/ |
| NumPy | BSD，https://numpy.org/ |
| OpenCV | Apache-2.0 等，https://opencv.org/ |
| Pillow | HPND 等，https://python-pillow.org/ |
| rawpy | MIT，https://github.com/letmaik/rawpy |
| LibRaw | LGPL-2.1 / CDDL 双许可，https://www.libraw.org/ |
| PySide6 / Qt / Shiboken（含 1.5.0 起用于麦克风录音的 Qt Multimedia 及其 FFmpeg 插件） | LGPL-3.0 / GPL / 商业许可及组件自身许可，https://www.qt.io/ |
| FFmpeg 动态库（Qt Multimedia 随附，1.5.0 起打包：avcodec、avformat、avutil、swresample、swscale） | LGPL-2.1 或更高版本，https://ffmpeg.org/ ；动态链接，可替换 |
| tifffile | BSD-3-Clause，https://github.com/cgohlke/tifffile |
| Noto Sans SC | SIL Open Font License 1.1，https://github.com/google/fonts/tree/main/ofl/notosanssc |
| ONNX Runtime / FlatBuffers | MIT / Apache-2.0，https://github.com/microsoft/onnxruntime / https://github.com/google/flatbuffers |
| Real-ESRGAN 摄影权重 | BSD-3-Clause，https://github.com/xinntao/Real-ESRGAN/releases/tag/v0.2.5.0；assets/RealESRGAN-LICENSE.txt |
| BasicSR 网络结构参考 | Apache-2.0，https://github.com/XPixelGroup/BasicSR；assets/BasicSR-LICENSE.txt |
| SkySeg 天空模型 | MIT，https://github.com/xiongzhu666/Sky-Segmentation-and-Post-processing；ONNX 发布 https://huggingface.co/JianyuanWang/skyseg；assets/SkySeg-LICENSE.txt |
| TorchVision DeepLabV3 MobileNetV3 人物模型 | BSD-3-Clause，https://github.com/pytorch/vision；assets/TorchVision-LICENSE.txt |
| U-2-Net / U2NetP 显著前景模型 | Apache-2.0，https://github.com/xuebinqin/U-2-Net；ONNX 发布 https://github.com/danielgatis/rembg/releases/tag/v0.0.0；assets/U2NET-LICENSE.txt |
| Inno Setup 安装程序（含 1.5.1 起使用的简体中文、繁体中文、日语、韩语、德语、法语、西班牙语、俄语安装界面翻译） | Inno Setup 自身许可，允许商业与非商业使用；https://jrsoftware.org/；licenses/Inno-Setup-LICENSE.txt |
| ExifTool 13.59 与随附 Perl | Perl 相同条款（Artistic / GPL）及依赖自身许可；https://exiftool.org/；完整许可在 assets/exiftool/exiftool_files/ |
| MiDaS v2.1 small 相对深度模型 | MIT，Intel ISL，https://github.com/isl-org/MiDaS；assets/MiDaS-LICENSE.txt |
| KAIR / FFDNet 彩色去杂色 | MIT，Kai Zhang，https://github.com/cszn/KAIR；assets/KAIR-LICENSE.txt |
| DRUNet color | MIT，Kai Zhang，https://github.com/cszn/DPIR；assets/DRUNet-LICENSE.txt |
| NAFNet SIDD | MIT，Megvii，https://github.com/megvii-research/NAFNet；assets/NAFNet-LICENSE.txt |
| lensfun 镜头数据库（1.5.1，镜头校正：畸变、横向色差、暗角） | CC BY-SA 3.0，lensfun 项目，https://lensfun.github.io/ ；原样收录于 assets/lensfun/db（提交 bbd4332），许可见 assets/lensfun/COPYING.CC_BY-SA_3.0；未包含或链接 lensfun 程序库（LGPL-3.0），校正模型由 lumen/lens.py 实现 |
| PyAV 19.0.1（2.0.0，延时摄影视频的编码） | BSD-3-Clause，https://github.com/PyAV-Org/PyAV |
| FFmpeg 9.0.2 动态库（随 PyAV 的 av.libs 打包：avcodec、avformat、avutil、avfilter、avdevice、swscale、swresample） | LGPL-3.0-or-later（按 PyAV 的构建配置），https://ffmpeg.org/ ；构建脚本与对应源码 https://github.com/PyAV-Org/pyav-ffmpeg ；动态链接，可替换 |
| x264、x265（随 PyAV 的 FFmpeg 打包的 H.264／H.265 编码库） | GPL-2.0-or-later，https://www.videolan.org/developers/x264.html 、https://www.x265.org/ ；以独立动态库形式随包分发，源码见上方 pyav-ffmpeg 构建脚本所引用的版本。安装包因此包含 GPL 组件；LUMEN RAW 自身仍为 MIT 许可，源码公开 |
| SVT-AV1、libvpx、dav1d、Opus、LAME、libwebp、libvmaf、oneVPL、zlib、libiconv 及 MinGW 运行库（随 PyAV 的 FFmpeg 打包） | 各自许可：BSD-3-Clause-Clear 与 AOM 专利许可（SVT-AV1）、BSD-3-Clause（libvpx、Opus、libwebp）、BSD-2-Clause（dav1d）、LGPL-2.0-or-later（LAME）、BSD-2-Clause-Patent（libvmaf）、MIT（oneVPL）、zlib、LGPL-2.1-or-later（libiconv）、GPL 附运行库例外（libgcc／libstdc++）与 MIT／ZPL（winpthreads） |
| YuNet 人脸检测模型（2.0.0，选片的闭眼提示） | MIT，Shiqi Yu，https://github.com/ShiqiYu/libfacedetection.train ；ONNX 取自 OpenCV Zoo `face_detection_yunet_2023mar.onnx`（提交 f12e127），https://github.com/opencv/opencv_zoo ；assets/YuNet-LICENSE.txt |
| MediaPipe Face Mesh V2 面部关键点模型（2.0.0，选片的闭眼提示） | Apache-2.0，Google，https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker （`face_landmarker.task` float16 v1 中的 `face_landmarks_detector.tflite`）；ONNX 格式转换 naklitechie/face-landmarks-onnx（Apache-2.0，权重未改动），https://huggingface.co/naklitechie/face-landmarks-onnx ；assets/MediaPipe-LICENSE.txt |
| LaMa big-lama 图像补全模型（2.1.0，AI 移除） | Apache-2.0，Suvorov 等 / advimman，https://github.com/advimman/lama ；ONNX 版 smile-forge/big-lama-onnx（Apache-2.0，权重未改动，固定版本 4211d2c），https://huggingface.co/smile-forge/big-lama-onnx ；assets/LaMa-LICENSE.txt |
| EfficientViT-SAM L0 与 EfficientViT-Seg B2（ADE20K）（2.1.0，点选物体与风光语义蒙版） | Apache-2.0，MIT Han Lab，https://github.com/mit-han-lab/efficientvit ；ONNX 取自 https://huggingface.co/mit-han-lab/efficientvit-sam（固定版本 a2f0c59），分割模型由 tools/convert_landscape.py 从 https://huggingface.co/han-cai/efficientvit-seg（固定版本 cf3ccaf）的权重导出；EfficientViT-SAM 基于 Meta 的 Segment Anything（Apache-2.0）；assets/EfficientViT-LICENSE.txt |
| SenseVoice-Small 语音识别模型（1.5.0） | FunASR 模型开源协议 1.1，FunAudioLLM / 阿里巴巴集团，https://huggingface.co/FunAudioLLM/SenseVoiceSmall；ONNX 导出 k2-fsa / sherpa-onnx（Apache-2.0），https://huggingface.co/csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17；assets/SenseVoice-LICENSE.txt |
| 相机与镜头品牌标志（1.6.2，水印边框） | 来自 Wikimedia Commons，均标为公有领域（简单字形的文字标志不受著作权保护），同时是各品牌所有者的注册商标；仅用于在照片边框上按 EXIF 标注拍摄所用的机身与镜头，不表示与这些公司有任何关联或获得认可；原始 SVG、文件名、来源页面与 SHA-256 见 assets/logos/src、assets/logos/README.txt 与 assets/logos/logos.json |
| 可选 CuPy | MIT，https://cupy.dev/ |
| PyObjC（macOS 版，调用 Metal） | MIT，https://github.com/ronaldoussoren/pyobjc |
| ExifTool 13.59 Perl 发行版（macOS 版） | 与 Perl 相同条款（Artistic / GPL），https://exiftool.org/；许可说明见随包 assets/exiftool/unix/README，由系统自带 Perl 运行 |
| python-build-standalone CPython 3.12（macOS 版构建所用 Python） | PSF 及其组件自身许可，https://github.com/astral-sh/python-build-standalone |
| PyInstaller 引导程序 | GPL-2.0 及引导程序例外条款，https://pyinstaller.org/ |
| dmgbuild（仅生成 DMG 时使用，不随包分发） | MIT，https://github.com/dmgbuild/dmgbuild |

安装依赖附带的许可证文本收集在便携目录 `licenses/`。字体许可证在 `assets/OFL.txt`。测试用的 Sony ARW 样片来自 https://raw.pixls.us/，记录为 CC0，仅用于验证，未将大体积原始样片加入便携包。

自 1.6.2 起，水印边框内置上表所列的相机与镜头品牌标志（商标归各自所有者）；没有内置标志的品牌仍以 Noto 字体排印品牌名称。1.6.1 及更早由用户导入的标志 PNG 保存在各自工程中，继续按原样显示。新增真实 RAW 测试照片来自 raw.pixls.us 的 CC0 样片库，不随包分发。
