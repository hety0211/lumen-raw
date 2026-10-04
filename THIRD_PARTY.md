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
| SenseVoice-Small 语音识别模型（1.5.0） | FunASR 模型开源协议 1.1，FunAudioLLM / 阿里巴巴集团，https://huggingface.co/FunAudioLLM/SenseVoiceSmall；ONNX 导出 k2-fsa / sherpa-onnx（Apache-2.0），https://huggingface.co/csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17；assets/SenseVoice-LICENSE.txt |
| 可选 CuPy | MIT，https://cupy.dev/ |
| PyObjC（macOS 版，调用 Metal） | MIT，https://github.com/ronaldoussoren/pyobjc |
| ExifTool 13.59 Perl 发行版（macOS 版） | 与 Perl 相同条款（Artistic / GPL），https://exiftool.org/；许可说明见随包 assets/exiftool/unix/README，由系统自带 Perl 运行 |
| python-build-standalone CPython 3.12（macOS 版构建所用 Python） | PSF 及其组件自身许可，https://github.com/astral-sh/python-build-standalone |
| PyInstaller 引导程序 | GPL-2.0 及引导程序例外条款，https://pyinstaller.org/ |
| dmgbuild（仅生成 DMG 时使用，不随包分发） | MIT，https://github.com/dmgbuild/dmgbuild |

安装依赖附带的许可证文本收集在便携目录 `licenses/`。字体许可证在 `assets/OFL.txt`。测试用的 Sony ARW 样片来自 https://raw.pixls.us/，记录为 CC0，仅用于验证，未将大体积原始样片加入便携包。

水印的内置品牌名称使用 Noto 通用字体排印，并非官方图形标志。没有随包再分发相机或镜头品牌图形 LOGO；自定义标志由用户提供。新增真实 RAW 测试照片来自 raw.pixls.us 的 CC0 样片库，不随包分发。
