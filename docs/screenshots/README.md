# Screenshot credits / 截图样片来源

All screenshots show the actual LUMEN RAW desktop application on Windows (DirectML). The interface is part of the MIT-licensed project. The visible photographs are CC0 test samples from [raw.pixls.us](https://raw.pixls.us/); the original RAW files are not included in this Git repository.

| Screenshot | Version and content | Photographs |
|---|---|---|
| `lens.png` (Chinese) and `workspace-en.png` (English) | 1.5.1. The Crop · Lens panel identified the Nikon Nikkor Z 14-24mm f/2.8 S from EXIF (24 mm, f/8) and corrects distortion, vignetting and lateral chromatic aberration with its lensfun profile. | Main preview and first thumbnail: Nikon Z 8 (`Nikon_Z8_raw_14_bit_lossless_compression.NEF`); second thumbnail: Sony A7R V (`7RM5-LosslessCompressedLarge.ARW`); third thumbnail: Canon EOS R5 Mark II (`R5m2_RAW.CR3`) |
| `library.png` (Chinese) and `library-en.png` (English) | 1.6.0. Filmstrip ratings: stars, the pick flag, a rejected photo (dimmed) and color labels, with the rating bar and filters in the toolbar. The status bar shows an edit applied by an AI assistant through the control channel (message set for the screenshot). | Main preview and first thumbnail: Nikon Z5 II (`Nikon_Z5_2_Lossless_compression_FX_1.NEF`); then Nikon Z 8, Sony A7R V, Canon EOS R5 Mark II and Nikon Z50 II (`Nikon_Z50_2_DSC_0060_DX_Lossless.NEF`) |
| `process2-sky.jpg`, `process2-highlights.jpg` | 1.6.0. The same edits rendered with process version 1 (left) and 2 (right). | Nikon Z5 II; Canon EOS R5 Mark II (`R5m2_RAW.CR3`) |
| `workspace.png` | 1.5.0. A real natural-language edit: the instruction “暖一点，暗部提亮，颜色更鲜艳一些” (warmer, brighter shadows, more vivid colors) was answered by qwen3-14b running locally in LM Studio, with thinking off. | raw.pixls.us records 3580 (Nikon Z 6, main preview and first thumbnail), 4515 (Canon EOS R) and 865 (Fujifilm X-T2) |

截图均为 LUMEN RAW 在 Windows 上的实际运行界面。`library.png`（中文）和 `library-en.png`（英文）为 1.6.0：底部图集的星级、留用旗标、排除（变暗）与色标，工具栏中的评级与筛选；状态栏显示 AI 助手通过控制通道完成的一步调整（截图时设置的提示文字）。`process2-sky.jpg` 与 `process2-highlights.jpg` 为同一组调整分别用处理版本 1（左）和 2（右）渲染。`lens.png`（中文）和 `workspace-en.png`（英文）为 1.5.1：「裁切·镜头」面板按 EXIF 识别出尼康 Z 14-24mm f/2.8 S（24 mm、f/8），并用 lensfun 配置文件校正畸变、暗角与横向色差。`workspace.png` 为 1.5.0：左侧是一次真实的自然语言修图，指令“暖一点，暗部提亮，颜色更鲜艳一些”由本机 LM Studio 运行的 qwen3-14b（关闭思考）返回参数。展示的照片都来自 raw.pixls.us 的 CC0 公共测试样片；仓库不包含这些 RAW 原始文件。
