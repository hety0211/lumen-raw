# Lightroom 预设与 LUT（1.7.0）

LUMEN RAW 可以导入 Lightroom Classic／Camera Raw 的修图预设和 `.cube` LUT，放进左侧「预设」面板，像内置风格一样带强度套用、批量套用到选中的照片。

**这是近似换算。** LUMEN 的滑块不是 Adobe 的：处理版本 2 在场景线性光中做曝光与局部色调映射，色彩在 OkLCh 中计算，Adobe 的滑块曲线也没有公开。同一个预设在 LUMEN 中的效果接近 Lightroom，但不会完全一致；需要时导入后再微调。

## 能导入的文件

| 文件 | 说明 |
|---|---|
| `.xmp` | Lightroom Classic 7.3 起、Camera Raw 的预设（`crs:` 命名空间，属性与元素两种写法都能读）。照片的 XMP 附属文件也能当预设导入。 |
| `.lrtemplate` | Lightroom 1–7 的修图预设（Lua 表）。由内置的小型解析器读取，**不会执行**文件中的任何代码；不是修图预设（过滤器、关键字、导出预设等）的模板会被拒绝。 |
| `.zip` | 预设包，可包含以上两种以及 `.cube`、`.lumenpreset`。Windows 压缩工具写入的 GBK 中文文件名能正确读取；同一个预设同时有 `.xmp` 和 `.lrtemplate` 两份时只导入 `.xmp`；`__MACOSX`、DNG（Lightroom 移动版预设）、嵌套的 zip 会跳过并在报告中说明。最多 4000 个文件、1 GB。 |
| `.cube` | LUT，导入为“只设置 LUT”的预设。 |
| `.lumenpreset` | LUMEN 自己的预设。 |

导入的预设保存为 `.lumenpreset`，放在数据目录的 `presets` 文件夹（Windows：`%LOCALAPPDATA%\LUMEN RAW\presets`），下次启动仍在。分组取预设自带的 `crs:Group`，否则取预设包或所在文件夹的名称。再次导入同一个包会覆盖原来的文件，不会重复。Adobe 配置文件（`crs:PresetType` 为 Look／Profile 的 `.xmp`）不是修图预设，无法导入。

## 预设只改变它包含的设置

与 Lightroom 相同，预设只改变文件中出现的设置：一个只含颗粒的预设不会把曝光、曲线归零。强度 0–100 在“中性值”与预设值之间插值（与内置风格相同）；裁切、蒙版、修复和取样的白平衡保持不变。1.6 及更早保存的 `.lumenpreset`（格式 1）仍是“完整风格”，套用时会替换全部调色参数，与之前一致。

## 映射表

“系数”是 LUMEN 数值 ÷ Adobe 数值，由 `tools/calibrate_lightroom.py` 在测试照片上拟合（见下文“校准”）；结果超出 LUMEN 的范围时截到 ±100。

| Adobe（crs:） | LUMEN | 系数 | 说明 |
|---|---|---|---|
| `Exposure2012` | 曝光 | 1 | 单位都是 EV |
| `Contrast2012` | 对比度 | 0.35 | |
| `Highlights2012` | 亮部 | 1.8 | |
| `Shadows2012` | 暗部 | 0.75 | |
| `Whites2012` | 白色 | 1.55 | |
| `Blacks2012` | 黑色 | 1.6 | |
| `Texture` | 纹理 | 0.95 | |
| `Clarity2012` | 清晰度 | 0.45 | |
| `Dehaze` | 去薄雾 | 0.65 | |
| `Vibrance` | 自然饱和度 | 0.7 | |
| `Saturation` | 饱和度 | 0.95 | −100 两边都是黑白 |
| `WhiteBalance`＝Custom 等 + `Temperature` | 目标色温（K） | 1 | 只对有相机色温的 RAW 照片生效（Lightroom 对 JPEG 同样不套用），按 mired 插值强度，同时清除取样的白平衡增益 |
| `Tint`／`IncrementalTint` | 色调 | 1.35 | 绝对色调按相对值套用 |
| `IncrementalTemperature` | 冷暖 | 1 | JPEG／TIFF 预设 |
| `WhiteBalance`＝As Shot | 冷暖、色调归零 | | |
| `Hue/Saturation/LuminanceAdjustment{Red…Magenta}` | 八色 HSL 的色相／饱和度／明度 | 0.5／0.9／0.6 | 八种颜色一一对应 |
| `ConvertToGrayscale` | 黑白 | | |
| `GrayMixer{Red…Magenta}` | 黑白时的八色明度 | 0.6 | 黑白时 Lightroom 不使用 HSL，LUMEN 同样忽略 |
| `SplitToningShadowHue/Saturation`、`SplitToningHighlightHue/Saturation`、`ColorGradeMidtoneHue/Sat` | 色彩分级 暗部／高光／中间调 | 色相 1，强度 2.3 | Lightroom 10 起的色彩分级把暗部和高光写在分离色调字段中 |
| `ColorGradeGlobalHue/Sat` | 三段分级 | 强度 2.3 | 在 OkLab 中作为向量叠加到三段（三段权重之和为 1，等于整体着色） |
| `SplitToningBalance` | 分级平衡 | 1 | |
| `ToneCurvePV2012`（含 `Red`／`Green`／`Blue`） | RGB 曲线与红绿蓝通道曲线 | ÷255 | 平滑曲线；旧的 `ToneCurve` 在没有 PV2012 曲线时使用 |
| `Parametric{Shadows,Darks,Lights,Highlights}` + 三个分割点 | 曝光曲线的精细曲线 | | 每个区段在区段及两侧各半个区段内平滑弯曲，±100 时中心移动区段宽度的 40%；端点固定、单调 |
| `Sharpness` | 锐化 | 0.35 | |
| `LuminanceSmoothing` | 明度降噪 | 0.85 | |
| `ColorNoiseReduction` | 彩色杂点 | 1.35 | |
| `PostCropVignetteAmount`／`Midpoint`／`Feather` | 暗角／中点／羽化 | 0.45／1／映射 | 羽化 50（Adobe 默认）对应 LUMEN 的 70 |
| `GrainAmount`／`GrainSize` | 颗粒数量／大小 | 1.3／1.2 | 大小 25（Adobe 默认）对应 LUMEN 的 30 |
| `Exposure`、`Contrast`、`FillLight`、`HighlightRecovery`、`Shadows`、`Clarity`（处理版本 2003／2010） | 曝光、对比度、暗部、亮部、黑色、清晰度 | | 只在没有 2012 版对应字段时使用 |

## 复制不了的 Adobe 行为

下列设置在 LUMEN 中没有对应项。导入后在报告中列出（不为 Adobe 默认值的才列），并随预设保存：预设的提示文字和右键「查看导入说明」都能看到。

- **Adobe 相机配置文件与创意 Look**（`CameraProfile`、`Look`，包括其中的 RGBTable／LookTable 颜色表）。LUMEN 用自己的相机显影基准；Adobe 的颜色表是专有格式。
- **校准面板**（阴影色调、红绿蓝原色的色相与饱和度）。
- **局部调整与蒙版**（画笔、渐变、径向、AI 蒙版、范围蒙版、修复与红眼）。
- **镜头校正与去边、变换（Upright／透视）、裁切与方向。** LUMEN 的镜头校正按照片自动匹配，裁切属于单张照片。
- **点颜色。**
- **色彩分级的明亮度与混合**（LUMEN 的分级只着色、不改明度，混合固定）。
- **锐化半径、细节与蒙版；降噪的细节、对比度与平滑度。**
- **暗角的圆度、样式与高光保护；颗粒粗糙度。**
- **自动白平衡、自动色调**等自动设置；AI 增强与降噪。
- **滑块的精确形状。** Adobe 没有公开 PV2012 及之后各滑块的曲线，LUMEN 的亮部／暗部是局部色调映射、颜色在 OkLCh 中计算，工作空间是 sRGB 原色的线性光（Adobe 为 ProPhoto 原色）。换算系数只让效果的“幅度”接近。
- 同一个预设在 Lightroom 中对不同相机会因配置文件而略有差异，LUMEN 不复制这一点。

## 校准

Adobe 没有公开滑块曲线，本机也没有 Adobe 软件可以对照，因此校准以 **估计的 Adobe 效果幅度** 为目标（写在 `tools/calibrate_lightroom.py` 的 `TARGETS`／`SYNTHETIC` 中，例如“对比度 +100 时中间一半色调的范围约扩大 30%”“亮部 −100 时第 90 百分位约暗 0.85 档”）。脚本把测试照片（默认 `E:\LumenSamples\playraw` 中的 8 张 RAW，1200 像素）用 LUMEN 以一系列滑块值渲染，测量同样的统计量，求出达到各目标所需的 LUMEN 数值；系数是这些比值的中位数。色彩分级、颗粒、降噪、锐化、八色和暗角用合成图像测量。

```
python tools/calibrate_lightroom.py [照片或文件夹 …] [--size 1200] [--process 2] [--json 结果.json]
```

1.7.0 的结果（处理版本 2，8 张照片）：对比度 0.35、清晰度 0.45、暗角 0.45、锐化 0.35、八色色相 0.5：LUMEN 的这些滑块比估计的 Adobe 强，换算后数值变小。亮部 1.8、白色 1.55、黑色 1.6、色彩分级 2.3、颗粒 1.3、色调 1.35：LUMEN 的这些滑块较弱，Adobe 的较大数值会截到 LUMEN 的 ±100（例如亮部 −56 以下都对应 LUMEN 的 −100）。冷暖、饱和度、纹理、暗部、降噪接近 1。每个目标的拟合值与超出范围的次数由脚本输出；各照片之间的离散不小（例如暗部 0.27–1.29），系数是中位数。

处理版本 1 的结果与之基本相同（暗部例外：版本 1 约为 0.9），因此两个处理版本使用同一组系数。目标本身是估计值；有 Lightroom 实际渲染可对照时，修改目标后重新运行即可得到新的系数（`lumen/lightroom.py` 的 `SCALE`）。

## LUT

- **位置：** 全局色彩阶段（曲线、HSL、分级）和局部蒙版之后、暗角与颗粒和裁切之前，作用在显示参考的 sRGB（0–1）上，也就是 `.cube` 文件通常预期的输入。处理版本 1 本来就在 sRGB 中渲染；处理版本 2 导出 Display P3／Adobe RGB 时，按每个颜色的 sRGB 编码查表，sRGB 以外的部分作为偏移保留。
- **插值：** 四面体插值（灰阶保持中性）；1D 表线性插值。支持 `LUT_1D_SIZE`、`LUT_3D_SIZE`（2–256，大于 65 时重采样为 65³）、`DOMAIN_MIN`／`DOMAIN_MAX`、`TITLE`，以及 DaVinci Resolve 的 1D 预整形加 3D 表与 `LUT_1D_INPUT_RANGE`／`LUT_3D_INPUT_RANGE`。其他关键字（如 `LUT_IN_VIDEO_RANGE`）忽略并提示。
- **强度：** 结果 = 原图 + (LUT 结果 − 原图) × 强度。
- **一致性：** 纯逐点运算，只用精确的算术，分块渲染（放大预览、超大图分条导出）与整幅渲染逐像素一致。目前在 CPU 上计算（在 GPU 图之后），1600 像素预览约多 50 ms。
- **保存：** 表数据（float32、zlib、base64）随工程、快照、选片集和预设保存，换一台没有 `.cube` 文件的电脑也能打开；同一工程或选片集中相同的表只存一份。工程格式因此升为版本 7。

## 命令行与 MCP

- `edit.apply`：`changes` 中的 `lut` 可为 `{"path": "….cube", "amount": 80}`（载入）、`{"amount": 50}`（调强度）或 `null`（移除）。自然语言修图只能调强度或移除。
- `preset.import`（MCP 工具 `import_presets`）：导入预设文件或预设包，返回导入的预设和未能套用的设置。
- `edit.presets`／`edit.preset`：列出并套用内置风格与导入的预设；`edit.preset` 的 `paths` 可一次套用到多张照片。
- `lumen-cli render … --preset 名称 --lut 文件.cube --lut-amount 80`，`lumen-cli import-presets 文件`。
