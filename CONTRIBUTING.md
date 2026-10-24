# Contributing / 参与贡献

Bug reports, reproducible examples, documentation improvements, and focused pull requests are welcome. The application interface is currently Chinese; issues may be written in Chinese or English.

欢迎提交问题、可复现样例、文档改进和范围明确的 Pull Request。可以使用中文或英文交流。

## Local development

1. Use Windows with Python 3.12 x64, an Apple silicon Mac with Python 3.12 (`requirements-macos.txt`, then also `python tools/fetch_exiftool.py`), or x86_64 Linux with Python 3.12 (`requirements.txt`; `requirements-linux-gpu.txt` adds the CUDA 13 / cuDNN 9 ONNX Runtime and CuPy native kernels). On Linux, `./run-source.sh` creates the environment and restores the assets. Create a virtual environment otherwise.
2. Install `requirements-dev.txt` (Windows) or `requirements-linux-dev.txt` (Linux) and run `python tools/fetch_assets.py`.
3. Run `python -m pytest tests -q` (Linux: `LD_LIBRARY_PATH` is prepared by `run-source.sh`; the build script and `main.py` also handle it). The full suite includes actual local model inference and requires runtime assets.
4. Run `python main.py` to inspect changes in the desktop application. Linux release builds use `tools/build_linux.sh --sample photo.DNG`.

Keep image processing off the UI thread. Preserve original photographs, avoid overwriting output names, and retain compatibility with existing `.lumen` / `.lumenalbum` recipes. Add meaningful regression coverage for processing, state, and I/O changes.

图像处理应放在后台线程，保留磁盘原片、避免覆盖同名副本，并注意旧工程兼容性。涉及处理结果、状态或文件读写的修改，请加入相应回归检查。

## Reporting a problem

Include the application version, Windows version, camera/model and RAW format, reproduction steps, and whether the issue occurs in the original-resolution view or exported image. Share a small sample only if you own it or have permission. Do not post passwords, API tokens, private paths, or unredacted sensitive EXIF information.

报告问题时请注明软件版本、Windows 版本、相机及 RAW 类型、复现步骤，以及问题出现在预览还是导出。样片需有分享权限，请先移除敏感信息。

## Repository contents

Do not commit model binaries, camera originals, personal edit projects, installers, virtual environments, or caches. Runtime assets are distributed in Releases and described by `assets/runtime-assets.json`; changing them requires updating sizes and SHA-256 checksums as well as model provenance and license notices.

Contributions are submitted under the project's MIT license. Third-party code or assets must retain compatible provenance and licensing information. Do not add official brand logos without redistribution rights.
