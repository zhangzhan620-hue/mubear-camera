# 木熊智能摄像头

基于 FastAPI + OpenCV + YOLO26 的智能摄像头应用：人体/人脸检测、手势识别、拍照上传腾讯云 COS。

## 环境要求

- Windows + Python 3.10~3.12
- 摄像头

## 安装与运行

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python run.py
```

启动后自动打开浏览器（默认 http://127.0.0.1:8000）。

## 配置（可选）

如需腾讯云 COS 上传功能，复制配置模板为 `.env` 并填入密钥（`.env` 不会被提交）：

```bash
copy .env.example .env
```

不配置也能运行，上传功能自动关闭。

## 模型文件

项目根目录需包含以下模型（已随仓库提供）：

- `yolo26n.onnx` — 人体检测
- `yolo26n-pose.onnx` — 姿态关键点
- `yolo26n.pt` / `yolo26n-pose.pt` — 训练源文件（重新导出 ONNX 时使用）

## 打包为 exe

```bash
build.bat
```

产物在 `dist\mubear\mubear.exe`。

## 目录结构

- `app/` — 主程序（检测、拍照、上传、Web 界面）
- `train/` — 训练与模型导出脚本
- `tools/` — 调试工具（相机测试、基准测试等）
