两阶段检测→分类管线说明

用途：在游戏画面中先用目标检测（YOLOv8）定位英雄头像/人像框，然后对裁切的子图使用轻量分类器（ResNet18）识别英雄名称。

快速开始：
1. 创建虚拟环境并安装依赖：
```
python -m venv .venv
# Windows PowerShell
. .venv\Scripts\Activate.ps1
pip install -r perception/two_stage_pipeline/requirements.txt
```
2. 填写 `names.txt`（每行一个英雄类名，顺序与分类模型训练时一致）。
3. 运行示例推理：
```
python perception/two_stage_pipeline/pipeline.py --source path/to/frame_or_dir --detector_weights yolov8n.pt --classifier_weights classifier.pth --names perception/two_stage_pipeline/names.txt --output out
```

说明：
- 检测器使用 `ultralytics` (YOLOv8)。
- 分类器示例使用 `torchvision` 的 `resnet18` 作为轻量分类器，推理快速，训练时可替换为更小的模型。
