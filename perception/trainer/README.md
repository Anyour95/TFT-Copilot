训练模块说明

包含脚本：
- `generate_dataset_structure.py`：生成检测与分类的数据集目录骨架与示例文件。
- `train_detector.py`：使用 `ultralytics` 的 YOLOv8 API 进行目标检测训练（YOLO 格式标签）。
- `train_classifier.py`：使用 `torchvision` 的 `ImageFolder` 训练 ResNet18 分类器（按文件夹组织的分类数据）。
- `dataset.yaml`：YOLO 数据集配置模板，请根据实际路径修改 `train`/`val` 路径与 `names` 列表。
- `requirements.txt`：训练所需 Python 包（可在虚拟环境中安装）。

快速流程：
1. 生成数据集骨架：
```
python perception/trainer/generate_dataset_structure.py --root ../dataset_detector
```
2. 准备好图片与标签（YOLO 格式），编辑 `dataset.yaml` 的路径与 `names`。
3. 训练检测模型：
```
python perception/trainer/train_detector.py --data dataset.yaml --model yolov8n.pt --epochs 100 --imgsz 640 --batch 16
```
4. 若使用两阶段方案，准备分类数据（按类分文件夹），训练分类器：
```
python perception/trainer/train_classifier.py --data classifier_data --epochs 50 --batch 32 --lr 1e-3
```
