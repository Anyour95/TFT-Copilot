import torch
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import json

class Classifier:
    def __init__(self, weights_path: str = None, device: str = None, num_classes: int = None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = models.resnet18(pretrained=(weights_path is None))
        if num_classes is not None:
            self.model.fc = torch.nn.Linear(self.model.fc.in_features, num_classes)
        self.model.to(self.device)
        if weights_path:
            self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
        self.model.eval()

        self.tf = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def predict_image(self, pil_img: Image.Image, topk: int = 1, names=None):
        img = self.tf(pil_img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(img)
            probs = F.softmax(logits, dim=1)
            topk_probs, topk_idxs = probs.topk(topk, dim=1)
            topk_probs = topk_probs.cpu().numpy().tolist()[0]
            topk_idxs = topk_idxs.cpu().numpy().tolist()[0]
        if names:
            labels = [names[i] if i < len(names) else str(i) for i in topk_idxs]
        else:
            labels = [str(i) for i in topk_idxs]
        return list(zip(labels, topk_probs))
