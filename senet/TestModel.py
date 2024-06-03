import time

import torch
import torchvision.transforms as transforms
from PIL import Image

# 假设你已经加载了模型
model = torch.load('../model/SENet.pth')
# model = torch.load_state_dict('model.pth')
model.eval()  # 设置为评估模式

# 如果模型在GPU上，则将其移动到GPU
if torch.cuda.is_available():
    model = model.cuda()

# 加载图片并进行预处理
# 注意：这里的预处理应该与训练模型时使用的预处理相同
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    # transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ImageNet的标准值
])
#测试图片路径
image = Image.open('../images/0128.jpg')
image_tensor = transform(image).unsqueeze(0)  # 添加batch维度

# 如果模型在GPU上，则将图片也移动到GPU
if torch.cuda.is_available():
    image_tensor = image_tensor.cuda()
start = time.time()
# 进行预测
with torch.no_grad():  # 不需要计算梯度
    outputs = model(image_tensor)

# 假设模型是一个分类模型，并且你想要获取概率最高的类别的索引
_, predicted_idx = torch.max(outputs, 1)
print('Predicted class index:', predicted_idx.item())

print(f"预测时间--->{time.time() - start}")

# 如果你有类别的标签列表，你可以将其转换为实际的类别名称
# 例如，假设你有一个类别标签列表 classes = ['cat', 'dog', 'bird', ...]
# predicted_class = classes[predicted_idx.item()]
# print('Predicted class:', predicted_class)