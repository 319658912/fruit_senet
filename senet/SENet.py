import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import timm
import time

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # SENet通常接受224x224的输入
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ImageNet的标准化参数
])

# 加载训练集和验证集（假设你的数据在'data'文件夹下，分为'train'和'val'子文件夹）
train_dataset = datasets.ImageFolder(root='your_path', transform=transform)
val_dataset = datasets.ImageFolder(root='your_path', transform=transform)

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# 加载SENet模型（这里使用se_resnet50作为示例）
model = timm.create_model('seresnet50', pretrained=False, num_classes=len(train_dataset.classes))
# model = timm.create_model('seresnet50', pretrained=False, num_classes=33)
model = model.cuda() if torch.cuda.is_available() else model

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

start_time = time.time()
# 训练模型

best_acc = 0.0

train_loss = []
train_acc_list = []

num_epochs = 10 # 训练次数
for epoch in range(num_epochs):
    model.train()
    print(f'第{epoch + 1} 次执行')
    running_loss = 0.0
    corrects = 0.0
    total = 0.0

    for images, labels in train_loader:
        images = images.cuda() if torch.cuda.is_available() else images
        labels = labels.cuda() if torch.cuda.is_available() else labels

        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        corrects += torch.sum(preds == labels.data)
        total += labels.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    train_acc = 100.0 * corrects.double() / total
    # train_acc = 100.0 * corrects / total
    train_loss.append(loss.item())
    train_acc_list.append(train_acc.item())

    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}, Train Acc: {train_acc:.2f}%')

    # 这里可以添加验证逻辑、保存模型等操作
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}')
    with open("train_loss.txt", 'w') as train_los:
        train_los.write(str(train_loss))

    with open("train_acc.txt", 'w') as train_ac:
        train_ac.write(str(train_acc_list))


    # 注意：在实际应用中，你通常还需要在验证集上评估模型性能，并可能需要在每个epoch后保存最佳模型。

    # 验证模型
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to('cuda'), labels.to('cuda')
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()


    val_acc = 100 * correct / total
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}, Val Acc: {val_acc:.2f}%')

    # 保存最佳模型（可选）
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model, '../../model/SENet_best_model.pth')

torch.save(model, '../model.SENet.pth')
print('Training complete')
print(f'耗时--->{time.time() - start_time}秒')
