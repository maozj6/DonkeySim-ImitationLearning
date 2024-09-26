
import os
import json
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
from tqdm import tqdm  # 导入 tqdm
class CustomImageDataset(Dataset):
    def __init__(self, json_folder, image_folder, transform=None):
        self.json_folder = json_folder
        self.image_folder = image_folder
        self.transform = transform
        self.data = self.load_json_files()

    def load_json_files(self):
        data = []
        for json_file in os.listdir(self.json_folder):
            if json_file.endswith('.json'):
                json_path = os.path.join(self.json_folder, json_file)
                with open(json_path, 'r') as f:
                    info = json.load(f)
                    image_path = os.path.join(self.image_folder, info['cam/image_array'])
                    data.append({
                        'image_path': image_path,
                        'throttle': info['user/throttle'],
                        'angle': info['user/angle'],
                        'mode': info['user/mode'],
                        'lap': info['track/lap'],
                        'loc': info['track/loc']
                    })
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path = self.data[idx]['image_path']
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        # Get additional metadata
        throttle = self.data[idx]['throttle']
        angle = self.data[idx]['angle']
        mode = self.data[idx]['mode']
        lap = self.data[idx]['lap']
        loc = self.data[idx]['loc']

        return image,np.array([ throttle, angle])

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # 输入通道数为3，输出通道数为16，卷积核大小为3x3
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)

        # 池化层
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # 全连接层
        self.fc1 = nn.Linear(64 * 20 * 15, 128)  # 展平后输入到全连接层
        self.fc2 = nn.Linear(128, 2)  # 输出2个分类

    def forward(self, x):
        # 卷积层 + ReLU 激活 + 池化层
        x = self.pool(F.relu(self.conv1(x)))  # 输出尺寸: (16, 80, 60)
        x = self.pool(F.relu(self.conv2(x)))  # 输出尺寸: (32, 40, 30)
        x = self.pool(F.relu(self.conv3(x)))  # 输出尺寸: (64, 20, 15)

        # 展平
        x = x.view(-1, 64 * 20 * 15)

        # 全连接层
        x = F.relu(self.fc1(x))
        x = self.fc2(x)  # 输出为2

        return x
# Example usage:

if __name__ == '__main__':

    image_folder = 'F:\\24fall\donkey-adam\donkey-unity-sim-new-branch\donkey-unity-sim-new-branch\donkey_data\donkey_data\\type1'
    json_folder = 'F:\\24fall\donkey-adam\donkey-unity-sim-new-branch\donkey-unity-sim-new-branch\donkey_data\donkey_data\\type1'

    # Define any transformations you want to apply to the images
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    device="cuda"
    model = SimpleCNN().to(device)    # Create dataset and dataloader
    dataset = CustomImageDataset(json_folder=json_folder, image_folder=image_folder, transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    mse_loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # Iterating over the dataloader
    save_dir = './'
    num_epochs = 10
    for epoch in range(num_epochs):
        running_loss = 0.0
        running_mae = 0.0
        with tqdm(dataloader, unit="batch") as tepoch:
            tepoch.set_description(f"Epoch {epoch + 1}/{num_epochs}")

            for batch_idx, (images, labels) in enumerate(tepoch):
                # images,labels = batch
                images = images.to(device).float()
                labels = labels.to(device).float()
                outputs = model(images)

                mse_loss = mse_loss_fn(outputs, labels)

                # 计算 MAE
                mae_loss = torch.mean(torch.abs(outputs - labels))

                # 反向传播和优化
                optimizer.zero_grad()  # 清除上次梯度
                mse_loss.backward()  # 反向传播计算梯度
                optimizer.step()  # 更新模型参数

                # 记录损失
                running_loss += mse_loss.item()/32
                running_mae += mae_loss.item()/32

                tepoch.set_postfix(MSE_Loss=running_loss / (batch_idx + 1), MAE=running_mae / (batch_idx + 1))
        save_path = os.path.join(save_dir, f"model_epoch_{epoch+1}.pth")
        torch.save(model.state_dict(), save_path)
        print(f"Model parameters saved at {save_path}")
    print("Finished Training")
    print(0)
            # Your processing logic here
