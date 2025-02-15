import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
from IPython.display import display
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.optim.lr_scheduler import StepLR
from torchsummary import summary
from tqdm import tqdm
import torchvision.models as models
import PIL

data_path = "D:\\pytorch nn\\data\\Pets-Facial-Expression\\Master Folder" #数据路径
class_name = ['Angry','Sad','happy'] #定义类别

def get_list_files(dirName):
    '''
    input - directory location
    output - list the files in the directory
    '''
    files_list = os.listdir(dirName)
    return files_list

files_list_Angry_train = get_list_files(data_path+'/train/'+class_name[0])
files_list_Sad_train = get_list_files(data_path+'/train/'+class_name[1])
files_list_happy_train = get_list_files(data_path+'/train/'+class_name[2])
files_list_Angry_test = get_list_files(data_path+'/test/'+class_name[0])
files_list_Sad_test = get_list_files(data_path+'/test/'+class_name[1])
files_list_happy_test = get_list_files(data_path+'/test/'+class_name[2])

print("Number of train samples in Angry category {}".format(len(files_list_Angry_train)))
print("Number of train samples in Sad category {}".format(len(files_list_Sad_train)))
print("Number of train samples in happy category {}".format(len(files_list_happy_train)))
print("Number of test samples in Angry category {}".format(len(files_list_Angry_test)))
print("Number of test samples in Sad category {}".format(len(files_list_Sad_test)))
print("Number of test samples in happy category {}".format(len(files_list_happy_test)))

train_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                          [0.229, 0.224, 0.225])])

test_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                          [0.229, 0.224, 0.225])])

train_data = datasets.ImageFolder(os.path.join(data_path, 'train'), transform= train_transform)
test_data = datasets.ImageFolder(os.path.join(data_path, 'test'), transform= test_transform)

train_loader = DataLoader(train_data,
                          batch_size= 16, shuffle= True, pin_memory= False)
test_loader = DataLoader(test_data,
                         batch_size= 16, shuffle= False, pin_memory= False)

class_names = train_data.classes
print(class_names)
print(f'Number of train images: {len(train_data)}')
print(f'Number of test images: {len(test_data)}')

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print("Available processor {}".format(device))

model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 3) 
model = model.to(device)

summary(model, input_size=(3, 224, 224))

train_losses = []   # 训练损失
test_losses = []    # 测试损失
train_acc = []      # 训练准确率
test_acc = []       # 测试准确率

criterion = nn.CrossEntropyLoss()  
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    pbar = tqdm(train_loader)
    correct = 0
    processed = 0

    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        y_pred = model(data)
        loss = criterion(y_pred, target)

        train_losses.append(loss.item())
        loss.backward()
        optimizer.step()

        pred = y_pred.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        processed += len(data)

        pbar.set_description(desc= f'Loss={loss.item()} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')
        train_acc.append(100*correct/processed)

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
    test_acc.append(100. * correct / len(test_loader.dataset))

EPOCHS = 25
for epoch in range(EPOCHS):
    print("EPOCH:", epoch)
    train(model, device, train_loader, optimizer, epoch)
    scheduler.step()
    print('current Learning Rate: ', optimizer.state_dict()["param_groups"][0]["lr"])
    test(model, device, test_loader)

fig, axs = plt.subplots(2,2,figsize=(16,10))

# 训练损失
axs[0, 0].plot(train_losses, color='green')
axs[0, 0].set_title("Training Loss")

# 训练准确率
axs[1, 0].plot(train_acc, color='green')
axs[1, 0].set_title("Training Accuracy")

# 测试损失
axs[0, 1].plot(test_losses)
axs[0, 1].set_title("Test Loss")

# 测试准确率
axs[1, 1].plot(test_acc)
axs[1, 1].set_title("Test Accuracy")

plt.tight_layout()  # 自动调整子图间距
plt.show()