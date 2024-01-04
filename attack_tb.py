import torch
import attacks
from torchvision import datasets, transforms
import numpy as np

print(torch.__version__)

print(f"Is CUDA supported by this system? {torch.cuda.is_available()} ")
print(f"CUDA version: {torch.version.cuda}")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

"""
This demo version only supports CIFAR10
Models and weights are from https://github.com/huyvnphan/PyTorch_CIFAR10
"""

""" load model """
from net.cifar10.resnet import resnet18
model0 = resnet18()
model0.name = 'resnet18'
model0.load_state_dict(torch.load('model/cifar10/resnet18.pt'))
model0.loss = 'cross entropy'

from net.cifar10.mobilenetv2 import MobileNetV2
model1 = MobileNetV2()
model1.name = 'mobilenet_v2'
model1.load_state_dict(torch.load('model/cifar10/mobilenet_v2.pt'))
model1.loss = 'cross entropy'

target_model = model0
surrogate_model = model1
test_batch_size = 1

surrogate_model.eval()
surrogate_model.to(device)
target_model.eval()
target_model.to(device)

""" load dataset """
transform_train = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
])

train_data = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
test_data = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=256, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=test_batch_size, shuffle=False)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

attacks.test(device, classes, target_model, surrogate_model, test_loader)
