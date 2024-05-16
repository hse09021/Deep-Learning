import torch
import tqdm
import time
import matplotlib.pyplot as plt

import torch.nn as nn
from torchsummary import summary

from torchvision.models.vgg import vgg16

from torchvision import datasets
from torchvision.transforms import Compose, ToTensor, Resize
from torchvision.transforms import RandomHorizontalFlip, RandomCrop, Normalize
from torch.utils.data.dataloader import DataLoader

from torch.optim.adam import Adam



device = "cuda" if torch.cuda.is_available() else "cpu"

# 사전 학습된 모델 준비
model = vgg16(weights='VGG16_Weights.IMAGENET1K_V1') # ❶ vgg16 모델 객체 생성
fc = nn.Sequential( # ❷ 분류층의 정의
       nn.Linear(512 * 7 * 7, 4096),
       nn.ReLU(),
       nn.Dropout(), #❷ 드롭아웃층 정의
       nn.Linear(4096, 4096),
       nn.ReLU(),
       nn.Dropout(),
       nn.Linear(4096, 10,bias=False),
       nn.Softmax(dim=1)
   )
model.classifier = fc # ➍ VGG의 classifier를 덮어씀
model.to(device)
print(model)

#VGGnet feature map and parameters 정보 요약
summary(model,input_size=(3, 224, 224))


# 데이터 전처리와 증강
transforms = Compose([
   ToTensor(),
   Resize(224, antialias=True),
   lambda x: x.repeat(3, 1, 1),
   RandomCrop((224, 224), padding=4),
   RandomHorizontalFlip(p=0.5),
   Normalize(mean=(0.2860, 0.2860, 0.2860), std=(0.3530, 0.3530, 0.3530))
])


# 데이터로더 정의
training_data = datasets.MNIST(root="./data", train=True, download=True, transform=transforms)
test_data = datasets.MNIST(root="./data", train=False, download=True, transform=transforms)

train_loader = DataLoader(training_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

# batch 경사하강법 학습 수행을 위해 사전 학습된 parameter freezing
for name, param in model.features.named_parameters():
    param.requires_grad = False

params_to_update = []
params_name_to_update = []
for name, param in model.classifier.named_parameters():
    param.requires_grad = True
    params_to_update.append(param)
    params_name_to_update.append(name)

print(params_name_to_update)
print(params_to_update)

# batch 경사하강법 학습 수행
lr = 1e-4
#optim = Adam(model.parameters(), lr=lr)
train_cost = []
epochs = 5
optim = Adam(params=params_to_update, lr=lr)
for epoch in range(epochs):
    cost = 0
    s_time = time.time()
    iterator = tqdm.tqdm(train_loader) # ➊ 학습 로그 출력
    for data, label in iterator:
       optim.zero_grad()

       preds = model(data.to(device)) # 모델의 예측값 출력

       loss = nn.CrossEntropyLoss()(preds, label.to(device))
       loss.backward()
       optim.step()
     
       # ❷ tqdm이 출력할 문자열
       iterator.set_description(f"epoch:{epoch+1} loss:{loss.item()}")
       cost += loss.item()
    
    e_time = time.time()

    training_loss = cost / len(train_loader)
    print("Training Loss: {:.4f}".format(training_loss))
    print("Elapsed Time: {:.4f}s".format(e_time - s_time))
    train_cost.append(training_loss)

#학습된 모델의 parameter 파일로 저정
torch.save(model.state_dict(), "CIFAR_pretrained.pth") # 모델 저장

#학습된 모델의 cost function 그래프 출력
plt.title("Training Cost Function")
plt.plot(epochs, train_cost, label="Training Cost")
plt.xlabel("Epoch")
plt.ylabel("Cross Entropy Loss")
plt.show()


#파일에 저장된 학습된 모델의 parameter들을 읽어 모델의 parameter를 초기화
model.load_state_dict(torch.load("CIFAR_pretrained.pth", map_location=device))

#test dataset에 대한 accuracy 성능 확인
model.eval()

num_corr = 0
correct_imgs = []
num_wrong = 0
wrong_imgs = []

with torch.no_grad():
    for data, label in test_data:
        data = data.unsqueeze(axis=0)
        data = data.to(device)
        pred = model(device)
        predicted = pred[0].argmax(0)

        if predicted != label:
            num_wrong += 1
            if num_wrong < 10:
                wrong_imgs.append((data.to('cpu'), label, predicted))
            else:
                continue
        else:
            num_corr += 1
            if num_corr < 10:
                correct_imgs.append((data.to('cpu'), label, predicted))
            else:
                continue

    print(f"Accuracy:{num_corr/len(test_data)}")

#정확하게 분류된 샘플 출력
plt.figure(figsize=(5, 5))
for i, (data, label, predicted) in enumerate(correct_imgs):
    plt.subplot(3, 3, i + 1)
    plt.imshow(data.squeeze().detach().numpy(), cmap='gray')
    plt.axis('off')
    plt.suptitle(f"Predicted: {predicted}, True: {label}")
plt.show()

#잘못 분류된 샘플 출력
plt.figure(figsize=(5, 5))
for i, (data, label, predicted) in enumerate(wrong_imgs):
    plt.subplot(3, 3, i + 1)
    plt.imshow(data.squeeze().detach().numpy(), cmap='gray')
    plt.axis('off')
    plt.suptitle(f"Predicted: {predicted}, True: {label}")
plt.show()