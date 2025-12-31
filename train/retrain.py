import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn, torch.optim as optim

device = "cuda" if torch.cuda.is_available() else "cpu"

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225])
])

dataset = datasets.ImageFolder("data/labeled", transform)
loader = DataLoader(dataset, batch_size=8, shuffle=True)

model = torch.hub.load(
    "selimsef/dfdc_deepfake_challenge",
    "efficientnet_b0",
    pretrained=True
).to(device)

loss_fn = nn.CrossEntropyLoss()
opt = optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(3):
    for x,y in loader:
        x,y = x.to(device), y.to(device)
        opt.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        opt.step()

torch.save(model.state_dict(), "deeptrust_finetuned.pth")
print("Model retrained and saved")