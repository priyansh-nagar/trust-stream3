import torch
from torchvision import transforms

device = "cuda" if torch.cuda.is_available() else "cpu"

model = torch.hub.load(
    "selimsef/dfdc_deepfake_challenge",
    "efficientnet_b0",
    pretrained=True
).to(device).eval()

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225])
])

def predict(image):
    tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(tensor)
        prob = torch.softmax(out, 1)
        conf, pred = torch.max(prob, 1)

    trust_score = int(conf.item()*100 if pred.item()==0 else (1-conf.item())*100)
    return pred.item(), conf.item(), trust_score, tensor, model