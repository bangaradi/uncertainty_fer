import sys
sys.path.append('..')
import torch
import torch.nn as nn
from dataset.dataset import DatasetEXPR, DatasetAU, DatasetVA
from facenet_pytorch import InceptionResnetV1
import torchvision.transforms as transforms

class BaselineModelEXPR(nn.Module):
    def __init__(self, pretrained=True, seq_len=32):
        super(BaselineModelEXPR, self).__init__()
        self.input_size = 512
        self.hidden_size = 256
        self.fc_size = 64
        self.output_dim = 8
        self.backbone = InceptionResnetV1(pretrained='vggface2') # output : (batch_size, seq_len, 512)
        for param in self.backbone.parameters(): # for freezing the weights of this part of the model.
            param.requires_grad = False
        self.gru = nn.GRU(input_size=self.input_size, hidden_size=self.hidden_size, batch_first=True, bidirectional=False) # output : (batch_size, seq_len, 256)
        self.fc = nn.Linear(self.hidden_size, self.fc_size)
        self.fc_output = nn.Linear(self.fc_size, self.output_dim)
    
    def forward(self, x):
        # reshape the input
        B, S, C, H, W = x.shape
        x_dim = B*S
        x_reshaped = x.view(x_dim, C, H, W)
        x_reshaped = self.backbone(x_reshaped)
        x = x_reshaped.view(B, S, -1)
        x, _ = self.gru(x)
        x = x[:, -1, :]
        x = self.fc(x)
        x = self.fc_output(x)
        x = nn.Softmax(dim=-1)(x)
        return x



if __name__ == "__main__":
    annotation_file = '/home/stud-1/aditya/datasets/affwild2/Third ABAW Annotations/annotations.pkl'
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((160, 160)),
    ])
    dataset = DatasetEXPR(annotation_file=annotation_file, mode='Train', seq_len=32, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=False)
    model = BaselineModelEXPR(pretrained=True, seq_len=32)
    for i, batch in enumerate(dataloader):
        print(f"Batch {i+1}")
        print(f"Image Tensor Shape: {batch['images'].shape}")
        print(f"Labels: {batch['labels']}")
        print(f"Frame IDs: {batch['frame_ids']}")
        # print(f"Labels: {batch['valence']}")
        # print(f"Labels: {batch['arousal']}")
        # print(f"Labels: {batch['AUs']}")
        # print(f"paths : {batch['paths']}")
        # Break after the first batch to keep the output short
        out = model(batch['images'])
        print(out.shape)
        break