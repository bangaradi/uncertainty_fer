import sys
sys.path.append('..')
import torch
import torch.nn as nn
from dataset.dataset import DatasetEXPR, DatasetAU, DatasetVA
from facenet_pytorch import InceptionResnetV1
import torchvision.transforms as transforms
from tqdm import tqdm
import wandb

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
    # hyperparams list : 
    '''
    seq_len
    batch_size
    lr
    epochs
    ---------------------
    ---------------------
    some others :
    ---------------------
    fc layer dimension
    fc_output layer dimension
    loss function
    optimizer criterion
    backbone choice
    '''
    wandb.init(
        entity = "adityavb",
        project = "second-run-EXPR-trial",
        config = {
            "seq len" : 32,
            "batch_size" : 64,
            "backbone" : "Facenet512",
            "optimizer lr" : 0.001,
            "epochs" : 10
        }
    )
    annotation_file = '/home/stud-1/aditya/datasets/affwild2/Third ABAW Annotations/annotations.pkl'
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((160, 160)),
    ])
    dataset = DatasetEXPR(annotation_file=annotation_file, mode='Train', seq_len=32, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BaselineModelEXPR(pretrained=True, seq_len=32).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    _loss = 0
    for epoch in range(10):
        print(f"Epoch {epoch+1}/{10}")  # Show progress
        for i, batch in tqdm(enumerate(dataloader)):
            # Move data and target to device
            target = batch['labels']
            # print(target)
            target = target[:, -1].unsqueeze(1)
            one_hot_vectors = torch.zeros(target.size(0), 8)
            one_hot_vectors.scatter_(1, target.long(), 1)
            target = one_hot_vectors
            data = batch['images']
            data = data.to(device)
            target = target.to(device)

            # Clear gradients from previous iteration
            optimizer.zero_grad()

            # Forward pass
            output = model(data)

            # Calculate loss
            loss = criterion(output, target)

            # Backward pass and parameter update
            loss.backward()
            optimizer.step()
            _loss += loss.item()
            # Print training details (optional)
            if i % 100 == 0 and i != 0:  # Print every 100 batches
                print(f"Batch {i}/{len(dataloader)} | Loss: {_loss/100:.4f}")
                wandb.log({f"epoch({epoch}) : (avg) loss" : _loss/100})
                _loss = 0
    wandb.finish()
