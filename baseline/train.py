import torch
import torch.nn as nn
import torchvision.transforms as transforms
from dataset.dataset import DatasetEXPR
from model.model_expr import BaselineModelEXPR
from facenet_pytorch import InceptionResnetV1
from tqdm import tqdm
import wandb


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
        project = "trial-run-EXPR-task",
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
                print(f"Batch {i}/{len(dataloader)} | Loss: {_loss/(i+1):.4f}")
                wandb.log({f"epoch({epoch}) : (avg) loss" : _loss/(i+1)})
    wandb.finish()



