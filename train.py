import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.tensorboard import SummaryWriter
from unet_arch import Unet
from load_data import makeloaders

#hpams:
lrate = 0.01
batch_size = 16
img_height = 160
img_width = 160
num_epochs = 10
num_workers = 2
load_model = True

def check_accuracy(data_loader, model):
    #our data_loader is going to be loading a list of dictionaries.
    total_correct = 0
    num_pixels = 0
    dice_score = 0
    for dict in data_loader:
        x = dict["Image"]
        y = dict["label"].float().unsqueeze(1)
        model.eval()
        with torch.no_grad():
            preds = model(x)
            #this is basically going to be filled with the converted logits.
            preds = (preds > 0.5).float()
            '''one thing important to note is that '''
            correct_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score +=  (2*preds*y).sum()/((y+preds).sum() + 1e-8)
    print('The accuracy is : {:.4f}'.format((total_correct/num_pixels)*100))      
    print(f"Dice score: {dice_score/len(data_loader)}")


def save_checkpoints(state, filename = "my_checkpoint.pth.tar"):
    print("=> saving checkpoint")
    torch.save(state, filename)  #state can be a model, tensor etc.


def load_checkpoint(checkpoint, model):
    print("=> loading the checkpoint")
    model.load_state_dict(checkpoint["state_dict"])


def train_epoch(dataloader, optim, loss_fn, model, writer, epoch_num): #this function is for training one epoch.
    for ind,dict in enumerate(tqdm(dataloader)):
        img_tensor = dict["Image"]
        label_tensor = dict["targets"].float().unsqueeze(1)

        model_pred = model(img_tensor)
        loss = loss_fn(model_pred, label_tensor)

        optim.zero_grad()
        loss.backward()
        optim.step()

        writer = SummaryWriter()
        num = epoch_num*10 + ind
        writer.add_scaler("train-loss_graph",loss,num)
        print("Loss : {.:4f}".format(loss))



def main():
    train_transform = A.compose(
        [
            A.Resize(height = img_height width = img_width),
            A.Rotate(limit = 35, p = 1.0).
            A.Horizontalflip(p=0.5),
            A.VerticalFlip(p = 0.1),
            A.normalize(
                mean = [0.0,0.0,0.0],
                std = [1.0,1.0,1.0]
                max_pixel_value = 255.0
            ),
            ToTensorV2()
        ]
    )

    val_transform = A.compose(
        [
            A.Resize(height = img_height, width = img_width),
            A.normalize(
                mean = [0.0,0.0,0.0],
                std = [1.0,1.0,1.0]
                max_pixel_value = 255.0
            ),
            ToTensorV2()
        ]
    )

    model = Unet(in_channels = 3, out_channels = 1)
    loss_fn = nn.BCEloss()  
    ''' BCE loss bcz the we are applying a sigmoid on the last layer.'''
    optimizer = optim.Adam(model.parameters(), lr = lrate)

    train_loader, val_loader = makeloaders()
    for epoch in range(num_epochs):
        train_epoch(train_loader, optimizer, loss_fn, model, epoch)
        checkpoint = {
            "state_dict" = model.state_dict(),
            "optimizer" = model.state_dict()
        }
        save_checkpoints(checkpoint)
        check_accuracy(_,model)
        save_preds_as_images(
            
        )

if __name__ == "__main__":
    main() 