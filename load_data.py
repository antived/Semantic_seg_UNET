import numpy as np
import os
import cv2
from PIL import Image
import pandas as pd
import torch 
from torchvision import transforms
from torch.utils.data import DataLoader,Dataset

meta_data_file = pd.read_csv("/Users/Vedant Dutta/Desktop/road_dataset/metadata.csv")
'''the columns are called: image_id, split, sat_image, maks_path
    split basically tells if the image is a part of the train, test or validate.'''

'''map-style database :- https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset
    it must basically be having a __len__ method ans also a __getitem__ method.'''

'''the process of dealing with the images from the csv file. 
    read from the row -> convert color-> Image.fromarray-> transform.ToTensor'''
class createtraindataset(Dataset):  #this is making a map-style database.
    def __init__(self, dataframe, path):
        super().__init__(self)
        self.dataframe = dataframe
        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Lambda(lambda x: x.to(torch.float32))])
        self.path = path

    def __len__(self):
        return self.dataframe.shape[0]

    def __getitem__(self, indx): #when this method is called for an indx, must read idxth img and its corresponding label form folder.
        img_path = self.dataframe.iloc[indx]['sat_image']
        full_path = os.path.join(self.path, img_path)
        img = cv2.imread(full_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img = self.transform(img)


        label_semi_path = self.dataframe.iloc[indx]['mask_path']
        label_path = os.path.join(self.path,label_semi_path)
        label = cv2.imread(label_path)
        label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
        label = np.where(label == 255.0, 1, 0)
        label = torch.tensor(label,dtype = torch.long)

        return {"Image": img, "targets": label}


class createvaldataset(Dataset):
    def __init__(self, df,val_dir):
        super().__init__(self)
        self.df = df
        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Lambda(lambda x:x.to(torch.float32))])
        self.path = val_dir

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self,indx):
        img_path = self.dataframe.iloc[indx]['sat_image']
        full_path = os.path.join(self.path, img_path)
        img = cv2.imread(full_path)
        img = cv2.cvtColor(img, cv2.BGR2RGB)
        img = Image.fromarray(img)
        img = self.transform(img)

        return img #this is going to be returning a tensor basically, that is full of torch.float32

class makeloaders:
    def __init__(self, train_df, val_df, train_dir, val_dir, batch_size = 32, shuffle = True, num_workers = 0):
        self.train_dataset = createtraindataset(train_df, train_dir)
        self.val_dataset = createvaldataset(val_df, val_dir)
        
        self.train_dataloader = DataLoader(self.train_dataset, batch_size = batch_size, shuffle = shuffle, num_workers = num_workers)
        self.val_dataloader = DataLoader(self.val_dataset, batch_size = batch_size, shuffle = shuffle, num_workers = num_workers)


    def get_loaders(self):
        return self.train_dataloader, self.val_dataloader

'''so now the dataset has been prepared and we have to make a class to make the dataloders for 
    the validation and the training.'''


df = pd.DataFrame(meta_data_file)
train_data = df[df['split'] == 'train']
val_data = df[df['split'] == 'valid']
main_path = "C:\Users\Vedant Dutta\Desktop\road_dataset\train"

#train_dataset = createtraindataset(train_data, main_path)

#loader = DataLoader(train_dataset, batch_size = 16, shuffle = True)


