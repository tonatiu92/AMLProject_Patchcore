from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torch
import os
import PIL


DATA_PATH = ".\data"
GROUND_TRUTH = "ground_truth"

def export_data(self,subdatasets: tuple[str], resize: int, imagesize: int, batchsize = 1) -> None:
    """
    Description: Function that extract the data and take the train/test set of the dataset
    @args:
    subdatasets [tuple[str]]: list of subdataset
    resize [int]: value of the desired image size
    imagesize [int]: sizes of the images
    batchsize [int]: number of learn batch
    """
    dataset = {}
    #For each subdataset we will load the train or the test dataloader
    for subdataset in subdatasets:
        dataset[subdataset] = {
            "train": Data(subdataset, resize, imagesize, "train", batchsize),
            "test": Data(subdataset, resize, imagesize, "test", batchsize)
        }
    return dataset


class Data():
    """
    Description: Class defining the dataloader
    """    
    
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]
    
    def __init__(self,subdataset: str, resize: int, imagesize: int, split: str, batchsize=1)->None:
        """
        Description: Class defining the dataloader
        
        @args 
        subdataset [str]: Name of the subdataset of MVTEC
        split [str]: Type of the data set (train or test)
        resize [int]: Size the loaded image initially get resized to
        imagesize [int]: Size the resized loaded image gets (center-) cropped to
        """    
        dataset = DatasetMVTEC(subdataset, split, imagesize, resize) #set the dataset subclass of dataset
        #set the dataloader
        self._dataloader = DataLoader(dataset, batch_size=batchsize, shuffle=False, num_workers=0,pin_memory=True)
    
    @property
    def dataloader(self):
        return self._dataloader
        
class DatasetMVTEC(Dataset):
    """
    SubclassOf torch.utils.data.Dataset
    Description: Define a dataset of type MVTEC
    """
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]
    
    def __init__(self, subdataset: str, split: str, imagesize:int, resize: int) -> None:
        """
        Description: Constructor of a datasetMVTEC instance
        @args:
            subdataset [str]: Name of the subdataset of MVTEC
            split [str]: Type of the data set (train or test)
            imagesize [int]: Size the resized loaded image gets (center-) cropped to

        """
        super().__init__()
        
        #region transform
        # This region realize the pre-processsing and the resizing on the images
        self.transform_img = [
            transforms.Resize(resize),
            transforms.CenterCrop(imagesize),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.IMAGENET_MEAN, std=self.IMAGENET_STD),
        ]
        self.transform_img = transforms.Compose(self.transform_img)

        self.transform_mask = [
            transforms.Resize(resize),
            transforms.CenterCrop(imagesize),
            transforms.ToTensor(),
        ]
        #Also in the mask
        self.transform_mask = transforms.Compose(self.transform_mask)
        #endregion
        
        self._source = subdataset  #name of the subdataset
        self._split = split  #split currently loaded
        self._datapath = os.path.join(DATA_PATH, self._source)  #example: /data/bottle
        self.data_to_iterate = [] #defining the data iterator
        
        for anomaly in os.listdir(os.path.join(self._datapath, split)) :
            #for each type of anomaly we will take the images
            images = [x for x in sorted(os.listdir(os.path.join(self._datapath,split,anomaly)))] #example: data/bottle/train/good
            if  anomaly != "good":
                #If it is an image take the mask using the ground_truth image
                anomaly_mask_path = os.path.join(self._datapath,GROUND_TRUTH, anomaly)#example: /data/bottle/ground_truth
                anomaly_mask_files = sorted(os.listdir(anomaly_mask_path)) #use sort as the image are named with  numbers
                maskpaths= [
                    os.path.join(anomaly_mask_path, x) for x in anomaly_mask_files
                ]
            else:
                #If the image is good, there is no mask to detect the anomaly
                maskpaths = None
            if maskpaths:
                #append the data inside the iterator when there is a mask
                for x,y in zip(images,maskpaths):
                    self.data_to_iterate.append([anomaly,x,y])
            else:
                #append the data inside the iterator with a good image
                for x in images:
                    self.data_to_iterate.append([anomaly,x,None])
        self._imagesize = (3, imagesize, imagesize)
    
    
    def __len__(self) -> int:
        """
        Override the __len__ method of the superclass
        """
        return len(self.data_to_iterate)  

    def __getitem__(self, index: int) -> dict:
        """
        Override the __getitem__ method of the superclass
        @args
        index [int]: index of the image in the data iterator
        @returns
        dictionary containing the image data
        """
        anomaly, image_path, mask_path = self.data_to_iterate[index]
        #get the tensor of the image
        image = PIL.Image.open(os.path.join(DATA_PATH,self._source,self._split,anomaly,image_path)).convert("RGB")
        image = self.transform_img(image) # applying the transformation as defined in the constructor
        
        if self._split == "test" and mask_path is not None:
            mask = PIL.Image.open(mask_path) # get the tensor of the mask
            mask = self.transform_mask(mask)
        else:
            mask = torch.zeros([1, *image.size()[1:]])
        #Definitionn of the image
        return {
            "image": image,
            "mask": mask,
            "classname": self._source,
            "anomaly": anomaly
        }