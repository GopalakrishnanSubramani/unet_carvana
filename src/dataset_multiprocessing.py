import torch
from skimage.io import imread
from torch.utils.data  import Dataset, DataLoader
from tqdm import tqdm

class SegmentationDataset(Dataset):
    def __init__(self,
                 inputs: list,
                 targets: list,
                 transform= None,
                 use_cache = False,
                 pre_transform = None):
        self.inputs = inputs
        self.targets = targets
        self.transform = transform
        self.inputs_dtype = torch.float32
        self.targets_dtype = torch.long
        self.use_cache = use_cache
        self.pre_transform = pre_transform

        if self.use_cache:
            from multiprocessing import Pool
            from itertools import repeat

            with Pool() as pool:
                self.cached_data = pool.starmap(self.read_images, zip(inputs, targets,repeat(self.pre_transform)))
    
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self,index: int):
        
        #use cached data
        if self.use_cache:
            x, y = self.cached_data[index]
        else:
            input_id = self.inputs[index]
            targets_id = self.targets[index]

            #Load input and targets
            x,y = imread(str(input_id)), imread(str(targets_id))

        #processing
        if self.transform is not None:
            x,y = self.transform(x,y)
        
        #Typescaping
        x,y = torch.from_numpy(x).type(self.inputs_dtype), torch.from_numpy(y).type(self.targets_dtype)

        return x, y
    
    @staticmethod
    def read_images(inputs, targets, pre_transform):
        inputs, targets = imread(str(inputs)), imread(str(targets))
        if pre_transform:
            inputs, targets = pre_transform(inputs, targets)
        return inputs, targets

if __name__ == '__main__':
    inputs = ['data/Input/0cdf5b5d0ce1_01.png','data/Input/0cdf5b5d0ce1_02.png']
    targets = ['data/Target/0cdf5b5d0ce1_01.png','data/Target/0cdf5b5d0ce1_02.png']
    training_dataset = SegmentationDataset(inputs=inputs, targets=targets, transform=None)
    training_dataloader = DataLoader(dataset=training_dataset, batch_size=2,shuffle=2)

    x,y = next(iter(training_dataloader))

    print(f'x = shape: {x.shape}; type: {x.dtype}')
    print(f'x = min: {x.min()}; max: {x.max()}')
    print(f'y = shape: {y.shape}; class: {y.unique()}; type: {y.dtype}')