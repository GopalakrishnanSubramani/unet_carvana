import torch
from skimage.io import imread
from torch.utils.data  import Dataset, DataLoader

class SegmentationDataset(Dataset):
    def __init__(self,
                 inputs: list,
                 targets: list,
                 transform= None):
        self.inputs = inputs
        self.targets = targets
        self.transform = transform
        self.inputs_dtype = torch.float32
        self.targets_dtype = torch.long
    
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self,index: int):
        #select the sample from the dataset
        input_id = self.inputs[index]
        targets_id = self.targets[index]

        #load input and target
        x,y = imread(input_id), imread(targets_id)

        #processing
        if self.transform is not None:
            x,y = self.transform(x,y)
        
        #Typescaping
        x,y = torch.from_numpy(x).type(self.inputs_dtype), torch.from_numpy(y).type(self.targets_dtype)

        return x, y

if __name__ == '__main__':
    inputs = ['data/Input/0cdf5b5d0ce1_01.png','data/Input/0cdf5b5d0ce1_02.png']
    targets = ['data/Target/0cdf5b5d0ce1_01.png','data/Target/0cdf5b5d0ce1_02.png']
    training_dataset = SegmentationDataset(inputs=inputs, targets=targets, transform=None)
    training_dataloader = DataLoader(dataset=training_dataset, batch_size=2,shuffle=2)

    x,y = next(iter(training_dataloader))

    print(f'x = shape: {x.shape}; type: {x.dtype}')
    print(f'x = min: {x.min()}; max: {x.max()}')
    print(f'y = shape: {y.shape}; class: {y.unique()}; type: {y.dtype}')