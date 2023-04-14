import numpy as np
from skimage.transform import resize
from transformations import ComposeDouble, FunctionWrapperDouble, create_dense_target, normalize_01, AlbuSeg2d
from dataset import SegmentationDataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import pathlib
import albumentations

root = pathlib.Path.cwd() / 'data'

def get_filenames_of_path(path: pathlib.Path, ext: str = '*'):
    """Returns a list of files in a directory/path. Uses pathlib."""
    filenames = [file for file in path.glob(ext) if file.is_file()]
    return filenames

# input and target files
inputs = get_filenames_of_path(root / 'Input')
targets = get_filenames_of_path(root / 'Target')

print(len(inputs))

# training transformations and augmentations
transforms_training = ComposeDouble([
    FunctionWrapperDouble(resize,
                          input=True,
                          target=False,
                          output_shape=(128, 128, 3)),
    FunctionWrapperDouble(resize,
                          input=False,
                          target=True,
                          output_shape=(128, 128),
                          order=0,
                          anti_aliasing=False,
                          preserve_range=True),
    AlbuSeg2d(albumentations.HorizontalFlip(p=0.5)),
    FunctionWrapperDouble(create_dense_target, input=False, target=True),
    FunctionWrapperDouble(np.moveaxis, input=True, target=False, source=-1, destination=0),
    FunctionWrapperDouble(normalize_01)
])

# validation transformations
transforms_validation = ComposeDouble([
    FunctionWrapperDouble(resize,
                          input=True,
                          target=False,
                          output_shape=(128, 128, 3)),
    FunctionWrapperDouble(resize,
                          input=False,
                          target=True,
                          output_shape=(128, 128),
                          order=0,
                          anti_aliasing=False,
                          preserve_range=True),
    FunctionWrapperDouble(create_dense_target, input=False, target=True),
    FunctionWrapperDouble(np.moveaxis, input=True, target=False, source=-1, destination=0),
    FunctionWrapperDouble(normalize_01)
])

# random seed
random_seed = 42

# split dataset into training set and validation set
train_size = 0.8  # 80:20 split

inputs_train, inputs_valid = train_test_split(
    inputs,
    random_state= random_seed,
    train_size=train_size,
    shuffle=True)

targets_train, targets_valid = train_test_split(
    targets,
    random_state=random_seed,
    train_size=train_size,
    shuffle=True)

dataset_train = SegmentationDataset(inputs=inputs_train,
                                    targets=targets_train,
                                    transform=transforms_training)

dataset_valid = SegmentationDataset(inputs=inputs_valid,
                                    targets=targets_valid,
                                    transform=transforms_validation)

dataloader_training = DataLoader(dataset=dataset_train,batch_size=2,shuffle=True)
dataloader_validation = DataLoader(dataset=dataset_valid, batch_size=2,shuffle=False)

if __name__ == '__main__':
    batch = dataset_train[0]
    x, y = batch

    print(f'x = shape: {x.shape}; type: {x.dtype}')
    print(f'x = min: {x.min()}; max: {x.max()}')
    print(f'y = shape: {y.shape}; class: {y.unique()}; type: {y.dtype}')