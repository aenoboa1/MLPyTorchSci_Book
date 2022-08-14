# import the torchvision library
import torch
import torchvision
import argparse,os

# Verify the directory existence
def dir_path(string):
    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError(string)

def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path',nargs="?", type=dir_path,default="./img") # optional parameter path
    args = parser.parse_args()
    return args

def read_data() -> torch.utils.data.Dataset:
    args = create_arg_parser()
    try:
        celeba_dataset = torchvision.datasets.CelebA(
            args.path,split='train',target_type='attr',download=True)
        return celeba_dataset
    except:
        pass

if __name__ == '__main__':
    read_data()

