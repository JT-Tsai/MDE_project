import torch
from torch.utils.data import Dataset

from datasets import register
from utils import to_pixel_samples

@register("coz_wrapper")
class coz_wrapper(Dataset):
    def __init__(self, dataset, cell_decode = False):
        self.dataset = dataset
        self.cell_decode = cell_decode

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        """
            Return :
                lr_image, hr_image (flatten), 
                hr_coord (shape),
                focal_length,
                cell (divided hr image shape)
        """
        hr_image = self.dataset[idx]["hr_image"]
        hr_coord, flatten_hr = to_pixel_samples(hr_image)

        if self.cell_decode is True:
            cell = torch.ones_like(hr_coord)
            cell[:, 0] *= 2 / hr_image.shape[-2]
            cell[:, 1] *= 2 / hr_image.shape[-1]

        return {
            "lr_image": self.dataset[idx]["lr_image"],
            "hr_image": flatten_hr,
            "hr_coord": hr_coord,
            "focal_length": self.dataset[idx]["focal_length"],
            "cell": cell if self.cell_decode else None
        }
        
if __name__ == "__main__":
    from torchsummary import summary
    import ipdb

    from .datasets import make 

    dataset_args = {
        'name': "coz_folder",
        'args':{
            'root_path' : "data/coz_data/train_LR"
        }
    }
    
    wrapper_args = {
        'name': 'coz_wrapper',
        'args': {
            'dataset': None,
            'cell_decode': True
        }
    }

    dataset = make(dataset_args)
    dataset = make(wrapper_args, args = {"dataset": dataset})
    ipdb.set_trace()

   