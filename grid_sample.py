from torchvision import transforms
import torch
import torch.nn.functional as F
from PIL import Image
from PIL import ImageFile
import ipdb

def test():
    # single channel testing
    input = torch.arange(16).view(1, 1, 4, 4).float()
    print(input)


    d = torch.linspace(-1, 1, 8)

    xx, yy = torch.meshgrid((d, d), indexing="xy")
    grid = torch.stack((xx, yy), 2)
    grid = grid.unsqueeze(0)

    output = F.grid_sample(input, grid, mode='nearest', align_corners=False)
    print(output)

    print(output.shape)
    print(output[:, :, 0, :].permute(0, 2, 1))
    ipdb.set_trace()


def image_test():
    # image testing
    img_path = "test.png"

    image = Image.open(img_path)
    image = image.convert('RGB')

    img_transforms = transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    image = img_transforms(image)

    # add batch dimension
    image = image.unsqueeze(dim = 0)

    d = torch.linspace(-1, 1, 338)

    meshx, meshy = torch.meshgrid((d, d), indexing = "xy")
    grid = torch.stack((meshx, meshy), 2)

    grid = grid.unsqueeze(0)
    warped = F.grid_sample(image, grid, mode = 'nearest', align_corners=True)

    print(warped)

    to_image = transforms.ToPILImage()
    to_image(image.squeeze()).show()
    to_image(warped.squeeze()).show(title = "Warped")

test()