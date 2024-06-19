from typing import Literal
from Pydnet import Pydnet
from Config import Config
import torch
import torchvision.transforms.v2 as transforms
from PIL import Image
import matplotlib.pyplot as plt
import os
import torch.nn.functional as F
import time

image_to_single_batch_tensor = transforms.Compose(
    [
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
    ]
)


def tensor_resizer(
    tensor: torch.Tensor,
    width: int,
    height: int,
    mode: Literal["bicubic", "area"] = "area",
) -> torch.Tensor:
    """
    It resizes the tensor to specified width and height.
        `tensor`: torch.Tensor[B,C,H,W]
        `width`: output tensor width
        `height`: output tensor height
    Returns torch.Tensor[B,C,heigth,width]
    """
    tensor = tensor if tensor.size().__len__() > 3 else tensor.unsqueeze(0)
    tensor = F.interpolate(tensor, size=(height, width), mode=mode)
    return tensor


def from_image_to_tensor(img: Image) -> torch.Tensor:
    """
    Converts a pillow image to a torch.Tensor
        `img`: pillow image
    Returns image tensor.
    """
    img = img.convert("RGB")
    img_tensor: torch.Tensor = image_to_single_batch_tensor(img).unsqueeze(0)
    return img_tensor


def post_process_disparity(disp: torch.Tensor) -> torch.Tensor:
    """
    Post processing of the disparity tensor.
        `disp`: torch.Tensor[C,H,W]
    Returns the processed disparity tensor.
    """
    _, h, w = disp.shape
    l_disp = disp[0, :, :]
    r_disp = torch.fliplr(disp[1, :, :])
    m_disp = 0.5 * (l_disp + r_disp)
    l = torch.meshgrid(torch.linspace(0, 1, h), torch.linspace(0, 1, w), indexing="ij")[
        0
    ].to(disp.device)
    l_mask = 1.0 - torch.clamp(20 * (l - 0.05), 0, 1)
    r_mask = torch.fliplr(l_mask)
    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp


def use_with_path(env: Literal["HomeLab", "Cluster"], img_path: str):
    # Configurations and checks
    config = Config(env).get_configuration()
    if config.checkpoint_to_use_path == None or config.checkpoint_to_use_path == "":
        print("You have to select a checkpoint to correctly configure the model.")
        exit(0)
    if img_path == None or img_path == "":
        print("You have to select an image to create the corresponding depth heatmap.")
        exit(0)
    folder = os.path.dirname(img_path)
    filename = os.path.basename(img_path)
    name, ext = os.path.splitext(filename)
    depth_map_filename = f"{name}_depth_map{ext}"
    depth_map_path = os.path.join(folder, depth_map_filename)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model creation and configuration
    model = Pydnet().to(device)
    checkpoint = torch.load(
        config.checkpoint_to_use_path,
        map_location=("cuda" if torch.cuda.is_available() else "cpu"),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    try:
        with Image.open(img_path) as img:
            original_width, original_height = img.size
            start_time = time.time()
            disp_to_img = use(
                model,
                img,
                config.image_width,
                config.image_height,
                original_width,
                original_height,
                device,
            )
            elapsed_time = time.time() - start_time
            print(f"Execution time: {elapsed_time:.2f}s")
            # Salva l'immagine
            plt.imsave(depth_map_path, disp_to_img, cmap="plasma")

            print(f"Depth map salvata al seguente path:\n{depth_map_path}")

    except Exception as e:
        raise RuntimeError(f"Error loading image: {img_path}. {e}")


def use(
    model: torch.nn.Module,
    img: Image,
    downscale_width: int,
    downscale_height: int,
    original_width: int,
    original_height: int,
    device: torch.device,
) -> Image:
    """
    It will use a Pillow image as an input and provide a pillow depth map as an output.
        `model`: the Pydnet model used
        `img`: the image in the pillow Image type
        `downscale_width`: it's the width that the model accepts in the input
        `downscale_height`: it's the height that the model accepts in the input
        `original_width`: it's the width of the output image
        `original_height`: it's the height of the output image
        `device`: the device where the model it's hosted
    """
    model.eval()
    model.to(device)
    img_tensor = from_image_to_tensor(img).to(device)
    img_tensor = tensor_resizer(img_tensor, downscale_width, downscale_height)
    with torch.no_grad():
        img_disparities: torch.Tensor = model(img_tensor)[0][:, 0, :, :].unsqueeze(
            1
        )  # [1, 1, H, W]
        # pp_img_disparities = (
        #    post_process_disparity(img_disparities).unsqueeze(0).unsqueeze(0) # [1,1,H,W]
        # )
        img_disparities = (
            tensor_resizer(
                img_disparities, original_width, original_height, mode="bicubic"
            )
            # tensor_resizer(pp_img_disparities, original_width, original_height)
            .squeeze()
            .cpu()
            .numpy()
        )

    return Image.fromarray(img_disparities)
