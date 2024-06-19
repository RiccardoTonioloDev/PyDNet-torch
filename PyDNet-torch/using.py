from typing import Literal
from Pydnet import Pydnet
from Config import Config
import torch
import torchvision.transforms.v2 as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os

image_to_single_batch_tensor = transforms.Compose(
    [
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
    ]
)


def from_image_to_tensor(img: Image) -> torch.Tensor:
    img = img.convert("RGB")
    img = img.resize((256, 512), Image.LANCZOS)
    img_tensor: torch.Tensor = image_to_single_batch_tensor(img)
    return img_tensor


def post_process_disparity(disp):
    _, h, w = disp.shape
    l_disp = disp[0, :, :]
    r_disp = np.fliplr(disp[1, :, :])
    m_disp = 0.5 * (l_disp + r_disp)
    l, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l_mask = 1.0 - np.clip(20 * (l - 0.05), 0, 1)
    r_mask = np.fliplr(l_mask)
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
    checkpoint = torch.load(config.checkpoint_to_use_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    try:
        with Image.open(img_path) as img:
            img_tensor = from_image_to_tensor(img)
            original_width, original_height = img.size
    except Exception as e:
        raise RuntimeError(f"Error loading image: {img_path}. {e}")

    img_tensor = img_tensor.to(device)

    disp_to_img = use(model, img_tensor, original_width, original_height)

    # Salva l'immagine
    plt.imsave(depth_map_path, disp_to_img, cmap="plasma")

    print(f"Depth map salvata al seguente path:\n{depth_map_path}")


def use(model: Pydnet, img: torch.Tensor, width: int, height: int) -> Image:
    with torch.no_grad():
        img_disparities: torch.Tensor = model(img.unsqueeze(0))[0].squeeze(0)
        pp_img_disparities = post_process_disparity(img_disparities.cpu().numpy())

    disp_to_img = Image.fromarray(pp_img_disparities)
    return disp_to_img.resize((width, height), Image.LANCZOS)
