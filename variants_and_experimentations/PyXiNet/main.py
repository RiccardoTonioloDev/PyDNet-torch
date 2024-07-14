import argparse
from training import train
from testing import generate_test_disparities
from evaluating import eval_disparities_file
from using import use_with_path
from webcam import Webcam
import tkinter as tk
from PyXiNet import PyXiNet
from Config import Config

parser = argparse.ArgumentParser(description="PyDNet pytorch implementation.")

parser.add_argument(
    "--mode",
    type=str,
    help="[train,test,eval,use]\ntrain: trains the model;\ntest: generates a disparities.npy file to be used for evaluation;\neval: evaluates the disparities.npy file;\nuse: to use it do generate a disparity heatmap from an image.",
    default="test",
)
parser.add_argument("--env", type=str, help="[HomeLab,Cluster]", default="HomeLab")
parser.add_argument(
    "--img_path",
    type=str,
    help="The path of the image to be used to make its disparity heatmap.",
)
parser.add_argument(
    "--efficiency",
    type=str,
    help="[S,M,L]\nS is the smallest model, L is the largest.",
    default="L",
)

args = parser.parse_args()

if args.env not in ["HomeLab", "Cluster"]:
    print(
        "You inserted the wrong mode argument in env. Choose between: 'HomeLab' and 'Cluster'."
    )
    exit(0)
if args.efficiency not in ["S", "M", "L"]:
    print(
        "You inserted the wrong efficiency argument in env. Choose between: 'S', 'M' and 'L'."
    )
    exit(0)

model = PyXiNet(Config(args.env).get_configuration())

if args.mode == "train":
    train(args.env, model)
elif args.mode == "test":
    generate_test_disparities(args.env, model)
elif args.mode == "eval":
    eval_disparities_file(args.env)
elif args.mode == "use":
    use_with_path(args.env, args.img_path, model)
elif args.mode == "webcam":
    root = tk.Tk()
    app = Webcam(root, args.env, model)
    root.mainloop()
else:
    print(
        "You inserted the wrong mode argument in mode. Choose between: 'train', 'test', 'eval' and 'use'."
    )
