import argparse
from training import train
from testing import generate_test_disparities
from evaluating import eval_disparities_file
from using import use

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

args = parser.parse_args()

if args.env not in ["HomeLab", "Cluster"]:
    print(
        "You inserted the wrong mode argument in env. Choose between: 'HomeLab' and 'Cluster'."
    )
    exit(0)

if args.mode == "train":
    train(args.env)
elif args.mode == "test":
    generate_test_disparities(args.env)
elif args.mode == "eval":
    eval_disparities_file(args.env)
elif args.mode == "use":
    use(args.env, args.img_path)
else:
    print(
        "You inserted the wrong mode argument in mode. Choose between: 'train', 'test', 'eval' and 'use'."
    )
