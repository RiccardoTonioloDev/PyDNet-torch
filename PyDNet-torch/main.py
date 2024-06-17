import argparse
from training import train
from testing import generate_test_disparities

parser = argparse.ArgumentParser(description="PyDNet pytorch implementation.")

parser.add_argument(
    "--mode",
    type=str,
    help="[train,test,eval]\ntrain: trains the model;\ntest: generates a disparities.npy file to be used for evaluation;\neval: evaluates the disparities.npy file.",
    default="test",
)
parser.add_argument("--env", type=str, help="[HomeLab,Cluster]", default="HomeLab")

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
    pass
else:
    print(
        "You inserted the wrong mode argument in mode. Choose between: 'train', 'test' and 'eval'."
    )
