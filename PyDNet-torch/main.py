import argparse
from training import train

parser = argparse.ArgumentParser(description="PyDNet pytorch implementation.")

parser.add_argument("--mode", type=str, help="[train,test,eval]", default="test")

args = parser.parse_args()

if args.mode == "train":
    train()
elif args.mode == "test":
    pass
elif args.mode == "eval":
    pass
else:
    print(
        "You inserted the wrong mode argument. Choose between: 'train', 'test' and 'eval'."
    )
