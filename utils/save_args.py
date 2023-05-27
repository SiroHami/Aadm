import argparse
import json

def save_args(args, to_path):
  with open(to_path, "w") as f:
      json.dump(args.__dict__, f, indent=2)