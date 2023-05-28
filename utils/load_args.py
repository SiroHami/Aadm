import argparse
import json

def load_args(from_path):
  parser = argparse.ArgumentParser()
  args = parser.parse_args()
  with open(from_path, "r") as f:
      args.__dict__ = json.load(f)
  return args    