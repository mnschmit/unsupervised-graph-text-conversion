#!/usr/bin/env python3

import json
import sys

with open(sys.argv[1]) as f:
    config = json.load(f)

print("train_data_path:")
print(config["train_data_path"])
