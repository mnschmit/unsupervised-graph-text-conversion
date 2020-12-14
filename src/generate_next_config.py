#!/usr/bin/env python3

import json
import sys
import re

with open(sys.argv[1]) as f:
    config = json.load(f)


match = re.search(r'bt([0-9]+)', config["train_data_path"])
if match is None:
    print(
        "train_data_path has invalid format:",
        config["train_data_path"],
        file=sys.stderr
    )
    sys.exit(-1)

num_iter = match.group(1)    
next_iter = int(num_iter) + 1
new_data_path = re.sub(
    r'bt[0-9]+', "bt{}".format(next_iter), config["train_data_path"]
)
config["train_data_path"] = new_data_path

new_config_file = re.sub(
    r'iter[0-9]+\.json$', 'iter{}.json'.format(next_iter), sys.argv[1]
)
assert new_config_file != sys.argv[1]

with open(new_config_file, 'w') as fout:
    json.dump(config, fout)

print(new_config_file)
