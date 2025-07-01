#!/usr/bin/env python
"""
Split a validation.json into N equal parts.

Usage:
    python split_validation.py <validation.json> <num_parts>

The script writes validation_part_0.json … validation_part_<N-1>.json
next to the original file and **prints** every new path to stdout
(one per line) so the caller can collect them.
"""
import json, math, pathlib, sys

if len(sys.argv) != 3:
    sys.exit("Usage: split_validation.py <validation.json> <num_parts>")

src         = pathlib.Path(sys.argv[1])
num_parts   = int(sys.argv[2])

with src.open() as fh:
    data = json.load(fh)["data"]

chunk = math.ceil(len(data) / num_parts)

for i in range(num_parts):
    part = data[i * chunk : (i + 1) * chunk]
    if not part:
        break
    out = src.with_name(f"validation_part_{i}.json")
    json.dump({"data": part}, out.open("w"))
    print(out)                         # picked up by the shell script