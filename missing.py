#!/usr/bin/env python3
import os
import json

DIR = "ruofan_result"
START, END = 0, 198

missing = []

for i in range(START, END + 1):
    filename = os.path.join(DIR, f"{i}_result.json")
    if not os.path.isfile(filename):
        missing.append(i)
        continue
    try:
        with open(filename, "r") as f:
            data = json.load(f)
        sel = data.get("selected_agent", "")
        tru = data.get("true_answer", "")
        if not sel or not tru:
            missing.append(i)
    except Exception as e:
        # if JSON is malformed, also treat as missing
        missing.append(i)

# compress into ranges
def compress(nums):
    if not nums:
        return []
    nums = sorted(nums)
    ranges = []
    start = prev = nums[0]
    for n in nums[1:]:
        if n == prev + 1:
            prev = n
        else:
            if start == prev:
                ranges.append(str(start))
            else:
                ranges.append(f"{start}-{prev}")
            start = prev = n
    # flush last
    if start == prev:
        ranges.append(str(start))
    else:
        ranges.append(f"{start}-{prev}")
    return ranges

if not missing:
    print(f"All JSONs present and valid in '{DIR}' for {START}..{END}.")
else:
    ranges = compress(missing)
    print(f"Missing or invalid ({len(missing)} total): {', '.join(ranges)}")
