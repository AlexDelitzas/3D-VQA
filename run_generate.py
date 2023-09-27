import os
import os.path as osp
import time

from lib.config import CONF

# Constants and statics
DATA_DIR = './data/'
SCANS_DIR = osp.join(DATA_DIR, 'scannet/scans')
OUTPUT_DIR = CONF.PATH.TOPDOWN

if not osp.isdir(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

SPLITS = ["val", "train"]

print(f"Generating train and validation topview images. "
      f"Saving them in {OUTPUT_DIR}/ .\n")

for split in SPLITS:

    print(f"\nGenerating {split} topview images")
    print("-----------------------------------")
    print("Sleeping for 5 Seconds. Afterwards launch python threads.")
    time.sleep(5)

    if not osp.isdir(osp.join(OUTPUT_DIR, split)):
        os.makedirs(osp.join(OUTPUT_DIR, split))
        print(f"\t mkdir {osp.join(OUTPUT_DIR, split)}")

    # Path to scanrefer file
    scanrefer = osp.join(DATA_DIR, f"ScanRefer_filtered_{split}.txt")

    with open(scanrefer, 'r') as file1:
        lines = file1.read().splitlines()

    for scene in lines:
        os.system("python generate_top_down.py --scene " + scene + f" --scans {SCANS_DIR} --output {OUTPUT_DIR}/{split}")

