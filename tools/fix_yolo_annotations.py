import pathlib
from pathlib import Path
import argparse
from tqdm import tqdm
import numpy as np
from contextlib import ExitStack

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", "-f", type=str, default='/data/MOT17', help="path to data, having benchmark name.")
    parser.add_argument("--recursive", "-r", action='store_true', default=False, help="recursive file search")
    args = parser.parse_args()
    parent_folder = Path(args.folder)

    # check recursive label search
    if args.recursive:
        ann_files = list(parent_folder.rglob("*.txt"))
    else:
        ann_files = list(parent_folder.glob("*.txt"))

    # new path to store fixed labels
    fix_path = pathlib.Path(args.folder) / pathlib.Path("fixed_bbox_labels")
    oob_cntr = 0
    cntr = 0
    for ann_file in tqdm(ann_files):
        with ExitStack() as fs:  # Exit read and write file
            # open original label file
            read_file = open(ann_file, "r")
            # create and open fixed label file
            fix_fp = Path(fix_path / ann_file.relative_to(parent_folder))
            fix_fp.parent.mkdir(parents=True, exist_ok=True)
            write_file = open(fix_fp, "w+")
            rows = ([float(i) for i in line.split()] for line in read_file)
            fixed_lines = []
            for row in rows:
                # create constrained x1,y2,x2,y2 bbox (coco format bbox)
                fix_bbox = np.asarray(
                    [row[1] - (row[3] / 2), row[2] - (row[4] / 2), row[1] + (row[3] / 2), row[2] + (row[4] / 2)])
                oob_cntr += ((1 < fix_bbox) | (fix_bbox < 0)).any()
                cntr += 1
                fix_bbox = fix_bbox.clip(0.0, 1.0)
                # apply yolo format
                fix_bbox = np.asarray([(fix_bbox[0] + fix_bbox[2]) / 2, (fix_bbox[1] + fix_bbox[3]) / 2,
                                       fix_bbox[2] - fix_bbox[0], fix_bbox[3] - fix_bbox[1]])
                # create string t write into enw file
                fixed_lines.append(f"{int(row[0])} {' '.join(str(x) for x in fix_bbox)}\n")
            # write multiple lines
            write_file.writelines(fixed_lines)
    print(f"Labels: {cntr} | Out of bounds: {oob_cntr}")
