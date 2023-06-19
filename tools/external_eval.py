
import argparse
import json
import os

import pandas
import numpy as np

import torch

from tools.utils.time_utils import current_time
from mmdet.evaluation import CocoMetric


from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate coco metrics of the '
        'results saved in numpy plus json format')
    parser.add_argument('gt_fpath', help='Ground truth file in json coco format')
    parser.add_argument('pred_dir', help='Predictions directory')
    parser.add_argument('dst_dir', help='Results destination directory')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    with open(args.gt_fpath, "r") as f:
        gt = json.load(f)
    categories = [category["name"] for category in gt["categories"]]
    img_ids = {image["file_name"][:-4]: image["id"] for image in gt["images"]}

    coco_metric = CocoMetric(
        ann_file=args.gt_fpath,
        metric=['bbox', 'segm'],
        classwise=False,
        outfile_prefix=args.dst_dir)
    
    coco_metric.dataset_meta = dict(classes=categories)

    flist = os.listdir(args.pred_dir)
    flist = [fname[:-4] for fname in flist if fname[-4:]==".npy"]
    flist.sort()

    for fname in flist:
        print(fname, end="\r")
        fpath = os.path.join(args.pred_dir,fname)
        array = np.load(fpath+".npy")
        with open(fpath+".json", "r") as f:
            meta = json.load(f)
        H,W = array.shape
        img_idx = img_ids[fname]
        inst_ids = np.unique(array[array>0])
        masks = []
        labels = []
        scores = []
        bboxes = []
        w_grid, h_grid = np.meshgrid(np.arange(W),np.arange(H))
        for inst_idx in inst_ids:
            mask = array == inst_idx
            overlap_ids = meta[str(inst_idx)]["ids"][1:]
            for overlap_idx in overlap_ids:
                mask += array == overlap_idx
            h_min, h_max = h_grid[mask].min(), h_grid[mask].max()+1
            w_min, w_max = w_grid[mask].min(), w_grid[mask].max()+1
            mask = mask.astype(np.uint8)
            masks.append(mask)
            labels.append(meta[str(inst_idx)]["cls"]-1)
            scores.append(meta[str(inst_idx)]["score"])
            bboxes.append([w_min,h_min,w_max,h_max])
        pred = {
            "masks": torch.from_numpy(np.array(masks)),
            "labels": torch.from_numpy(np.array(labels)),
            "scores": torch.from_numpy(np.array(scores)),
            "bboxes": torch.from_numpy(np.array(bboxes))
        }
        coco_metric.process(
            {},
            [dict(pred_instances=pred, img_id=img_idx, ori_shape=(H, W))])
    
    pred_size = len(flist)
    eval_results = coco_metric.evaluate(size=pred_size)

    results = pandas.DataFrame(eval_results,index=["avg"]).T
    stamp = current_time()+"_coco"
    if not os.path.exists(os.path.join(args.dst_dir,stamp)):
        os.makedirs(os.path.join(args.dst_dir,stamp))
    fpath = os.path.join(args.dst_dir,stamp,"summary.csv")
    results.to_csv(fpath, sep="\t", encoding="utf-8")
    print()
    print(results)
    print()


if __name__ == '__main__':
    main()
