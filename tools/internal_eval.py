
import argparse
import pickle
import json
import os

import pandas

from tools.utils.time_utils import current_time
from mmdet.evaluation import CocoMetric



def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate coco metrics of the '
        'results saved in pkl format')
    parser.add_argument('gt_fpath', help='Ground truth file in json coco format')
    parser.add_argument('pred_fpath', help='Predictions file in pickle format')
    parser.add_argument('dst_dir', help='Results destination directory')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    with open(args.pred_fpath, "rb") as f:
        preds = pickle.load(f)

    with open(args.gt_fpath, "r") as f:
        categories = json.load(f)["categories"]
    categories = [category["name"] for category in categories]

    coco_metric = CocoMetric(
        ann_file=args.gt_fpath,
        metric=['bbox', 'segm'],
        classwise=False,
        outfile_prefix=args.dst_dir)
    
    coco_metric.dataset_meta = dict(classes=categories)

    for pred in preds:
        fname = pred["img_path"].split("/")[-1].split(".")[0]
        print(fname, end="\r")
        H,W = pred["ori_shape"]
        img_idx = pred["img_id"]
        coco_metric.process(
            {},
            [dict(pred_instances=pred["pred_instances"], img_id=img_idx, ori_shape=(H, W))])
    
    pred_size = len(preds)
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
