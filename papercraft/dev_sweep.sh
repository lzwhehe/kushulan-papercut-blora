#!/bin/bash
# Hyper-parameter selection on the 3 dev line arts only (never on test contents).
set -e
source /home/ubuntu/pcenv/bin/activate
cd /home/ubuntu/kushulan-papercut-blora/papercraft
R=/home/ubuntu/kushulan-papercut-blora/outputs
DEV=$R/contents_dev/contents.json
[ -f $DEV ] || python make_dev.py
S="--contents_json $DEV --seeds 0 1"
# 1) ControlNet scale for all ControlNet methods (collection style block, no guidance)
for cn in 0.4 0.6 0.8; do
  python sample.py --method coll_cn --lora $R/style/coll --cn_scale $cn $S --out $R/dev/cn$cn
done
# 2) CraftGuide strength on top of CutCraft (cn fixed later; run with 0.6 first)
for g in 0.1 0.2 0.4; do
  python sample.py --method cutcraft_cn --lora $R/style/cutcraft --cn_scale 0.6 --guide --guide_strength $g $S --out $R/dev/guide$g
done
python sample.py --method cutcraft_cn --lora $R/style/cutcraft --cn_scale 0.6 $S --out $R/dev/guide0
for d in $R/dev/*; do python evaluate.py --gen $d --contents_json $DEV; done
echo SWEEP DONE
