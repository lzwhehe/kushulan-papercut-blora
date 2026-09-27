#!/bin/bash
set -e
source /home/ubuntu/pcenv/bin/activate
cd /home/ubuntu/kushulan-papercut-blora/papercraft
R=/home/ubuntu/kushulan-papercut-blora/outputs
DEV=$R/contents_dev/contents.json
S="--contents_json $DEV --seeds 0 1"
L="--method coll_cn --lora $R/style/cutline"
python sample.py $L $S --cn_scale 0.6 --out $R/dev3/cutline_cn0.6
python sample.py $L $S --cn_scale 0.8 --out $R/dev3/cutline_cn0.8
for g in 0.3 0.6 1.0; do
  python sample.py $L $S --cn_scale 0.6 --guide --guide_strength $g --out $R/dev3/cutline_g$g
done
python sample.py --method coll_cn --lora $R/style/coll $S --cn_scale 0.6 --guide --guide_strength 0.6 --out $R/dev3/coll_g0.6
echo GEN DONE
