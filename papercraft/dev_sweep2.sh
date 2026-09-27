#!/bin/bash
set -e
source /home/ubuntu/pcenv/bin/activate
cd /home/ubuntu/kushulan-papercut-blora/papercraft
R=/home/ubuntu/kushulan-papercut-blora/outputs
DEV=$R/contents_dev/contents.json
S="--contents_json $DEV --seeds 0 1 --cn_scale 0.6"
python sample.py --method prompt_cn $S --out $R/dev/prompt_cn
for m in coll cutcraft_pd; do
  for g in 0.2 0.5 1.0; do
    python sample.py --method coll_cn --lora $R/style/$m --guide --guide_strength $g $S --out $R/dev/${m}_g$g
  done
done
python sample.py --method coll_cn --lora $R/style/cutcraft_pd $S --out $R/dev/cutcraft_pd_g0
python sample.py --method coll_cn --lora $R/style/coll_self $S --out $R/dev/coll_self_g0
echo GEN DONE
