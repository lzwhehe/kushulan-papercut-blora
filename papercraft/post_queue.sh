#!/bin/bash
source /home/ubuntu/pcenv/bin/activate
cd /home/ubuntu/kushulan-papercut-blora/papercraft
R=/home/ubuntu/kushulan-papercut-blora/outputs
until grep -q "ALL DONE" $R/master_queue.log; do sleep 30; done
python sample.py --method coll_cn --lora $R/style/cutcraft_pd --contents_json $R/contents_dev/contents.json \
  --seeds 0 1 --cn_scale 0.6 --out $R/dev/cutcraft_pd_g0
sed -i 's#("dev/cutcraft_pd_g0.2", "projection self-distill.")#("dev/cutcraft_pd_g0", "projection self-distill.")#' fig_results.py
python evaluate.py --contents_json $R/contents_dev/contents.json --gen $R/dev/cutcraft_pd_g0
python make_dev_table.py
python fig_results.py qual fail
echo POST DONE
