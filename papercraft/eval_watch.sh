#!/bin/bash
# Score every finished generation directory (and its cut projection) on CPU.
source /home/ubuntu/pcenv/bin/activate
cd /home/ubuntu/kushulan-papercut-blora/papercraft
G=/home/ubuntu/kushulan-papercut-blora/outputs/gen
ALL="prompt_cn coll cutline cutcraft coll_guide cutcraft_noground cutline_self blora blora_style_cn blora_guide ksl_original"
touch $G/ksl_original/DONE
while true; do
  left=0
  for m in $ALL; do
    if [ -f $G/$m/DONE ]; then
      if [ ! -f $G/$m/metrics.csv ]; then
        nice -n 15 python evaluate.py --device cpu --gen $G/$m >> $G/eval.log 2>&1
      fi
      if [ ! -f $G/$m/metrics_projected.csv ]; then
        nice -n 15 python evaluate.py --device cpu --project --gen $G/$m >> $G/eval.log 2>&1
        echo "scored $m $(date)" >> $G/eval_done.log
      fi
    fi
    [ -f $G/$m/metrics_projected.csv ] || left=1
  done
  [ $left = 0 ] && break
  sleep 60
done
echo ALL SCORED >> $G/eval_done.log
