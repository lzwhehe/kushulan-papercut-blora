#!/bin/bash
# Score round-3/4 generations on the CPU as soon as each finishes (one at a time, niced).
source /home/ubuntu/pcenv/bin/activate
cd /home/ubuntu/kushulan-papercut-blora/papercraft
G=/home/ubuntu/kushulan-papercut-blora/outputs/gen
O=/home/ubuntu/kushulan-papercut-blora/outputs
ALL="fulllora fulllora_guide fulllora_cl fulllora_cl_guide cl_dense cl_coarse cl_canny robust_broken robust_thick robust_thin robust_jitter oodall cutcraft_strict cutline_strict"
while true; do
  left=0
  for m in $ALL; do
    if [ -f $G/$m/DONE ] && [ ! -f $G/$m/SCORED ]; then
      CJ=""; CD=""
      if [ $m = oodall ]; then CJ="--contents_json $O/contents_oodall/contents.json"; CD=$O/contents_oodall; fi
      nice -n 15 python evaluate.py --device cpu --gen $G/$m $CJ >> $G/eval3.log 2>&1
      nice -n 15 python evaluate.py --device cpu --project --gen $G/$m $CJ >> $G/eval3.log 2>&1
      nice -n 15 python palette_usage.py $m >> $G/eval3.log 2>&1
      REV_CDIR=${CD:-$O/contents} nice -n 15 python revision_metrics.py $m >> $G/eval3.log 2>&1
      nice -n 15 python revision_metrics.py --embed $m >> $G/eval3.log 2>&1
      touch $G/$m/SCORED; echo "scored $m $(date)" >> $G/eval3_done.log
    fi
    [ -f $G/$m/SCORED ] || left=1
  done
  [ $left = 0 ] && break
  sleep 60
done
echo ALL SCORED >> $G/eval3_done.log
