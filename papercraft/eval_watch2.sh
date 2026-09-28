#!/bin/bash
source /home/ubuntu/pcenv/bin/activate
cd /home/ubuntu/kushulan-papercut-blora/papercraft
G=/home/ubuntu/kushulan-papercut-blora/outputs/gen
ALL="abl_no_pal abl_no_flat abl_no_edge var_coll_self var_energy var_pd sens_g0.3 sens_g0.6 sens_g2.0 sens_cn0.4 sens_cn0.8"
while true; do
  left=0
  for m in $ALL; do
    if [ -f $G/$m/DONE ] && [ ! -f $G/$m/SCORED ]; then
      nice -n 15 python evaluate.py --device cpu --gen $G/$m >> $G/eval2.log 2>&1
      nice -n 15 python evaluate.py --device cpu --project --gen $G/$m >> $G/eval2.log 2>&1
      nice -n 15 python palette_usage.py $m >> $G/eval2.log 2>&1
      touch $G/$m/SCORED; echo "scored $m $(date)" >> $G/eval2_done.log
    fi
    [ -f $G/$m/SCORED ] || left=1
  done
  [ $left = 0 ] && break
  sleep 60
done
echo ALL SCORED >> $G/eval2_done.log
