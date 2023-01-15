#! /bin/bash


# to run the script write: ./avg.sh <model name> <checkpoint path> <prune type>
# for example: 
#	<model name>: VGG16_cifar10
#	<checkpoint path>: /mnt/disk_data/yaelf/saved/models/VGG16_Cifar10/0715_151237/checkpoint-epoch100.pth
#	<prune type>: random / l1 / lastN
num=5
# mask numbers bigger than 0.4 for random run
#for percent in 0 0.01 0.02 0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7
#for percent in 0 0.1 0.2 0.3 0.35 0.4 0.45 0.5 0.52 0.54 0.56 0.58 0.6 0.62 0.64 0.66 0.68 0.7
for percent in 0 0.02 0.05 0.1 0.15 0.2 0.25 \
0.3 0.32 0.34 0.36 0.38 0.4 0.42 0.44 0.46 0.48 0.5 0.52 0.54 0.56 0.58 0.6 \
0.62 0.64 0.66 0.68 0.7 0.72 0.74 0.76 0.78 0.8 0.82 0.84 0.86 0.88 0.9
do

   loss=0
   acc=0
   top5=0
   tpr=0
   tnr=0

   all_loss=""
   all_acc=""
   all_top5=""
   all_tpr=""
   all_tnr=""

   for (( c=1; c<=$num; c++ ))
   do

      ans=$(python test.py -r $2 -p $percent -t $3 | grep loss)
      echo $ans
      
      ans_loss=$(echo $ans | cut -d "," -f 1 | cut -d ":" -f 2)
      loss=$(echo "$loss + $ans_loss" | bc)
      all_loss+="$ans_loss"

      ans_acc=$(echo $ans | cut -d "," -f 2 | cut -d ":" -f 2)
      acc=$(echo "$acc + $ans_acc" | bc)
      all_acc+="$ans_acc"

      ans_top5=$(echo $ans | cut -d "," -f 3 | cut -d ":" -f 2)
      top5=$(echo "$top5 + $ans_top5" | bc)
      all_top5+="$ans_top5"

      ans_tpr=$(echo $ans | cut -d "," -f 4 | cut -d ":" -f 2)
      tpr=$(echo "$tpr + $ans_tpr" | bc)
      all_tpr+="$ans_tpr"

      ans_tnr=$(echo $ans | cut -d "," -f 5 | cut -d ":" -f 2 | cut -d "}" -f 1)
      tnr=$(echo "$tnr + $ans_tnr" | bc)
      all_tnr+="$ans_tnr"

   done
   
   loss_tot=$(echo "scale=4; x=$loss/$num; if(x<1) print 0; x" | bc)
   acc_tot=$(echo "scale=4; x=$acc/$num; if(x<1) print 0; x" | bc)
   top5_tot=$(echo "scale=4; x=$top5/$num; if(x<1) print 0; x" | bc)
   tpr_tot=$(echo "scale=4; x=$tpr/$num; if(x<1) print 0; x" | bc)
   tnr_tot=$(echo "scale=4; x=$tnr/$num; if(x<1) print 0; x" | bc)

   loss_err=$(python std.py $all_loss)
   acc_err=$(python std.py $all_acc)
   top5_err=$(python std.py $all_top5)
   tpr_err=$(python std.py $all_tpr)
   tnr_err=$(python std.py $all_tnr)

   #echo $loss_err $acc_err $top5_err $tpr_err $tnr_err

   per=$(echo "$percent * 100" | bc)
   total_line=$(echo $per"%" $loss_tot $acc_tot $top5_tot $tpr_tot $tnr_tot $loss_err $acc_err $top5_err $tpr_err $tnr_err)
   
   file_name=$(echo $1"_"$3".csv")
   echo $total_line >> $file_name
   
done





