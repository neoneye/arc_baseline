#!/usr/bin/env bash

kaggle_arc_dir="/kaggle/input/abstraction-and-reasoning-challenge"
kaggle_csv_file="/kaggle/working/submission.csv"
lab42_arc_dir="/data"
lab42_solution_dir="/data/solution"

################################################################################
# 
cd /kaggle/working/
################################################################################

################################################################################
# Replace the kaggle test dir by the lab42 evaluation dir
mkdir -p $kaggle_arc_dir
unzip abstraction-and-reasoning-challenge.zip -d $kaggle_arc_dir

rm -rf $kaggle_arc_dir/test
rm -rf $kaggle_arc_dir/*.csv
cp -r $lab42_arc_dir/evaluation $kaggle_arc_dir/test
################################################################################

################################################################################
# Generate the sample_submission.csv
python arc_pre.py
################################################################################

################################################################################
# Your main code
python arc_crop.py
################################################################################

################################################################################
# Convert from submission.csv to the json file
cp $kaggle_arc_dir/*.csv $lab42_solution_dir/
cp *.csv $lab42_solution_dir/

python arc_post.py
rm $lab42_solution_dir/*csv
################################################################################

################################################################################
# Inform the evaluation process that you're done
echo "Program has finished!"
################################################################################
