#!/usr/bin/env bash

kaggle_arc_dir="/kaggle/input/abstraction-and-reasoning-challenge"
kaggle_csv_file="/kaggle/working/submission.csv"
lab42_arc_dir="/data"
lab42_solution_dir="/data/solution"

################################################################################
# 
rm $lab42_solution_dir/*.json
cd /kaggle/working/

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
# python arc_crop.py

# g++ -pthread -lpthread -O3 -std=c++17 -o m3_t7 m3_t7.cpp
# ./m3_t7
# python3 m3_t7.py
# cp submission_top3_tree7.csv submission_ensemble.csv

# TOP 8
cd /kaggle/working/ 
echo "Running TOP 8 ..."
python top8_decision_trees.py
# submission_top8_tree.csv

# TOP 10
cd /kaggle/working/
echo "Running TOP 10 ..."
cd /kaggle/working/arc-top10-source-files
cp ../mt10_run.py .
pip install . --no-color --verbose --no-deps --disable-pip-version-check
python mt10_run.py
cp submission_top10.csv ..
# submission_top10.csv

# TOP 3
echo "Running TOP 3 ..."
cd /kaggle/working/
python m3_d8.py
python m3_d5.py
python3 m3_zoltan.py
# submission_top3_dsl8.csv
# submission_top3_dsl5_r1.csv, submission_top3_dsl5_r2.csv
# submission_top3_zoltan.csv

# submission_top10.csv
cp submission_top10.csv submission_ensemble.csv

################################################################################

################################################################################
# Convert from submission.csv to the json file
cp $kaggle_arc_dir/*.csv $lab42_solution_dir/
cp *.csv $lab42_solution_dir/

python arc_post.py
rm $lab42_solution_dir/*.csv
################################################################################

################################################################################
# Inform the evaluation process that you're done
echo "Program has finished!"
################################################################################
