#!/usr/bin/env bash

submit_func ()
{

# Setup
local user=`whoami`
local starting_dir=`pwd`

# Set unique folder on temp for running job
local dir_name=enter_dir_name_here

hostname >> $starting_dir/nodeloc.txt
local working_dir=/tmp/$user/$dir_name/
local final_dir=$starting_dir/output/
mkdir -p $final_dir

echo "" >> $starting_dir/nodeloc.txt
echo "starting directory = $starting_dir" >> $starting_dir/nodeloc.txt
echo "working directory = $working_dir" >> $starting_dir/nodeloc.txt
echo "final directory = $final_dir" >> $starting_dir/nodeloc.txt
echo "----------------------------------"
echo "" >> $starting_dir/nodeloc.txt

# Make working directory
mkdir -p $working_dir
cp -a $starting_dir/* $working_dir
cd $working_dir
echo $(pwd) >> $starting_dir/nodeloc.txt

# Run job
source /data/karvelis03/dl_kcat/.env/bin/activate
python $working_dir/lstm_1.py $working_dir/lstm_1_config.txt > py_output.out


# Cleanup
echo "" >> $starting_dir/nodeloc.txt
echo "Copying all results back..." >> $starting_dir/nodeloc.txt
mkdir -p $final_dir
#if rsync -ar $working_dir $final_dir # Not all remote nodes have rsync installed
if cp -a $working_dir/* $final_dir
then
    echo "Removing working directory..." >> $starting_dir/nodeloc.txt
    rm -rf $working_dir
else
    echo "Copy was not successful." >> $starting_dir/nodeloc.txt
fi
echo "Job Complete!!!" >> $starting_dir/nodeloc.txt


}

submit_func
