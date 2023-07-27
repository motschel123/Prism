export NEPTUNE_TOKEN=
export NEPTUNE_PROJECT=
conda activate prism

DATE_TIME=$(date +%Y-%m-%d_%H_%M)
OUTPUT_FILE=/data/$USER/prism_log/run_$DATE_TIME.txt
PARAMS=("1 2 3" "4 5 6") # Insert parameters here

# Make sure neptune data is stored locally and not in the RRZE network 
mkdir -p /data/$USER/neptune
ln -s /data/$USER/neptune

echo PRISM > $FILE

for i in "${PARAMS[@]}"
DATE_TIME=$(date +%Y-%m-%d_%H_%M)
OUTPUT_FILE=/data/$USER/prism_log/run_$DATE_TIME.txt
do
    python -m workstation/run_dustin_generic.py $i > $OUTPUT_FILE 2>&1
done