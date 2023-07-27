export NEPTUNE_TOKEN=
export NEPTUNE_PROJECT=
conda activate prism

# Arguments for runs
# Usage: rigid <n rigid phases> <rigid cov> <transition cov> <name>
# Usage: normal <name>
PARAMS=("normal no_rigid_phases" "rigid 30 0.001 0.0001 many_tiny_stops" "rigid 5 0.02 0.005 some_short_stops" "rigid 3 0.05 0.01 some_rigid_phases")

# Make sure neptune data is stored locally and not in the RRZE network 
mkdir -p /data/$USER/neptune
mkdir -p /data/$USER/prism_log
ln -s /data/$USER/neptune .neptune

echo PRISM > $FILE

for i in "${PARAMS[@]}"
do
    DATE_TIME=$(date +%Y-%m-%d_%H_%M)
    OUTPUT_FILE=/data/$USER/prism_log/run_$DATE_TIME.txt
    echo Performing run for parameters "$i"...
    echo Log is stored in $OUTPUT_FILE

    # Start run
    python -m workstation/run_dustin_generic.py $i > $OUTPUT_FILE 2>&1
done

echo Done!