config='./pypolo2/configs/CONF.yaml'

pids=()

# Function to handle termination
cleanup() {
    echo "Terminating processes..."
    # Terminate all background processes
    for pid in "${pids[@]}"
    do
        kill $pid
    done
    exit 1
}

# Catch keyboard interrupt signal (SIGINT)
trap cleanup SIGINT

# for seed in 0 3 7 11 13 15 18 20 32 42
for seed in 0 3 7 11 13 15 18 20 32 42 \
            50 51 52 53 54 55 56 57 58 59 \
            60 61 62 63 64 65 66 67 68 69 \
            70 71 72 73 74 75 76 77 78 79 \
            80 81 82 83 84 85 86 87 88 89
do
    for strategy in SA_time_non_uniform
    do
        (
            python main.py --config $config --seed $seed --strategy_name $strategy --team_size 5 --sche_step 18
            python main.py --config $config --seed $seed --strategy_name $strategy --team_size 5 --sche_step 15
            python main.py --config $config --seed $seed --strategy_name $strategy --team_size 5 --sche_step 12

        ) &
        # Store the PID of the background process
        pids+=($!)
    done
done
wait