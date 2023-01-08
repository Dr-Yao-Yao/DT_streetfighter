python run_style_dt_atari.py --seed 520 --epochs 30 --num_steps 300000 --trajectories_per_buffer 10 --mask --num_buffers 30 --game 'Boxing' --batch_size 128 --data_dir_prefix './dqn_replay/'
python run_style_dt_atari.py --seed 520 --epochs 30 --num_steps 300000 --trajectories_per_buffer 10 --num_buffers 30 --game 'Boxing' --batch_size 128 --data_dir_prefix './dqn_replay/'
for rtg in 0 30 60 90 
do
    for epoch in  0 3 5 15 29
    do
        for diff in 0 1 2 3
        do
            python run_style_pretrained_dt.py --rtg $rtg --renderrate 20 --difficulty $diff --epoch $epoch --eval_times 20 --seed 123 --context_length 30 --model_type 'reward_conditioned' --game 'Boxing'
        done
    done
done
