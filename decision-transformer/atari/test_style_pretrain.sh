for rtg in 0 
do
    for epoch in  0
    do
        for diff in 3
        do
            python run_style_pretrained_dt.py --render --rtg $rtg --renderrate 20 --difficulty $diff --epoch $epoch --eval_times 1 --seed 123 --context_length 30 --model_type 'reward_conditioned' --game 'Boxing'
        done
    done
done
