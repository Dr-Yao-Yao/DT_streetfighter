for rtg in 45 
do
    for epoch in  29 15 0 3 5 10
    do
        for diff in 0 1 2 3
        do
            python run_style_pretrained_dt.py --rtg $rtg --renderrate 20 --difficulty $diff --epoch $epoch --eval_times 20 --seed 123 --context_length 30 --model_type 'reward_conditioned' --game 'Boxing'
        done
    done
done
