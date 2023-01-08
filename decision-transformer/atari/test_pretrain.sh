for rtg in 10
do
    for difficulty in  3
    do
        python run_pretrained_dt.py --rtg $rtg --render --renderrate 20 --difficulty $difficulty --pretrained_epochs "0 " --eval_times 1 --seed 123 --context_length 30 --model_type 'reward_conditioned' --game 'Boxing'
    done
done