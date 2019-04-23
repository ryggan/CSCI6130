bazel build -c opt single_task:tune.par

# PG and TopK Tuning.
MAX_NPE=5000000
CONFIG="
env=c(task_cycle=['reverse-tune','remove-tune']),
agent=c(
  algorithm='pg',
  grad_clip_threshold=50.0,param_init_factor=0.5,entropy_beta=0.05,lr=1e-5,
  optimizer='rmsprop',ema_baseline_decay=0.99,topk_loss_hparam=0.0,topk=0,
  replay_temperature=1.0,alpha=0.0,eos_token=False),
timestep_limit=50,batch_size=64"

./single_task/launch_tuning.sh \
    --job_name "iclr_pg_gridsearch.reverse-remove" \
    --config "$CONFIG" \
    --max_npe "$MAX_NPE" \
    --num_workers_per_tuner 1 \
    --num_ps_per_tuner 0 \
    --num_tuners 1 \
    --num_repetitions 50 \
    --hparam_space_type "pg" \
    --stop_on_success true