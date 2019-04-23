# These tasks are the fastest to learn. 'echo' and 'count-down' are very
# easy. run_eval_tasks.py will do most of the work to run all the jobs.
# Should take between 10 and 30 minutes.

# How many repetitions each experiment will run. In the paper, we use 25. Less
# reps means faster experiments, but noisier results.
REPS=25

# Extra description in the job names for these experiments. Use this description
# to distinguish between multiple runs of the same experiment.
DESC="simtest"

# The tasks to run.
TASKS="sort0"

# The model types and max NPE.
EXPS=( pg-20M )

# Where training data is saved. This is chosen by launch_training.sh. Custom
# implementations of launch_training.sh may use different locations.
# MODELS_DIR="/tmp/models"

# Run run_eval_tasks.py for each experiment name in EXPS.
for exp in "${EXPS[@]}"
do
  ./single_task/run_eval_tasks.py \
      --exp "$exp" --tasks $TASKS --desc "$DESC" --reps $REPS
done

