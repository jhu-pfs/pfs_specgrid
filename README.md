# Deep learning tools for astronomical spectroscopy














# Use with SLURM

## Run job interactively

	$ srun -p v100 --gpus=1 --cpu-per-task=12 ./scripts/train.sh ...
	
* -p: name of the partition of servers with GPUs
* --gpus: number of GPUs to allocate to learning process
* --cpu-per-task: number of CPU cores to allocate to learning process

## Submit to the queue

## Cancel slurm job

Get job id

	$ squeue
	
Output:
	
```
	             JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)
               125      v100 train.sh    dobos  R       0:28      1 v100
```

Cancel job gracefully by sending signal SIGUSR1 to the job (SIGINT kills). This will stop the
learning process at the end of the epoch and finish the job by executing evaluation notebooks.

	$ scancel 124 -s SIGUSR1
	
