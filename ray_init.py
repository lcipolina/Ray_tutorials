import ray
import train_model_ray
import os

# Ensure all required environment variables are set
os.environ['RAY_AIR_NEW_OUTPUT'] = '0'



# Access SLURM resource variables and set default values if not defined
num_gpus = int(os.getenv('SLURM_GPUS_PER_TASK', 1))
num_cpus_per_task = int(os.getenv('SLURM_CPUS_PER_TASK', 1))
num_workers = int(os.getenv('SLURM_NTASKS', 1))

print(f"Number of GPUs per task defined on ray_init.py: {num_gpus}")
print(f"Number of CPUs per task: {num_cpus_per_task}")

# Replace with your fixed IP address
#HEAD_NODE_IP = "10.11.31.162"
#RAY_PORT = "6379"

# Initialize Ray with the fixed head node IP
#ray.init(address=f"{HEAD_NODE_IP}:{RAY_PORT}", include_dashboard=True, ignore_reinit_error=True, log_to_driver=True)



ray.init(
    address='auto',
    include_dashboard=True,
    ignore_reinit_error=True,
    log_to_driver=True
)



# Define the resources dictionary
config = {
    'num_workers': num_workers, # how many concurrent tasks (as per SLURM)
    'num_cpus': num_cpus_per_task,
    'num_gpus': num_gpus,
    'num_cpus_per_worker': num_cpus_per_task,
    'use_gpu': True,
    'checkpoint_freq': 5 # Save every 10 iterations
}


# Call the python function to run, passing the resources
train_model_ray.train_papyrus_sheet_detection(config)

# Shutdown Ray
ray.shutdown()
