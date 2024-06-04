#!/bin/bash
#SBATCH --job-name=ray_vit
#SBATCH --account=cstdl
#SBATCH --partition=booster  #batch # devel
#SBATCH --nodes=1
#SBATCH --ntasks=4  #how many parallel tasks at the same time equals num_workers
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=10   #96 CPUs in Booster
#SBATCH --time=06:00:00     #02:00:00 max time allowed on devel, 6hrs on "batch" when no budget otherwise 24hs with budget
#SBATCH --output=ray_job_%j.out


# Load modules or source your Python environment

# Load the necessary modules
#module load Stages/2024
module load CUDA/12
#module load cuDNN/8.9.5.29-CUDA-12
#module load Intel/2023.2.1
#module load ParaStationMPI/5.9.2-1-mt
#module load CP2K/2023.2-mkl


# Update the environment with paths to the CUDA and cuDNN libraries
#export LD_LIBRARY_PATH=/p/software/juwelsbooster/stages/2024/software/cuDNN/8.9.5.29-CUDA-12/lib:$LD_LIBRARY_PATH
#export LD_LIBRARY_PATH=/p/software/juwelsbooster/stages/2024/software/cuDNN/8.9.5.29-CUDA-12/lib:/usr/local/cuda/lib64:$LD_LIBRARY_PATH


source /p/home/jusers/cipolina-kun1/juwels/vision_env/sc_venv_template/activate.sh


export RAY_AIR_NEW_OUTPUT=0
#export GLOO_SOCKET_IFNAME=eth0
#export RAY_ADDRESS="10.11.31.162:6379"



# Replace with your fixed IP address
#HEAD_NODE_IP="10.11.31.162"
# Start the Ray head node with the fixed IP address
#ray start --head --node-ip-address="$HEAD_NODE_IP" --port=6379 --block --verbose &

# Start the Ray head node - WORKING
#ray start --head --port=6379 --block --verbose &
#ray start --head --port=6379 --block --verbose --temp-dir=/p/fastdata/mmlaion/cipolina/ &

#Include dashboard
ray start --head --port=6379 --block --verbose --temp-dir=/p/fastdata/mmlaion/cipolina/ --dashboard-host 0.0.0.0 &


#ray start --head --node-ip-address=$(hostname) --port=6379 --block &
#sleep 10

# Sleep for a bit to ensure the head node starts properly
sleep 10

# Run your Python script
python -u /p/fastdata/mmlaion/cipolina/VWTSegmentationPlayground/papyrus-sheet-detection/ray_init.py

# Stop Ray when done
ray stop
