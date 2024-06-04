import math
import os
import pathlib
from pathlib import Path
from typing import Dict
#import tempfile
import ray.train
import torch
from kornia import augmentation
from ray import tune
from ray import train
from ray.train import ScalingConfig, RunConfig
from ray.train.torch import TorchTrainer
from ray.train import Checkpoint,CheckpointConfig, Result
from torch.nn.parallel import DistributedDataParallel

from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from slow_but_save_dataset import SubvolumeDataset
from model import VisionTransformer

'''Model to detect the presence of sheet on an image of a rolled papyrus. Used for image segmentation to split sheet from noise '''

storage_path = "/p/fastdata/mmlaion/cipolina/VWTSegmentationPlayground/papyrus-sheet-detection/output"
checkpoint_path = storage_path
train_path = Path(
    "/p/fastdata/mmlaion/cipolina/papyrus/data/train")
test_path = Path(
    "/p/fastdata/mmlaion/cipolina/papyrus/data/test")

if not os.path.exists(checkpoint_path):
    os.makedirs(checkpoint_path)


input_shape = (32, 32, 32)
output_shape = (32, 32, 32)

direction_transforms = augmentation.AugmentationSequential(
    # Random flip along the horizontal axis with a probability of 0.5
    augmentation.RandomHorizontalFlip3D(p=0.5, same_on_batch=True),
    # Random flip along the vertical axis with a probability of 0.5
    augmentation.RandomVerticalFlip3D(p=0.5, same_on_batch=True),
    augmentation.RandomDepthicalFlip3D(p=0.5, same_on_batch=True)
)

noise_transforms = augmentation.VideoSequential(
    augmentation.RandomGaussianNoise(p=0.3),
    augmentation.RandomGaussianBlur(p=0.3, kernel_size=(3, 7), sigma=(0.1, 2.0)),
    data_format="BCTHW",
    same_on_frame=True
)


def get_dataloaders(batch_size):

    print(f"Entering get_dataloaders")

    training_data = SubvolumeDataset(
        input_path=train_path,
        voxel_shape=input_shape, label_shape=output_shape, stride=(8, 8, 8))

    test_data = SubvolumeDataset(
        input_path=test_path,
        voxel_shape=input_shape, label_shape=output_shape, stride=(8, 8, 8))

    # Create data loaders
    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True, drop_last=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, drop_last=True)

    return train_dataloader, test_dataloader


def train_function_per_worker(config: Dict):

    print(f"Entering train_function_per_worker")

    lr = config["lr"]
    epochs = config["epochs"]
    batch_size = config["batch_size_per_worker"]
    checkpoint_epoch_freq =  config['checkpoint_epoch_freq']

    # Get dataloaders inside the worker training function
    train_dataloader, test_dataloader = get_dataloaders(batch_size=batch_size)

    # [1] Prepare Dataloader for distributed training
    # Shard the datasets among workers and move batches to the correct device
    # =======================================================================
    train_dataloader = ray.train.torch.prepare_data_loader(train_dataloader)
    test_dataloader = ray.train.torch.prepare_data_loader(test_dataloader)

    model = VisionTransformer(shape=input_shape, output_shape=output_shape, d_model=512, n_heads=8, n_layers=12,
                              dim_feedforward=2048, patch_size=4, dropout=0.1)

    # [2] Prepare and wrap your model with DistributedDataParallel
    # Move the model to the correct GPU/CPU device
    # ============================================================

    print(f"Entering Prepare model")

    model = ray.train.torch.prepare_model(model)
    #direction_transforms_on_device = ray.train.torch.prepare_model(direction_transforms)
    #noise_transforms_on_device = ray.train.torch.prepare_model(noise_transforms)

    direction_transforms_on_device = direction_transforms
    noise_transforms_on_device = noise_transforms

    loss_fn = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    #TODO: review the step size
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.9) #Multiplies the learning rate with gamma every step_size (i.e.epochs).

    print(f"Entering training loop")


    # Model training loop
    best_loss = float('inf')  # Initialize with a high value (positive infinity)
    patience = 5
    epochs_without_improvement = 0
    for epoch in range(epochs):

        print(f"Current epoch: {epoch}")

        if ray.train.get_context().get_world_size() > 1:
            # Required for the distributed sampler to shuffle properly across epochs.
            train_dataloader.sampler.set_epoch(epoch)

        model.train()
        for X, y in tqdm(train_dataloader, desc=f"Train Epoch {epoch}"):
            X = X.unsqueeze(-4)
            y = y.unsqueeze(-4)

             # Apply transformations and augmentations
            X = direction_transforms_on_device(X)
            y = direction_transforms_on_device(y, params=direction_transforms_on_device._params)
            X = noise_transforms_on_device(X)

            # Remove the added dimension after processing
            X = X.squeeze(-4)
            y = y.squeeze(-4)

            # Forward pass through the model
            pred = model(X)
            loss = loss_fn(pred, y)

            # Backpropagation
            optimizer.zero_grad() # Clear gradients from the previous step
            loss.backward()       # Compute the derivatives of the loss wrt the parameters
            optimizer.step()      # Update parameters based on gradients

        # Call scheduler.step() after completing all batches in an epoch
        scheduler.step()  # Adjust the learning rate based on the epoch

        # Switch the model to evaluation mode.
        model.eval()
        test_loss, num_correct, num_total = 0, 0, 0
        with torch.no_grad():
            for X, y in tqdm(test_dataloader, desc=f"Test Epoch {epoch}"):
                X = X.unsqueeze(-4)   # Add a dimension to the input tensor to match the model's expected input shape.
                y = y.unsqueeze(-4)   # Similarly, adjust the label tensor's shape if necessary.

                # Apply the same transformations to the test data as were applied during training.
                X = direction_transforms_on_device(X)
                y = direction_transforms_on_device(y, params=direction_transforms_on_device._params)

                X = X.squeeze(-4)
                y = y.squeeze(-4)

                pred = model(X)
                loss = loss_fn(pred, y)
                test_loss += loss.item()
                num_total += math.prod(y.shape)

                num_correct += ((pred >= 0.5) == (y >= 0.5)).sum().item()

        # Report metrics
        test_loss /= len(test_dataloader)
        accuracy = num_correct / num_total

        # Checkpoint every `checkpoint_epoch_freq` epochs
        # ===============================
        if epoch % checkpoint_epoch_freq == 0:
            base_model = model.module if isinstance(model, DistributedDataParallel) else model
            checkpoint_data = {
                "epoch": epoch,
                "model_state_dict": base_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }
            # Save checkpoint directly to the specified directory
            checkpoint_file = os.path.join(checkpoint_path, f"checkpoint_epoch_{epoch}.pt")
            torch.save(checkpoint_data, checkpoint_file)

            # Report metrics and the new checkpoint to Ray Train
            train.report(
                metrics={"loss": test_loss, "accuracy": accuracy},
                checkpoint=Checkpoint.from_directory(checkpoint_path)
            )

        # Early stopping for training
        if test_loss < best_loss:
            best_loss = test_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        # Early stopping condition
        if epochs_without_improvement >= patience:
            print(f"Stopping early after {epoch} epochs due to no improvement")
            break

def train_papyrus_sheet_detection(resources):

    # Initialize Ray with the provided resources
    num_workers = resources.get('num_workers', 1) #number of concurrent tasks (matchign with SLURM)
    num_cpus    =resources.get('num_cpus', 8)
    num_gpus    =resources.get('num_gpus', 0)
    num_cpus_per_worker = resources.get('num_cpus_per_worker',1)
    use_gpu     = resources.get('use_gpu',False)


    global_batch_size = 64 #TODO: this is too small when we have lots of workers


    print(f"Number of workers: {num_workers}")

    lower, upper = 1e-5, 1e-1
    train_config = {
       # 'lr': tune.uniform(lower, upper),
        'lr': 0.01,
        'epochs': 5, #TODO: change this
        'batch_size_per_worker': 50, # global_batch_size // num_workers, #TODO: fix this
        'checkpoint_epoch_freq': 1
    }

    # Configure computation resources

    print(f"num_cpus_per_worker: {num_cpus_per_worker}")
    scaling_config = ScalingConfig(
        num_workers=num_workers,
        use_gpu=use_gpu,
        resources_per_worker={"CPU": num_cpus_per_worker , "GPU": 1 if use_gpu else 0}
    )

    # Set up the checkpoint configuration
    run_config = RunConfig(
        name="vit_run",
        verbose = 3,
        storage_path= storage_path,
        checkpoint_config=CheckpointConfig(num_to_keep=3) #keep only last 3 checkpoints to save space
    )

    # Initialize trainer. Checks wether there is an existing Ray cluster.

    print("About to enter TorchTrainer")
    trainer = TorchTrainer(
        train_loop_per_worker=train_function_per_worker,
        train_loop_config=train_config,
        scaling_config=scaling_config,
        run_config=run_config,
       )

    # [4] Start distributed training
    # Run `train_func_per_worker` on all workers
    # =============================================
    result = trainer.fit()

    # Save best results
    # =============================================
    print(f"Training result: {result}")


'''
if __name__ == "__main__":

    resources = {'num_gpus':1,
                'num_cpus': 7,
                'num_cpus_per_worker':12, # Optionally decide to allocate a quarter of your CPUs per worker
                'use_gpu':False
   }

    train_papyrus_sheet_detection(resources)
'''
