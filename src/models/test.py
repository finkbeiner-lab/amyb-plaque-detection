import torch
import wandb


wandb_config = dict(
        project='nps-ad-vivek',
        entity='hellovivek',
	    mode = "offline"
)


run = wandb.init(**wandb_config)
assert run is wandb.run # run was successfully initialized, is not None
print("Wandb Run Mode", run.settings.mode)
run_id, run_dir = run.id, run.dir
exp_name = run.name
run_id = 'test1'
artifact_name = f'{run_id}-logs'
artifact = wandb.Artifact(artifact_name, type='files')
run.log_artifact(artifact)
run.finish()


print("Device Count",torch.cuda.device_count())
print("Current Device", torch.cuda.current_device())
print('Device 0', torch.cuda.get_device_properties(0).total_memory)
print('Device 1', torch.cuda.get_device_properties(1).total_memory) 
# print('Device 2', torch.cuda.get_device_properties(2).total_memory) 
# print('Device 3', torch.cuda.get_device_properties(3).total_memory) 