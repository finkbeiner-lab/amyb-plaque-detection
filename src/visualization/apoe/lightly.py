# execute the following code once to get a worker_id
from lightly.api import ApiWorkflowClient

client = ApiWorkflowClient(token="MY_LIGHTLY_TOKEN") # replace this with your token

# Create a Lightly Worker. If a worker with this name already exists, the id of the existing
# worker is returned.
worker_id = client.register_compute_worker(name="Default")
print(worker_id)