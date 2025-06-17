import wandb

wandb.login(key="95ab75842f9b83eb3d3827739cdcb91239e81de7")  # or set WANDB_API_KEY env var if on remote

run = wandb.init(
    project="TwoTower",
    entity="week2_two_tower_neural_network",
    job_type="dataset-upload"
)

artifact = wandb.Artifact(
    name="msmarco-triplets",
    type="dataset",
    description="Pickled MSMARCO triplets (tokenized and raw)"
)

artifact.add_file(".data/BERTtokenized_triplets.pkl")
artifact.add_file(".data/msmarco_triplets.pkl")

run.log_artifact(artifact)
run.finish()