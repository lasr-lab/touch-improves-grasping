from lerobot.datasets.dataset_tools import merge_datasets
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# Paths to your datasets
dataset1_path = LeRobotDataset("/lerobot_side/")
dataset2_path = LeRobotDataset("/lerobot_top/")

# Path for the output merged dataset
output_dataset_path = "/lerobot_combined/"

# Merge the datasets
merged_dataset = merge_datasets([dataset1_path, dataset2_path], output_dataset_path)

# Save the merged dataset
merged_dataset.save_to_disk(output_dataset_path)

print(f"Merged dataset saved to: {output_dataset_path}")
