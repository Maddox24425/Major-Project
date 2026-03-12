import splitfolders

# The folder that currently contains your two subfolders: NORMAL/ and PNEUMONIA/
input_folder = r"C:\Users\Maddox\Desktop\chest_xray" 

# Where you want the new structured dataset to be created
output_folder = "./dataset_split"

print("Splitting dataset into Train, Val, and Test...")

# This splits it into 70% for training, 15% for validation, and 15% for testing
splitfolders.ratio(
    input_folder, 
    output=output_folder, 
    seed=42, 
    ratio=(0.7, 0.15, 0.15), 
    group_prefix=None
)

print("Done! Your dataset is perfectly structured.")