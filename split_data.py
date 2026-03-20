import splitfolders

input_folder=r"C:\Users\Maddox\Desktop\chest_xray" 

output_folder="./dataset_split"

print("Splitting dataset into Train, Val, and Test...")

#70:15:15 ratio
splitfolders.ratio(
    input_folder, 
    output=output_folder, 
    seed=42, 
    ratio=(0.7, 0.15, 0.15), 
    group_prefix=None
)
print("Done")