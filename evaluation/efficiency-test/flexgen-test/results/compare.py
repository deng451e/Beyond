import os
import csv
from sentence_transformers import SentenceTransformer, util

# Initialize the SentenceTransformer model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Directory paths
base_dir = "./"  # Replace with your base directory
methods = ["beyond", "h2o", "infinigen", "flexgen"]

# Function to get files from a directory
def get_files(directory):
    return {file_name: os.path.join(directory, file_name) for file_name in os.listdir(directory) if file_name.endswith(".txt")}

# Collect all files
files = {method: get_files(os.path.join(base_dir, method)) for method in methods}

# Compare flexgen files with other methods
flexgen_files = files["flexgen"]
results = []

for file_name in flexgen_files:
    row = [file_name]
    flexgen_path = flexgen_files[file_name]

    # Read the content of the flexgen file
    with open(flexgen_path, 'r', encoding='utf-8') as f:
        flexgen_content = f.read()

    # Compare with other methods
    for method in ["beyond", "h2o", "infinigen"]:
        if file_name in files[method]:
            method_path = files[method][file_name]
            with open(method_path, 'r', encoding='utf-8') as f:
                method_content = f.read()
            
            # Compute similarity
            embedding_1 = model.encode(flexgen_content, convert_to_tensor=True)
            embedding_2 = model.encode(method_content, convert_to_tensor=True)
            similarity = util.pytorch_cos_sim(embedding_1, embedding_2).item()
            row.append(similarity)
        else:
            # File not found in this method
            row.append(-1)
    
    results.append(row)

# Write results to CSV
output_csv = "similarity_results.csv"
with open(output_csv, 'w', newline='', encoding='utf-8') as csv_file:
    writer = csv.writer(csv_file)
    # Write header
    writer.writerow(["File Name", "Beyond", "H2O", "Infinigen"])
    # Write data rows
    writer.writerows(results)

print(f"Results written to {output_csv}")

