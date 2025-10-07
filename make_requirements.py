import re

# Input file name containing the dependencies (or replace this with direct input text)
input_file = "./lmms-eval.reqs"
output_file = "./requirements.txt"

def clean_dependencies(input_file, output_file):
    with open(input_file, "r") as infile, open(output_file, "w") as outfile:
        for line in infile:
            # Skip empty lines or lines with local paths
            if not line.strip():
                continue
            
            # Match lines with valid package specifications
            match = re.match(r"^(\S+)\s+(\S+)(?!.*\s+/\S+)$", line)
            if match:
                package, version = match.groups()
                outfile.write(f"{package}=={version}\n")

# Run the function to clean and generate requirements.txt
clean_dependencies(input_file, output_file)

print(f"Cleaned requirements saved to {output_file}.")
