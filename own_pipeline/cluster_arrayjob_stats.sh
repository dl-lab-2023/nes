#!/bin/bash

# Initialize counters
files_with_occurrence=0
total_files=0

# Check if the correct number of arguments is provided
if [ $# -ne 1 ]; then
  echo "Usage: $0 <moab_logs_directory_path>"
  exit 1
fi

moab_logs_directory_path="$1"
search_string="Finished at " # Will only be logged when 1) the cluster job didn't time out, and 2) the job was sucessful ("set -e" is included in each cluster script)

# Check if the specified directory exists
if [ ! -d "$moab_logs_directory_path" ]; then
  echo "Error: Directory not found: $moab_logs_directory_path"
  exit 1
fi

# Loop through each file in the directory
for file in "$moab_logs_directory_path"/*; do
  # Check if the file matches the glob (you can adjust the glob pattern as needed)
  if [[ "$file" == */Moab.o* ]]; then
    ((total_files++))
    # Check if the file is a regular file (not a directory or a special file)
    if [ -f "$file" ]; then
      # Use grep to search for the string in the file
      if grep -i -q "$search_string" "$file"; then
        ((files_with_occurrence++))
      else
        echo Failed job: $file
      fi
    fi
  fi
done

# Calculate percentage of files with search results
if [ "$total_files" -gt 0 ]; then
  percentage_with_occurrence=$((files_with_occurrence * 100 / total_files))
else
  percentage_with_occurrence=0
fi

echo ""
echo "Number of arrayjobs that were successful: $files_with_occurrence"
echo "Total number of arrayjobs: $total_files"
echo "Percentage of SUCCESSFUL arrayjobs: $percentage_with_occurrence%"
echo "Percentage of FAILED arrayjobs: $((100-$percentage_with_occurrence))%"
