#!/bin/bash

# Check if the correct number of arguments is provided
if [ $# -ne 2 ]; then
  echo "Usage: $0 <directory_path> <search_string>"
  exit 1
fi

directory_path="$1"
search_string="$2"

# Check if the specified directory exists
if [ ! -d "$directory_path" ]; then
  echo "Error: Directory not found: $directory_path"
  exit 1
fi

# Loop through each file in the directory
for file in "$directory_path"/*; do
  # Check if the file is a regular file (not a directory or a special file)
  if [ -f "$file" ]; then
    # Use grep to search for the string in the file
    if grep -i -q "$search_string" "$file"; then
      less "$file"
    fi
  fi
done