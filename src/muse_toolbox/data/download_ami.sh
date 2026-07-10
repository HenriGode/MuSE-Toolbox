#!/bin/bash

# Exit on any error
set -e

# Define the target directory where the AMI corpus will be stored
TARGET_DIR="/data4/Henri/j3/framewiseSpeakerCounting/databases/ami"

# Create the directory if it doesn't exist
mkdir -p "${TARGET_DIR}"

# Navigate to the target directory
cd "${TARGET_DIR}"

echo "Starting download of AMI Corpus meeting recordings..."
echo "Target directory: ${TARGET_DIR}"
echo "This might take a while depending on your internet connection."

# Use wget to recursively download all files from the official mirror
# -c: continue getting a partially-downloaded file
# -r: turn on recursive retrieving
# -np: do not ascend to the parent directory
# -nH: disable generation of host-prefixed directories
# --cut-dirs=3: ignore the /ami/AMICorpusMirror/amicorpus/ path component
# -R "*.avi": reject all files with extension .avi (and by extension .mp4, .rm, etc)
wget -c -r -np -nH --cut-dirs=3 -R "*.avi" "https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus/"

echo "AMI Corpus recordings download complete!"
