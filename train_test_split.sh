#!/bin/bash

# Count b_ and nonblink_ files
b_count=$(ls b_* 2>/dev/null | wc -l)
nb_count=$(ls nonblink_* 2>/dev/null | wc -l)

echo "Number of b_ files: $b_count"
echo "Number of nonblink_ files: $nb_count"

# Create directories
mkdir -p train/b train/nb test/b test/nb

# Function to distribute files
distribute_files() {
    local pattern=$1
    local train_dir=$2
    local test_dir=$3
    
    # Get list of files
    files=($(ls $pattern 2>/dev/null))
    total_files=${#files[@]}
    
    if [ $total_files -eq 0 ]; then
        return
    fi
    
    # Calculate number of files for test set (20%)
    test_count=$((total_files * 20 / 100))
    
    # Shuffle files randomly
    shuffled_files=($(printf "%s\n" "${files[@]}" | shuf))
    
    # Copy files to test directory
    for ((i=0; i<$test_count; i++)); do
        cp "${shuffled_files[$i]}" "$test_dir/"
    done
    
    # Copy remaining files to train directory
    for ((i=$test_count; i<$total_files; i++)); do
        cp "${shuffled_files[$i]}" "$train_dir/"
    done
}

# Distribute b_ files
if [ $b_count -gt 0 ]; then
    echo "Distributing $b_count b_ files..."
    distribute_files "b_*" "train/b" "test/b"
    echo "  - $(ls test/b | wc -l) files copied to test/b"
    echo "  - $(ls train/b | wc -l) files copied to train/b"
fi

# Distribute nonblink_ files
if [ $nb_count -gt 0 ]; then
    echo "Distributing $nb_count nonblink_ files..."
    distribute_files "nonblink_*" "train/nb" "test/nb"
    echo "  - $(ls test/nb | wc -l) files copied to test/nb"
    echo "  - $(ls train/nb | wc -l) files copied to train/nb"
fi
