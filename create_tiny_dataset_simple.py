"""
Simple dataset sampler that works without pandas
Creates a 10% sample of the CSV file using basic Python
"""

import csv
import random
from pathlib import Path

def create_tiny_dataset_simple():
    """Create 10% sample using basic Python CSV tools"""
    
    input_file = "Dataset/half_cleaned.csv"
    output_file = "Dataset/tiny_cleaned.csv" 
    
    if not Path(input_file).exists():
        print(f"❌ {input_file} not found")
        return False
    
    print(f"📖 Reading {input_file}...")
    
    # Read all rows
    rows = []
    with open(input_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)  # Get header
        rows = list(reader)    # Get all data rows
    
    original_count = len(rows)
    print(f"Original dataset: {original_count:,} rows")
    
    # Calculate 10% sample size (minimum 1000 rows)
    sample_size = max(1000, int(original_count * 0.1))
    sample_size = min(sample_size, original_count)  # Don't exceed available rows
    
    print(f"Creating sample: {sample_size:,} rows ({sample_size/original_count*100:.1f}%)")
    
    # Create systematic sample (every Nth row)
    step = max(1, original_count // sample_size)
    sampled_rows = rows[::step][:sample_size]
    
    print(f"Selected {len(sampled_rows):,} rows")
    
    # Write sampled data
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(header)  # Write header
        writer.writerows(sampled_rows)  # Write sampled data
    
    # Calculate file sizes
    original_size = Path(input_file).stat().st_size / 1024**2
    new_size = Path(output_file).stat().st_size / 1024**2
    reduction = ((original_size - new_size) / original_size) * 100
    
    print(f"✅ Created {output_file}")
    print(f"📊 File size: {original_size:.1f}MB → {new_size:.1f}MB ({reduction:.1f}% reduction)")
    
    return True

def create_ultra_tiny_dataset():
    """Create an even smaller 1% dataset for testing"""
    
    input_file = "Dataset/half_cleaned.csv"
    output_file = "Dataset/ultra_tiny.csv"
    
    if not Path(input_file).exists():
        print(f"❌ {input_file} not found")
        return False
    
    print(f"📖 Creating ultra-tiny dataset (1% sample)...")
    
    # Read all rows
    rows = []
    with open(input_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)
    
    original_count = len(rows)
    sample_size = max(100, int(original_count * 0.01))  # 1% or minimum 100 rows
    sample_size = min(sample_size, original_count)
    
    # Systematic sampling
    step = max(1, original_count // sample_size)
    sampled_rows = rows[::step][:sample_size]
    
    # Write ultra-tiny dataset
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(sampled_rows)
    
    new_size = Path(output_file).stat().st_size / 1024**2
    print(f"✅ Created ultra-tiny dataset: {len(sampled_rows):,} rows, {new_size:.1f}MB")
    
    return True

if __name__ == "__main__":
    print("🚀 Creating memory-optimized datasets...")
    print("=" * 50)
    
    # Create 10% dataset
    success1 = create_tiny_dataset_simple()
    
    if success1:
        print("\n" + "=" * 50)
        # Create 1% dataset for ultra-low memory
        success2 = create_ultra_tiny_dataset()
        
        print("\n✅ Dataset creation complete!")
        print("\nFiles created:")
        if success1:
            print("• Dataset/tiny_cleaned.csv (10% sample)")
        if success2:
            print("• Dataset/ultra_tiny.csv (1% sample)")
        
        print("\n🎯 Your Streamlit app will automatically use the smallest available dataset.")
        print("💡 Try running your app now - memory usage should be dramatically reduced!")
    
    else:
        print("❌ Failed to create datasets")
        print("💡 Make sure Dataset/half_cleaned.csv exists")