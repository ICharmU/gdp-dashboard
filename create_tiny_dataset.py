import pandas as pd
import numpy as np
from pathlib import Path

def create_10_percent_dataset():
    """
    Create a 10% sample of the original dataset to reduce memory usage
    """
    input_file = "Dataset/half_cleaned.csv"
    output_file = "Dataset/tiny_cleaned.csv"
    
    print("Loading original dataset...")
    try:
        # First check the size of the original dataset
        df = pd.read_csv(input_file)
        original_size = len(df)
        original_memory = df.memory_usage(deep=True).sum() / 1024**2
        
        print(f"Original dataset: {original_size:,} rows, {original_memory:.1f} MB")
        
        # Calculate 10% sample size
        sample_size = max(1000, int(original_size * 0.1))  # At least 1000 rows
        
        print(f"Creating sample of {sample_size:,} rows ({sample_size/original_size*100:.1f}% of original)")
        
        # Create stratified sample to maintain quality distribution
        # Sample based on answer scores to keep high-quality answers
        if 'Score_answer' in df.columns:
            # Sort by answer score and take every 10th row (systematic sampling)
            df_sorted = df.sort_values('Score_answer', ascending=False)
            step = len(df_sorted) // sample_size
            if step < 1:
                step = 1
            df_sample = df_sorted.iloc[::step].head(sample_size)
            print("✅ Used systematic sampling based on answer scores")
        else:
            # Random sample if no score column
            df_sample = df.sample(n=sample_size, random_state=42)
            print("✅ Used random sampling")
        
        # Reset index
        df_sample = df_sample.reset_index(drop=True)
        
        # Calculate memory savings
        sample_memory = df_sample.memory_usage(deep=True).sum() / 1024**2
        memory_reduction = ((original_memory - sample_memory) / original_memory) * 100
        
        print(f"Sample dataset: {len(df_sample):,} rows, {sample_memory:.1f} MB")
        print(f"Memory reduction: {memory_reduction:.1f}%")
        
        # Save the sample
        df_sample.to_csv(output_file, index=False)
        print(f"✅ Saved sample dataset to {output_file}")
        
        # Create summary statistics
        print("\n📊 Sample Statistics:")
        if 'Score_answer' in df_sample.columns:
            print(f"Answer scores - Min: {df_sample['Score_answer'].min()}, Max: {df_sample['Score_answer'].max()}, Avg: {df_sample['Score_answer'].mean():.1f}")
        
        if 'Score_question' in df_sample.columns:
            print(f"Question scores - Min: {df_sample['Score_question'].min()}, Max: {df_sample['Score_question'].max()}, Avg: {df_sample['Score_question'].mean():.1f}")
            
        print(f"Columns: {list(df_sample.columns)}")
        
        return df_sample
        
    except FileNotFoundError:
        print(f"❌ Error: {input_file} not found")
        return None
    except Exception as e:
        print(f"❌ Error creating sample: {e}")
        return None

def optimize_tiny_dataset():
    """
    Further optimize the tiny dataset by reducing data types
    """
    try:
        df = pd.read_csv("Dataset/tiny_cleaned.csv")
        
        print("\n🔧 Optimizing data types...")
        original_memory = df.memory_usage(deep=True).sum() / 1024**2
        
        # Optimize integer columns
        int_columns = ['Id', 'Score_question', 'Score_answer']
        for col in int_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], downcast='integer')
        
        # Convert Tag to category if it exists
        if 'Tag' in df.columns:
            df['Tag'] = df['Tag'].astype('category')
        
        # Optimize string columns - remove extra whitespace
        string_columns = ['Title', 'Body_question', 'Body_answer']
        for col in string_columns:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()
        
        optimized_memory = df.memory_usage(deep=True).sum() / 1024**2
        type_reduction = ((original_memory - optimized_memory) / original_memory) * 100
        
        print(f"Memory after type optimization: {optimized_memory:.1f} MB")
        print(f"Additional reduction: {type_reduction:.1f}%")
        
        # Save optimized version
        df.to_csv("Dataset/tiny_optimized.csv", index=False)
        print("✅ Saved optimized tiny dataset to Dataset/tiny_optimized.csv")
        
        return df
        
    except Exception as e:
        print(f"❌ Error optimizing: {e}")
        return None

if __name__ == "__main__":
    print("🚀 Creating 10% dataset for memory optimization...")
    
    # Create 10% sample
    sample_df = create_10_percent_dataset()
    
    if sample_df is not None:
        # Further optimize the sample
        optimized_df = optimize_tiny_dataset()
        
        print("\n✅ Dataset reduction complete!")
        print("\nFiles created:")
        print("• Dataset/tiny_cleaned.csv - 10% sample")
        print("• Dataset/tiny_optimized.csv - 10% sample with optimized data types")
        print("\n🎯 Update your streamlit_app.py to use 'tiny_optimized.csv' for minimum memory usage")
    else:
        print("❌ Failed to create sample dataset")