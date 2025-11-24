#!/usr/bin/env python3
"""Create ultra-tiny versions of all supporting datasets for chatbot functionality"""

import pandas as pd
from pathlib import Path
from scipy.sparse import load_npz, save_npz
import numpy as np
from collections import Counter

def create_ultra_tiny_supporting_datasets():
    """Create ultra-tiny versions of idf, word_index, tf_idf_matrix, and unique_df"""
    
    print("🔧 Creating ultra-tiny supporting datasets...")
    
    # Load the ultra_tiny main dataset to get the IDs we need
    if not Path("Dataset/ultra_tiny.csv").exists():
        print("❌ Dataset/ultra_tiny.csv not found. Run create_tiny_dataset_simple.py first.")
        return False
    
    df_ultra = pd.read_csv("Dataset/ultra_tiny.csv")
    print(f"✅ Loaded ultra_tiny dataset: {len(df_ultra):,} rows")
    
    # Get unique IDs from ultra_tiny dataset
    ultra_ids = set(df_ultra['Id'].unique())
    print(f"📊 Ultra-tiny dataset contains {len(ultra_ids):,} unique question IDs")
    
    # Load original supporting datasets
    try:
        print("\n📁 Loading original supporting datasets...")
        
        # Load full datasets
        df_full = pd.read_csv("Dataset/full_cleaned.csv", usecols=['Id', 'Score_question', 'question', 'Score_answer', 'Body_answer'])
        unique_df_full = pd.read_csv("Dataset/unique_df.csv")
        word_index_df = pd.read_csv('Dataset/word_to_index.csv', keep_default_na=False)
        idf_df = pd.read_csv('Dataset/idf.csv', keep_default_na=False)
        tf_idf_matrix = load_npz("Dataset/sparse_matrix.npz")
        
        print(f"✅ Full dataset: {len(df_full):,} rows")
        print(f"✅ Full unique_df: {len(unique_df_full):,} rows")
        print(f"✅ Word index: {len(word_index_df):,} words")
        print(f"✅ IDF scores: {len(idf_df):,} words")
        print(f"✅ TF-IDF matrix: {tf_idf_matrix.shape}")
        
    except Exception as e:
        print(f"❌ Error loading original datasets: {e}")
        return False
    
    # Create ultra-tiny unique_df by filtering to only IDs in ultra_tiny
    print("\n🔄 Creating ultra-tiny unique_df...")
    # For full_cleaned.csv, we need to create unique_df with 'question' column renamed to 'Body_question'
    unique_df_full_renamed = unique_df_full.rename(columns={'question': 'Body_question'}) if 'question' in unique_df_full.columns else unique_df_full
    unique_df_ultra = unique_df_full_renamed[unique_df_full_renamed['Id'].isin(ultra_ids)].copy()
    unique_df_ultra = unique_df_ultra.reset_index(drop=True)
    
    # Save ultra-tiny unique_df
    unique_df_ultra.to_csv("Dataset/ultra_tiny_unique_df.csv", index=False)
    print(f"✅ Created ultra_tiny_unique_df.csv: {len(unique_df_ultra):,} rows")
    
    # Create mapping from old row indices to new row indices
    print("\n🔄 Creating row index mapping...")
    old_to_new_mapping = {}
    for new_idx, row in unique_df_ultra.iterrows():
        # Find the original row index in unique_df_full
        original_matches = unique_df_full[unique_df_full['Id'] == row['Id']]
        if not original_matches.empty:
            old_idx = original_matches.index[0]
            old_to_new_mapping[old_idx] = new_idx
    
    print(f"✅ Created index mapping: {len(old_to_new_mapping):,} entries")
    
    # Extract corresponding rows from TF-IDF matrix
    print("\n🔄 Extracting ultra-tiny TF-IDF matrix...")
    old_indices = list(old_to_new_mapping.keys())
    
    # Extract rows corresponding to ultra_tiny questions
    tf_idf_matrix_ultra = tf_idf_matrix[old_indices]
    
    # Save ultra-tiny TF-IDF matrix
    save_npz("Dataset/ultra_tiny_sparse_matrix.npz", tf_idf_matrix_ultra)
    print(f"✅ Created ultra_tiny_sparse_matrix.npz: {tf_idf_matrix_ultra.shape}")
    
    # Copy word index and IDF as-is (they contain all vocabulary)
    print("\n📋 Copying word index and IDF files...")
    
    # Copy word_to_index.csv as ultra_tiny version
    word_index_df.to_csv("Dataset/ultra_tiny_word_to_index.csv", index=False)
    print(f"✅ Created ultra_tiny_word_to_index.csv: {len(word_index_df):,} words")
    
    # Copy idf.csv as ultra_tiny version
    idf_df.to_csv("Dataset/ultra_tiny_idf.csv", index=False)
    print(f"✅ Created ultra_tiny_idf.csv: {len(idf_df):,} words")
    
    # Verify the ultra-tiny datasets work together
    print("\n🧪 Verifying ultra-tiny datasets...")
    
    try:
        # Test loading all ultra-tiny datasets
        df_test = pd.read_csv("Dataset/ultra_tiny.csv")
        unique_df_test = pd.read_csv("Dataset/ultra_tiny_unique_df.csv")
        word_index_test = pd.read_csv("Dataset/ultra_tiny_word_to_index.csv")
        idf_test = pd.read_csv("Dataset/ultra_tiny_idf.csv")
        tf_idf_test = load_npz("Dataset/ultra_tiny_sparse_matrix.npz")
        
        print(f"✅ Main dataset: {len(df_test):,} rows")
        print(f"✅ Unique questions: {len(unique_df_test):,} rows")
        print(f"✅ TF-IDF matrix: {tf_idf_test.shape}")
        print(f"✅ Word vocabulary: {len(word_index_test):,} words")
        print(f"✅ IDF scores: {len(idf_test):,} words")
        
        # Verify dimensions match
        if tf_idf_test.shape[0] == len(unique_df_test):
            print("✅ Matrix dimensions match unique questions count")
        else:
            print(f"❌ Dimension mismatch: TF-IDF has {tf_idf_test.shape[0]} rows, unique_df has {len(unique_df_test)} rows")
        
        return True
        
    except Exception as e:
        print(f"❌ Verification failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Creating ultra-tiny supporting datasets for StackBot...\n")
    
    success = create_ultra_tiny_supporting_datasets()
    
    if success:
        print("\n🎉 Successfully created all ultra-tiny supporting datasets!")
        print("\n📁 Ultra-tiny dataset files created:")
        print("  • Dataset/ultra_tiny.csv (main dataset)")
        print("  • Dataset/ultra_tiny_unique_df.csv (unique questions)")
        print("  • Dataset/ultra_tiny_sparse_matrix.npz (TF-IDF matrix)")
        print("  • Dataset/ultra_tiny_word_to_index.csv (word index)")
        print("  • Dataset/ultra_tiny_idf.csv (IDF scores)")
        print("\n💡 Update streamlit_app.py to use these ultra_tiny_* files for full chatbot functionality!")
    else:
        print("\n⚠️ Failed to create ultra-tiny supporting datasets.")
        print("Check the error messages above and ensure all original datasets are present.")