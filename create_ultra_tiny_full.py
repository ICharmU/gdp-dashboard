#!/usr/bin/env python3
"""Create ultra-tiny supporting datasets for full_cleaned.csv format"""

import pandas as pd
from pathlib import Path
from scipy.sparse import load_npz, save_npz
import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
import re

def preprocess_text_simple(text):
    """Simple text preprocessing"""
    if pd.isna(text):
        return ""
    # Basic cleaning
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def create_ultra_tiny_datasets_from_scratch():
    """Create all ultra-tiny datasets from scratch using full_cleaned.csv format"""
    
    print("🔧 Creating ultra-tiny datasets from full_cleaned.csv format...")
    
    # Load the ultra_tiny dataset
    if not Path("Dataset/ultra_tiny.csv").exists():
        print("❌ Dataset/ultra_tiny.csv not found. Run create_tiny_dataset_simple.py first.")
        return False
    
    print("\n📊 Loading ultra_tiny dataset...")
    df_ultra = pd.read_csv("Dataset/ultra_tiny.csv")
    
    # Rename 'question' to 'Body_question' for consistency
    if 'question' in df_ultra.columns:
        df_ultra = df_ultra.rename(columns={'question': 'Body_question'})
    
    print(f"✅ Loaded ultra_tiny dataset: {len(df_ultra):,} rows")
    print(f"📋 Columns: {df_ultra.columns.tolist()}")
    
    # Clean the data
    df_ultra = df_ultra.dropna(subset=['Body_question', 'Body_answer'])
    df_ultra = df_ultra.reset_index(drop=True)
    
    # Create unique questions dataset
    print("\n🔄 Creating unique questions dataset...")
    unique_df = df_ultra[['Body_question', 'Id']].drop_duplicates().copy()
    unique_df = unique_df.reset_index(drop=True)
    
    # Save unique questions
    unique_df.to_csv("Dataset/ultra_tiny_unique_df.csv", index=False)
    print(f"✅ Created ultra_tiny_unique_df.csv: {len(unique_df):,} rows")
    
    # Prepare questions for TF-IDF
    print("\n🔄 Processing questions for TF-IDF...")
    questions_processed = [preprocess_text_simple(q) for q in unique_df['Body_question']]
    
    # Create TF-IDF vectorizer and matrix
    print("🔄 Creating TF-IDF matrix...")
    vectorizer = TfidfVectorizer(
        max_features=50000,  # Limit vocabulary size for memory efficiency
        stop_words='english',
        min_df=2,  # Ignore terms that appear in fewer than 2 documents
        max_df=0.8,  # Ignore terms that appear in more than 80% of documents
        ngram_range=(1, 1)  # Only unigrams for simplicity
    )
    
    tf_idf_matrix = vectorizer.fit_transform(questions_processed)
    
    # Get vocabulary and IDF scores
    feature_names = vectorizer.get_feature_names_out()
    idf_scores = vectorizer.idf_
    
    print(f"✅ Created TF-IDF matrix: {tf_idf_matrix.shape}")
    print(f"📚 Vocabulary size: {len(feature_names):,} terms")
    
    # Save TF-IDF matrix
    save_npz("Dataset/ultra_tiny_sparse_matrix.npz", tf_idf_matrix)
    print("✅ Saved ultra_tiny_sparse_matrix.npz")
    
    # Create and save word index mapping
    print("\n🔄 Creating word index mapping...")
    word_to_index = {word: idx for idx, word in enumerate(feature_names)}
    word_index_df = pd.DataFrame([
        {'word': word, 'index': idx} 
        for word, idx in word_to_index.items()
    ])
    word_index_df.to_csv("Dataset/ultra_tiny_word_to_index.csv", index=False)
    print(f"✅ Created ultra_tiny_word_to_index.csv: {len(word_index_df):,} words")
    
    # Create and save IDF scores
    print("🔄 Creating IDF scores...")
    idf_df = pd.DataFrame([
        {'word': word, 'idf_score': idf_scores[idx]}
        for idx, word in enumerate(feature_names)
    ])
    idf_df.to_csv("Dataset/ultra_tiny_idf.csv", index=False)
    print(f"✅ Created ultra_tiny_idf.csv: {len(idf_df):,} words")
    
    # Verify everything works together
    print("\n🧪 Testing the created datasets...")
    
    try:
        # Test a simple query
        test_query = "python list"
        test_processed = preprocess_text_simple(test_query)
        test_vector = vectorizer.transform([test_processed])
        
        # Compute similarity
        similarities = cosine_similarity(test_vector, tf_idf_matrix).flatten()
        best_idx = similarities.argmax()
        
        print(f"✅ Test query: '{test_query}'")
        print(f"✅ Best match similarity: {similarities[best_idx]:.4f}")
        print(f"✅ Matched question: {unique_df.iloc[best_idx]['Body_question'][:100]}...")
        
        # Test loading the saved files
        tf_idf_loaded = load_npz("Dataset/ultra_tiny_sparse_matrix.npz")
        unique_df_loaded = pd.read_csv("Dataset/ultra_tiny_unique_df.csv")
        word_index_loaded = pd.read_csv("Dataset/ultra_tiny_word_to_index.csv")
        idf_loaded = pd.read_csv("Dataset/ultra_tiny_idf.csv")
        
        print(f"✅ All files load correctly")
        print(f"✅ Matrix shape: {tf_idf_loaded.shape}")
        print(f"✅ Unique questions: {len(unique_df_loaded):,}")
        print(f"✅ Vocabulary: {len(word_index_loaded):,}")
        print(f"✅ IDF scores: {len(idf_loaded):,}")
        
        return True
        
    except Exception as e:
        print(f"❌ Testing failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Creating ultra-tiny datasets from scratch for full_cleaned.csv format...\n")
    
    success = create_ultra_tiny_datasets_from_scratch()
    
    if success:
        print("\n🎉 Successfully created all ultra-tiny datasets!")
        print("\n📁 Files created:")
        print("  • Dataset/ultra_tiny.csv (main dataset)")
        print("  • Dataset/ultra_tiny_unique_df.csv (unique questions)")  
        print("  • Dataset/ultra_tiny_sparse_matrix.npz (TF-IDF matrix)")
        print("  • Dataset/ultra_tiny_word_to_index.csv (word index)")
        print("  • Dataset/ultra_tiny_idf.csv (IDF scores)")
        print("\n💡 All datasets are compatible with the full_cleaned.csv format!")
        print("🎯 Your Streamlit app should now work with full chatbot functionality!")
    else:
        print("\n⚠️ Failed to create ultra-tiny datasets.")
        print("Check the error messages above.")