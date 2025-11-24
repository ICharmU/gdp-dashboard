import pandas as pd
import pickle
import numpy as np
from pathlib import Path

def optimize_dataset_memory(csv_path, output_dir="Dataset/optimized/"):
    """
    Optimize dataset memory usage using multiple strategies
    """
    Path(output_dir).mkdir(exist_ok=True)
    #use chunks to prevent memory spikes
    print("Loading original dataset in chunks...")
    chunksize = 100_000
    chunks = []
    #df = pd.read_csv(csv_path)
    #original_memory = df.memory_usage(deep=True).sum() / 1024**2
    #print(f"Original memory usage: {original_memory:.2f} MB")

    for chunk in pd.read_csv(csv_path, chunksize = chunksize):
        for col in ['Id', 'Score_question', 'Score_answer']:
            if col in chunk.columns:
                chunk[col] = pd.to_numeric(chunk[col], downcast = 'integer')

        for col in ['Tag', 'Title']:
            if col in chunk.columns:
                chunk[col] = chunk[col.astype('category')]
        chunks.append(chunk)
    
    df = pd.concat(chunks, ignore_index= True)
    original_memory = df.memory_usage(deep = True).sum()/ 1024**2
    print(f"Original memory usage: {original_memory:.2f} MB")
    print("Optimizing data types...")
    print("Converting repeated strings to categorical...")
    
    # Strategy 1: Optimize data types
    # print("Optimizing data types...")
    
    # Convert integer columns to smaller types
    # if 'Id' in df.columns:
    #     df['Id'] = pd.to_numeric(df['Id'], downcast='integer')
    # if 'Score_question' in df.columns:
    #     df['Score_question'] = pd.to_numeric(df['Score_question'], downcast='integer') 
    # if 'Score_answer' in df.columns:
    #     df['Score_answer'] = pd.to_numeric(df['Score_answer'], downcast='integer')
    
    # Strategy 2: Convert text columns to categorical where beneficial
    # print("Converting repeated strings to categorical...")
    
    # if 'Tag' in df.columns:
    #     df['Tag'] = df['Tag'].astype('category')
    
    # For titles, check if there are duplicates worth categorizing
    if 'Title' in df.columns:
        title_counts = df['Title'].value_counts()
        if (title_counts > 1).sum() > len(title_counts) * 0.1:  # If >10% are duplicates
            df['Title'] = df['Title'].astype('category')
    
   
    print("Creating question/answer ID mappings...")
    df['question_id'], unique_questions = pd.factorize(df['Body_question'])
    df['anser_id'], unique_answers = pd.factorize(df['Body_answer'])

    
    # Create unique question and answer mappings
    # unique_questions = df['Body_question'].drop_duplicates().reset_index(drop=True)
    # unique_answers = df['Body_answer'].drop_duplicates().reset_index(drop=True)
    
    # Create mapping dictionaries
    # question_to_id = {text: idx for idx, text in enumerate(unique_questions)}
    # answer_to_id = {text: idx for idx, text in enumerate(unique_answers)}
    
    # Replace text with IDs
    # df['question_id'] = df['Body_question'].map(question_to_id)
    # df['answer_id'] = df['Body_answer'].map(answer_to_id)
    
    # Drop original text columns
    df_optimized = df.drop(['Body_question', 'Body_answer'], axis=1)
    
    # Calculate memory savings
    optimized_memory = df_optimized.memory_usage(deep=True).sum() / 1024**2
    question_map_size = len(pickle.dumps(question_to_id)) / 1024**2
    answer_map_size = len(pickle.dumps(answer_to_id)) / 1024**2
    unique_questions_size = unique_questions.memory_usage(deep=True).sum() / 1024**2
    unique_answers_size = unique_answers.memory_usage(deep=True).sum() / 1024**2
    
    total_optimized = optimized_memory + question_map_size + answer_map_size + unique_questions_size + unique_answers_size
    
    print(f"Optimized main dataframe: {optimized_memory:.2f} MB")
    print(f"Question mappings: {question_map_size:.2f} MB")
    print(f"Answer mappings: {answer_map_size:.2f} MB") 
    print(f"Unique questions: {unique_questions_size:.2f} MB")
    print(f"Unique answers: {unique_answers_size:.2f} MB")
    print(f"Total optimized: {total_optimized:.2f} MB")
    print(f"Memory reduction: {((original_memory - total_optimized) / original_memory * 100):.1f}%")
    
    # Save optimized data
    print("Saving optimized files...")
    df_optimized.to_csv(f"{output_dir}/optimized_main.csv", index=False)
    unique_questions.to_csv(f"{output_dir}/unique_questions.csv", index=False, header=['text'])
    unique_answers.to_csv(f"{output_dir}/unique_answers.csv", index=False, header=['text'])
    
    # Save mappings as pickle for faster loading
    with open(f"{output_dir}/question_mapping.pkl", 'wb') as f:
        pickle.dump(question_to_id, f)
    with open(f"{output_dir}/answer_mapping.pkl", 'wb') as f:
        pickle.dump(answer_to_id, f)
    
    return df_optimized, unique_questions, unique_answers, question_to_id, answer_to_id

def create_streamlit_optimized_loader():
    """
    Create an optimized loader for Streamlit
    """
    content = '''
import pandas as pd
import pickle
import streamlit as st
from pathlib import Path

@st.cache_data
def load_optimized_dataset():
    """Load optimized dataset with caching"""
    base_path = Path("Dataset/optimized/")
    
    # Load main dataframe
    df_main = pd.read_csv(base_path / "optimized_main.csv")
    
    # Load text mappings
    with open(base_path / "question_mapping.pkl", 'rb') as f:
        question_mapping = pickle.load(f)
    with open(base_path / "answer_mapping.pkl", 'rb') as f:
        answer_mapping = pickle.load(f)
    
    # Load unique texts (only when needed)
    unique_questions = pd.read_csv(base_path / "unique_questions.csv")['text']
    unique_answers = pd.read_csv(base_path / "unique_answers.csv")['text']
    
    return df_main, unique_questions, unique_answers, question_mapping, answer_mapping

def get_question_text(question_id, unique_questions):
    """Get question text by ID"""
    return unique_questions.iloc[question_id]

def get_answer_text(answer_id, unique_answers):
    """Get answer text by ID"""
    return unique_answers.iloc[answer_id]
'''
    
    with open("optimized_loader.py", 'w') as f:
        f.write(content)
    
    print("Created optimized_loader.py")

if __name__ == "__main__":
    # Optimize the dataset
    optimize_dataset_memory("Dataset/half_cleaned.csv")
    create_streamlit_optimized_loader()
    
    print("\nOptimization complete!")
    print("Next steps:")
    print("1. Update your streamlit_app.py to use the optimized loader")
    print("2. Test the memory usage with the optimized dataset")