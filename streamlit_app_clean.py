import streamlit as st
import pandas as pd
from pathlib import Path
import numpy as np
import os

# Import the helper function
try:
    from helper_functions import chatbot_reply as helper_chatbot_reply
    HELPER_AVAILABLE = True
except ImportError as e:
    st.error(f"Failed to import helper_functions: {e}")
    HELPER_AVAILABLE = False

# Setup NLTK data on first run
@st.cache_resource
def setup_nltk_data():
    """Download NLTK data once per session"""
    try:
        import nltk
        import ssl
        
        # Handle SSL certificate issues
        try:
            _create_unverified_https_context = ssl._create_unverified_context
        except AttributeError:
            pass
        else:
            ssl._create_default_https_context = _create_unverified_https_context
        
        # Download required data
        nltk.download('stopwords', quiet=True)
        nltk.download('punkt', quiet=True) 
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)
        return True
    except Exception as e:
        st.warning(f"NLTK setup failed: {e}")
        return False

from scipy.sparse import load_npz

# Global variables for chatbot functionality (will be loaded by load_essential_data)
tf_idf_matrix = None
unique_words = None
idf = None
unique_df = None

def chatbot_reply(user_query):
    """Generate chatbot reply using helper_functions.py"""
    global tf_idf_matrix, unique_words, idf, unique_df, df
    
    # Check if required data is available
    if any(x is None for x in [tf_idf_matrix, unique_words, idf, unique_df, df]):
        return "Chatbot functionality not available - missing required datasets", "System Error", 0.0
    
    if not HELPER_AVAILABLE:
        return "Helper functions not available", "System Error", 0.0
    
    try:
        # Call the helper function with all required parameters
        return helper_chatbot_reply(user_query, unique_words, idf, tf_idf_matrix, unique_df, df)
        
    except Exception as e:
        return f"Error in chatbot_reply: {str(e)}", "System Error", 0.0

# Set the title and favicon that appear in the Browser's tab bar.
st.set_page_config(
    page_title='StackBot',
    page_icon=':earth_americas:', # This is an emoji shortcode. Could be a URL too.
)

@st.cache_data
def load_essential_data():
    """Load all required datasets for chatbot functionality"""
    global tf_idf_matrix, unique_words, idf, unique_df
    
    try:
        # Try to find main dataset
        dataset_path = None
        if Path("Dataset/ultra_tiny.csv").exists():
            dataset_path = "Dataset/ultra_tiny.csv"
        elif Path("data/ultra_tiny.csv").exists():
            dataset_path = "data/ultra_tiny.csv"
        
        if dataset_path is None:
            st.error("❌ Ultra-tiny dataset not found in Dataset/ or data/ folders")
            st.info("💡 Run `python create_tiny_dataset_simple.py` to create the dataset")
            return pd.DataFrame()
        
        # Load main dataset with standard Body_question column
        df = pd.read_csv(dataset_path, usecols=['Id', 'Score_question', 'Body_question', 'Score_answer', 'Body_answer'])
        df = df.dropna()
        st.success(f"✅ Loaded main dataset from {dataset_path} ({len(df):,} rows)")
        
        # Try to load ultra-tiny supporting files for full chatbot functionality
        missing_files = []
        
        # Load ultra-tiny TF-IDF matrix
        if Path("Dataset/ultra_tiny_sparse_matrix.npz").exists():
            tf_idf_matrix = load_npz("Dataset/ultra_tiny_sparse_matrix.npz")
            st.success("✅ Loaded ultra-tiny TF-IDF matrix")
        else:
            missing_files.append("ultra_tiny_sparse_matrix.npz")
        
        # Load ultra-tiny word index mapping
        if Path("Dataset/ultra_tiny_word_to_index.csv").exists():
            word_index_df = pd.read_csv('Dataset/ultra_tiny_word_to_index.csv', keep_default_na=False)
            unique_words = dict(zip(word_index_df['word'], word_index_df['index']))
            st.success("✅ Loaded ultra-tiny word index mapping")
        else:
            missing_files.append("ultra_tiny_word_to_index.csv")
        
        # Load ultra-tiny IDF scores
        if Path("Dataset/ultra_tiny_idf.csv").exists():
            idf_df = pd.read_csv('Dataset/ultra_tiny_idf.csv', keep_default_na=False)
            idf = dict(zip(idf_df['word'], idf_df['idf_score']))
            st.success("✅ Loaded ultra-tiny IDF scores")
        else:
            missing_files.append("ultra_tiny_idf.csv")
        
        # Load ultra-tiny unique questions dataframe
        if Path("Dataset/ultra_tiny_unique_df.csv").exists():
            unique_df = pd.read_csv("Dataset/ultra_tiny_unique_df.csv")
            st.success("✅ Loaded ultra-tiny unique questions")
        else:
            # Fallback: create from main dataset
            unique_df = df[['Body_question', "Id"]].drop_duplicates()
            missing_files.append("ultra_tiny_unique_df.csv")
        
        if missing_files:
            st.warning(f"⚠️ Missing files for full chatbot functionality: {', '.join(missing_files)}")
            st.info("💡 Chatbot will use basic text matching instead of TF-IDF similarity")
        else:
            st.success("✅ All datasets loaded successfully - full chatbot functionality available")
        
        # Optimize data types
        if 'Id' in df.columns:
            df['Id'] = pd.to_numeric(df['Id'], downcast='integer')
        if 'Score_question' in df.columns:
            df['Score_question'] = pd.to_numeric(df['Score_question'], downcast='integer')
        if 'Score_answer' in df.columns:
            df['Score_answer'] = pd.to_numeric(df['Score_answer'], downcast='integer')
        
        return df
        
    except Exception as e:
        st.error(f"Error loading datasets: {e}")
        return pd.DataFrame()

# Load essential data
df = load_essential_data()

def generate_response(query: str):
    """Generate response using chatbot functionality or fallback methods"""
    if not query or query.strip() == "":
        return "Please enter a question to get started!"
    
    try:
        answer, question, score = chatbot_reply(query)
        
        # Check if we got a valid response
        if "System Error" in question or "No Matches" in question:
            # Fallback to basic search in the dataset
            return fallback_search(query)
        
        # Format successful response
        if score > 0.1:  # Good similarity score
            return f"**Similarity Score: {score:.4f}**\n\n**Answer:** {answer}\n\n---\n\n**Related Question:** {question}"
        else:  # Low similarity, show with warning
            return f"**⚠️ Low confidence match (Score: {score:.4f})**\n\n**Answer:** {answer}\n\n---\n\n**Related Question:** {question}\n\n*Try rephrasing your question for better results.*"
            
    except Exception as e:
        st.warning(f"Chatbot error: {str(e)}")
        return fallback_search(query)

def fallback_search(query: str):
    """Fallback search method when TF-IDF is not available"""
    if df is None or df.empty:
        return f"Dataset not available. Echo response: {query[::-1]}"
    
    try:
        # Simple text matching fallback
        query_lower = query.lower()
        
        # Search in questions first
        question_matches = df[df['Body_question'].str.lower().str.contains(query_lower, na=False)]
        
        if not question_matches.empty:
            best_match = question_matches.sort_values('Score_question', ascending=False).iloc[0]
            return f"**🔍 Basic text search result:**\n\n**Answer:** {best_match['Body_answer']}\n\n---\n\n**Question:** {best_match['Body_question']}"
        
        # Search in answers if no question matches
        answer_matches = df[df['Body_answer'].str.lower().str.contains(query_lower, na=False)]
        
        if not answer_matches.empty:
            best_match = answer_matches.sort_values('Score_answer', ascending=False).iloc[0]
            return f"**🔍 Found in answers:**\n\n**Answer:** {best_match['Body_answer']}\n\n---\n\n**Related Question:** {best_match['Body_question']}"
        
        return f"No matches found for '{query}'. Try different keywords or check your spelling."
        
    except Exception as e:
        return f"Search error: {str(e)}. Please try a simpler query."

def thumbs_up():
    st.session_state["feedback_given"] = True
    st.session_state["user_satisfied"] = True

def thumbs_down():
    # Increment bad query counter
    st.session_state["bad_query_count"] += 1
    # Regenerate response
    if st.session_state.user_query:
        st.session_state["response_text"] = generate_response(st.session_state.user_query)
    st.session_state["feedback_given"] = False
    st.session_state["user_satisfied"] = False

def text_update():
    if st.session_state.user_query:
        st.session_state["response_text"] = generate_response(st.session_state.user_query)
        st.session_state["feedback_given"] = False
        st.session_state["user_satisfied"] = False

# Initialize NLTK in the background (don't block startup)
setup_nltk_data()

# Initialize session state
if "response_text" not in st.session_state:
    st.session_state["response_text"] = ""
if "bad_query_count" not in st.session_state:
    st.session_state["bad_query_count"] = 0
if "feedback_given" not in st.session_state:
    st.session_state["feedback_given"] = False
if "user_satisfied" not in st.session_state:
    st.session_state["user_satisfied"] = False

# Display memory usage info and search statistics
st.sidebar.header("📊 System Status")

if df is not None and len(df) > 0:
    memory_usage = df.memory_usage(deep=True).sum() / 1024**2
    st.sidebar.metric("Dataset Memory Usage", f"{memory_usage:.1f} MB")
    st.sidebar.metric("Dataset Size", f"{len(df):,} rows")
    st.sidebar.metric("Available Columns", f"{len(df.columns)}")
    
    # Memory status indicator
    if memory_usage < 10:
        st.sidebar.success(f"🟢 Excellent memory usage")
    elif memory_usage < 50:
        st.sidebar.info(f"🟡 Good memory usage") 
    else:
        st.sidebar.warning(f"🟠 High memory usage")
    
    # Show column info
    with st.sidebar.expander("Dataset Columns"):
        for col in df.columns:
            st.write(f"• {col}")
            
    st.sidebar.success("✅ Dataset loaded successfully")
    
    # Dataset options
    with st.sidebar.expander("📁 Dataset Configuration"):
        st.write("**Currently using:**")
        st.write("• ultra_tiny.csv (1% sample, ~11MB)")
        st.write("**Status:** Maximum memory optimization")
        st.info("App uses ultra-tiny dataset + helper_functions.py for chatbot responses.")
        
else:
    st.sidebar.error("❌ Dataset not available")
    st.sidebar.write("The system will use fallback responses.")

# Main interface
st.title("StackBot - Helper Functions Integration")

# Add some example queries to help users
st.markdown("**Ask a programming question** (Ex: *python arrays*, *javascript functions*, *error handling*, *data structures*)")

user_query = st.text_input("Please input a question", key="user_query", on_change=text_update, placeholder="e.g., 'python list comprehension' or 'javascript async await'")

# Display bad query counter
if st.session_state["bad_query_count"] > 0:
    st.error(f"Bad queries: {st.session_state['bad_query_count']}")

# Display the response with feedback buttons
if st.session_state["response_text"]:
    # Show response in a nice container
    st.subheader("📝 Response")
    
    # Use markdown for better formatting
    st.markdown(st.session_state["response_text"])
    
    # Feedback section
    st.divider()
    col1, col2, col3, col4 = st.columns([3, 1, 1, 2])
    
    with col1:
        st.caption("Was this response helpful?")
    
    # Only show feedback buttons if user hasn't given feedback yet
    if not st.session_state["feedback_given"]:
        with col2:
            st.button("👍 Helpful", key="thumbs_up", on_click=thumbs_up, help="Good response")
        
        with col3:
            st.button("👎 Not helpful", key="thumbs_down", on_click=thumbs_down, help="Bad response - regenerate")
    else:
        if st.session_state["user_satisfied"]:
            with col2:
                st.success("👍 Thanks for the feedback!")
        else:
            with col2:
                st.info("🔄 Regenerated")
            with col4:
                st.caption("Try rephrasing your question for better results")
else:
    # Show helpful message when no query yet
    if not st.session_state.get("user_query", "").strip():
        st.info("👆 Enter a programming question above to get started!")

# Integration status
with st.expander("🔧 Integration Status"):
    st.markdown("""
    **Current setup:**
    - ✅ Uses `helper_functions.py` for chatbot logic
    - ✅ Ultra-tiny datasets (11MB total)
    - ✅ TF-IDF similarity matching
    - ✅ Error handling and fallbacks
    
    **Helper function status:**
    """)
    if HELPER_AVAILABLE:
        st.success("✅ helper_functions.py imported successfully")
    else:
        st.error("❌ helper_functions.py import failed")