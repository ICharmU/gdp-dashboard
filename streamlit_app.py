import streamlit as st
import pandas as pd
from pathlib import Path
import numpy as np
import os
from helper_functions import chatbot_reply as helper_chatbot_reply

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
        return True
    except Exception as e:
        st.warning(f"NLTK setup failed: {e}")
        return False

# Initialize NLTK data
setup_nltk_data()
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
import re
import tqdm
from collections import defaultdict
from collections import Counter
# Try to import NLTK with fallback handling
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    from nltk.stem import WordNetLemmatizer
    
    # Download required NLTK data if not already present
    try:
        # Check if stopwords are available
        stopwords.words('english')
    except LookupError:
        # Download stopwords if not available
        nltk.download('stopwords', quiet=True)
        
    try:
        # Check if punkt tokenizer is available
        word_tokenize("test")
    except LookupError:
        # Download punkt if not available
        nltk.download('punkt', quiet=True)
        
    try:
        # Check if wordnet is available
        lemmatizer = WordNetLemmatizer()
        lemmatizer.lemmatize("test")
    except LookupError:
        # Download wordnet if not available
        nltk.download('wordnet', quiet=True)
        
    NLTK_AVAILABLE = True
    
except ImportError:
    # NLTK not available, define fallback functions
    NLTK_AVAILABLE = False
    
    def word_tokenize(text):
        """Fallback tokenizer using regex"""
        import re
        return re.findall(r'\b\w+\b', text.lower())
    
    class WordNetLemmatizer:
        """Fallback lemmatizer that does nothing"""
        def lemmatize(self, word):
            return word.lower()
    
    # Define basic stopwords set as fallback
    basic_stopwords = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 
        'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 
        'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 
        'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those', 'i', 
        'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'
    }
from scipy.sparse import save_npz, load_npz

contraction_map = {
    # Negative contractions
    "ain't": "am not",
    "aren't": "are not",
    "can't": "cannot",
    "couldn't": "could not",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hasn't": "has not",
    "haven't": "have not",
    "isn't": "is not",
    "mightn't": "might not",
    "mustn't": "must not",
    "needn't": "need not",
    "shan't": "shall not",
    "shouldn't": "should not",
    "wasn't": "was not",
    "weren't": "were not",
    "won't": "will not",
    "wouldn't": "would not",
    
    # Pronoun contractions
    "i'm": "i am",
    "you're": "you are",
    "he's": "he is",
    "she's": "she is",
    "it's": "it is",
    "we're": "we are",
    "they're": "they are",
    "i've": "i have",
    "you've": "you have",
    "we've": "we have",
    "they've": "they have",
    "i'd": "i would",
    "you'd": "you would",
    "he'd": "he would",
    "she'd": "she would",
    "we'd": "we would",
    "they'd": "they would",
    "i'll": "i will",
    "you'll": "you will",
    "he'll": "he will",
    "she'll": "she will",
    "we'll": "we will",
    "they'll": "they will",
    
    # Misc contractions
    "let's": "let us",
    "who's": "who is",
    "what's": "what is",
    "here's": "here is",
    "there's": "there is",
    "when's": "when is",
    "where's": "where is",
    "why's": "why is",
    "how's": "how is",
    "y'all": "you all",
    "o'clock": "of the clock",
    
    # Informal / common text contractions
    "ma'am": "madam",
    "gonna": "going to",
    "wanna": "want to",
    "gotta": "got to",
    "lemme": "let me",
    "gimme": "give me",
    "kinda": "kind of",
    "ain’t": "am not",
    "y’all": "you all",
    "could’ve": "could have",
    "should’ve": "should have",
    "would’ve": "would have",
    "might’ve": "might have",
    "must’ve": "must have",
    "shan’t": "shall not",
    "let’s": "let us"
}
def expand_contractions(text):
    for contraction, expanded in contraction_map.items():
        text = text.replace(contraction, expanded)
    return text

# Load all datasets at module level
@st.cache_data
def _load_datasets():
    """Load all required datasets for chatbot functionality"""
    from scipy.sparse import load_npz
    
    # Load main dataset
    dataset_path = "Dataset/ultra_tiny.csv"
    df = pd.read_csv(dataset_path, usecols=['Id', 'Score_question', 'Body_question', 'Score_answer', 'Body_answer'])
    df = df.dropna()
    
    # Load TF-IDF matrix
    tf_idf_matrix = load_npz("Dataset/ultra_tiny_sparse_matrix.npz")
    
    # Load word index mapping
    word_index_df = pd.read_csv('Dataset/ultra_tiny_word_to_index.csv', keep_default_na=False)
    unique_words = dict(zip(word_index_df['word'], word_index_df['index']))
    
    # Load IDF scores
    idf_df = pd.read_csv('Dataset/ultra_tiny_idf.csv', keep_default_na=False)
    idf = dict(zip(idf_df['word'], idf_df['idf_score']))
    
    # Load unique questions dataframe
    unique_df = pd.read_csv("Dataset/ultra_tiny_unique_df.csv")
    
    # Optimize data types
    df['Id'] = pd.to_numeric(df['Id'], downcast='integer')
    df['Score_question'] = pd.to_numeric(df['Score_question'], downcast='integer') 
    df['Score_answer'] = pd.to_numeric(df['Score_answer'], downcast='integer')
    
    return df, tf_idf_matrix, unique_words, idf, unique_df

# Load datasets immediately
df, tf_idf_matrix, unique_words, idf, unique_df = _load_datasets()

# Initialize NLTK components or fallbacks
if NLTK_AVAILABLE:
    try:
        stop_words = set(stopwords.words('english')) - {"not", "no", "never"}
        lemmatizer = WordNetLemmatizer()
    except Exception as e:
        # Fallback if NLTK data is still not available
        stop_words = basic_stopwords - {"not", "no", "never"}
        lemmatizer = WordNetLemmatizer()
        st.warning(f"⚠️ NLTK data not fully available, using basic stopwords: {e}")
else:
    stop_words = basic_stopwords - {"not", "no", "never"}
    lemmatizer = WordNetLemmatizer()
    st.info("ℹ️ Using basic text processing (NLTK not available)")
def preprocess_text(text):
    # Expand contractions
    text = expand_contractions(text)
    
    # Tokenization
    tokens = word_tokenize(text)

    # Lowercase and keep only alphabetic words
    tokens = [word for word in tokens if word.isalpha()]

    # Remove stopwords
    tokens = [w for w in tokens if w not in stop_words]

    # Lemmatize
    tokens = [lemmatizer.lemmatize(w) for w in tokens]
    return " ".join(tokens)

def chatbot_reply(user_query):
    """Generate chatbot reply using helper_functions.py"""
    # Use global variables loaded at module level
    # global tf_idf_matrix, unique_words, idf, unique_df, df
    
    # Call the helper function with all required parameters
    return helper_chatbot_reply(user_query, unique_words, idf, tf_idf_matrix, unique_df, df)

# Set the title and favicon that appear in the Browser's tab bar.
st.set_page_config(
    page_title='StackBot',
    page_icon=':earth_americas:', # This is an emoji shortcode. Could be a URL too.
)



@st.cache_data 
def load_text_by_id(question_id=None, answer_id=None):
    """Load specific text content only when needed"""
    try:
        if Path("Dataset/optimized/unique_questions.csv").exists():
            if question_id is not None:
                questions = pd.read_csv("Dataset/optimized/unique_questions.csv")
                return questions.iloc[question_id]['text'] if question_id < len(questions) else ""
            if answer_id is not None:
                answers = pd.read_csv("Dataset/optimized/unique_answers.csv") 
                return answers.iloc[answer_id]['text'] if answer_id < len(answers) else ""
        else:
            # Fallback to loading from main dataset
            df = pd.read_csv("Dataset/half_cleaned.csv")
            if question_id is not None:
                return df.iloc[question_id]['Body_question'] if question_id < len(df) else ""
            if answer_id is not None:
                return df.iloc[answer_id]['Body_answer'] if answer_id < len(df) else ""
    except Exception as e:
        st.error(f"Error loading text: {e}")
        return ""

# Display loading status
st.success(f"✅ Loaded main dataset ({len(df):,} rows)")
st.success("✅ Loaded ultra-tiny TF-IDF matrix")
st.success("✅ Loaded ultra-tiny word index mapping")
st.success("✅ Loaded ultra-tiny IDF scores")
st.success("✅ Loaded ultra-tiny unique questions")
st.success("✅ All datasets loaded successfully - full chatbot functionality available")

def generate_response(query: str):
    """Generate response using chatbot functionality"""
    if not query or query.strip() == "":
        return "Please enter a question to get started!"
    
    answer, question, score = chatbot_reply(query)
    
    # Format response with similarity score
    if score > 0.1:  # Good similarity score
        return f"**Similarity Score: {score:.4f}**\n\n**Answer:** {answer}\n\n---\n\n**Related Question:** {question}"
    else:  # Low similarity, show with warning
        return f"**⚠️ Low confidence match (Score: {score:.4f})**\n\n**Answer:** {answer}\n\n---\n\n**Related Question:** {question}\n\n*Try rephrasing your question for better results.*"
    

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
    
# Memory optimization tools
with st.sidebar.expander("🔧 Repository Status"):
    st.write("**Available datasets in repo:**")
    dataset_files = [
        ("Dataset/ultra_tiny.csv", "1%", "🟢 Active"),
        ("Dataset/tiny_cleaned.csv", "10%", "⚪ Available"), 
        ("Dataset/half_cleaned.csv", "50%", "⚪ Available")
    ]
    for file_path, size, status in dataset_files:
        if Path(file_path).exists():
            file_size = Path(file_path).stat().st_size / 1024**2
            st.write(f"{status} {size}: {file_size:.1f}MB")
        else:
            st.write(f"❌ {size} sample: Not found")
    
    st.info("💡 App uses ultra_tiny.csv only for maximum memory efficiency")
    
    if st.button("Recreate Ultra-Tiny Dataset", help="Regenerate the 1% sample"):
        st.info("Run: `python create_tiny_dataset_simple.py`")

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
    - ✅ Proper semantic similarity ("list" and "what is a list" work similarly)
    
    **Helper function status:**
    ✅ helper_functions.py imported and working
    """)