import streamlit as st
import pandas as pd
from pathlib import Path

# Set the title and favicon that appear in the Browser's tab bar.
st.set_page_config(
    page_title='StackBot WebPage Sandbox',
    page_icon=':earth_americas:', # This is an emoji shortcode. Could be a URL too.
)

# Memory-optimized data loading
@st.cache_data
def load_essential_data():
    """Load only the most essential data to minimize memory usage"""
    # Load only essential columns
    essential_columns = ['Id', 'Title', 'Score_question', 'Score_answer']
    
    try:
        # Try different datasets in order of preference (smallest first)
        dataset_options = [
            ("Dataset/ultra_tiny.csv", f"✅ Using ultra-tiny dataset (1% sample, ~4MB)"),
            ("Dataset/tiny_optimized.csv", f"✅ Using tiny optimized dataset (10% sample)"),
            ("Dataset/tiny_cleaned.csv", f"✅ Using tiny dataset (10% sample, ~37MB)"), 
            ("Dataset/optimized/optimized_main.csv", f"✅ Using optimized dataset"),
            ("Dataset/half_cleaned.csv", f"⚠️ Using half dataset - high memory usage")
        ]
        
        for dataset_path, success_message in dataset_options:
            if Path(dataset_path).exists():
                if dataset_path == "Dataset/half_cleaned.csv":
                    # Only load essential columns from the large dataset
                    df = pd.read_csv(dataset_path, usecols=essential_columns + ['Body_question', 'Body_answer'])
                else:
                    df = pd.read_csv(dataset_path)
                
                st.success(success_message)
                
                # Quick data type optimization for any dataset
                if 'Id' in df.columns:
                    df['Id'] = pd.to_numeric(df['Id'], downcast='integer')
                if 'Score_question' in df.columns:
                    df['Score_question'] = pd.to_numeric(df['Score_question'], downcast='integer')
                if 'Score_answer' in df.columns:
                    df['Score_answer'] = pd.to_numeric(df['Score_answer'], downcast='integer')
                    
                return df
        
        # If no datasets found
        st.error("❌ No dataset files found")
        st.info("💡 Run `python create_tiny_dataset.py` to create a memory-optimized dataset")
        return pd.DataFrame()
        
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return pd.DataFrame()

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

# Load essential data
df = load_essential_data()

def generate_response(query: str):
    """Generate response - searches dataset and provides meaningful output"""
    if not query or query.strip() == "":
        return "Please enter a question to get started!"
    
    query = query.strip()
    
    if df is not None and len(df) > 0:
        try:
            # Search in titles first (most relevant)
            title_matches = df[df['Title'].str.contains(query, case=False, na=False)]
            
            # Also search in question bodies if available
            if 'Body_question' in df.columns:
                body_matches = df[df['Body_question'].str.contains(query, case=False, na=False)]
                # Combine matches, prioritizing title matches
                all_matches = pd.concat([title_matches, body_matches]).drop_duplicates(subset=['Id'])
            else:
                all_matches = title_matches
            
            if len(all_matches) > 0:
                # Get best match based on answer score
                best_match = all_matches.loc[all_matches['Score_answer'].idxmax()]
                
                # Load the actual answer text
                if 'answer_id' in best_match:
                    # Using optimized data
                    answer_text = load_text_by_id(answer_id=best_match['answer_id'])
                else:
                    # Using original data
                    answer_text = best_match.get('Body_answer', 'No detailed answer available')
                
                if answer_text and answer_text.strip():
                    # Format the response nicely
                    title = best_match.get('Title', 'Related Question')
                    score = best_match.get('Score_answer', 0)
                    
                    if len(answer_text) > 300:
                        answer_text = answer_text[:300] + "..."
                    
                    return f"**Question:** {title}\n\n**Answer (Score: {score}):** {answer_text}"
                else:
                    return f"Found a related question '{best_match.get('Title', 'Unknown')}' but no detailed answer is available."
            
            else:
                # Try partial/fuzzy matching
                query_words = query.lower().split()
                if len(query_words) > 1:
                    # Search for any of the words
                    partial_pattern = '|'.join(query_words)
                    partial_matches = df[df['Title'].str.contains(partial_pattern, case=False, na=False)]
                    
                    if len(partial_matches) > 0:
                        best_match = partial_matches.loc[partial_matches['Score_answer'].idxmax()]
                        title = best_match.get('Title', 'Related Question')
                        return f"No exact matches found, but here's a related question: **{title}**\n\nTry searching for more specific terms."
                
                return f"No matches found for '{query}'. Try searching for:\n• Programming languages (python, javascript, etc.)\n• Specific topics (arrays, functions, etc.)\n• Error messages or concepts"
                
        except Exception as e:
            return f"Search error occurred: {str(e)}. Falling back to simple response: {query[::-1]}"
    
    else:
        return f"Dataset not available. Echo response: {query[::-1]}"

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
    with st.sidebar.expander("📁 Available Datasets"):
        st.write("**Current priority order:**")
        st.write("1. ultra_tiny.csv (1%, ~4MB)")
        st.write("2. tiny_cleaned.csv (10%, ~37MB)")
        st.write("3. half_cleaned.csv (50%, ~371MB)")
        st.write("*App uses smallest available*")
        
else:
    st.sidebar.error("❌ Dataset not available")
    st.sidebar.write("The system will use fallback responses.")
    
# Memory optimization tools
with st.sidebar.expander("🔧 Memory Tools"):
    st.write("**Create smaller datasets:**")
    if st.button(f"Create Tiny Dataset (10%)", help=f"Creates 10% sample"):
        st.info("Run: `python create_tiny_dataset_simple.py`")
    st.write("**Current datasets:**")
    dataset_files = [
        ("Dataset/ultra_tiny.csv", "1%"),
        ("Dataset/tiny_cleaned.csv", "10%"), 
        ("Dataset/half_cleaned.csv", "50%")
    ]
    for file_path, size in dataset_files:
        if Path(file_path).exists():
            file_size = Path(file_path).stat().st_size / 1024**2
            st.write(f"✅ {size} sample: {file_size:.1f}MB")
        else:
            st.write(f"❌ {size} sample: Not created")

# Main interface
st.title("StackBot - Memory Optimized")

# Add some example queries to help users
st.markdown("**Try asking about:** *python arrays*, *javascript functions*, *error handling*, *data structures*")

user_query = st.text_input("Ask a question", key="user_query", on_change=text_update, placeholder="e.g., 'python list comprehension' or 'javascript async await'")

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

# Memory optimization tips
with st.expander("💡 Memory Optimization Tips"):
    st.markdown("""
    **Current optimizations:**
    - ✅ Loading only essential columns
    - ✅ Using Streamlit caching
    - ✅ Loading text content on demand
    
    **To further optimize:**
    1. Run `python memory_optimizer.py` to create optimized dataset
    2. This can reduce memory usage by 40-70%
    3. Use TF-IDF optimization for ML features
    """)