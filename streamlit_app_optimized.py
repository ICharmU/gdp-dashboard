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
        # Try to load optimized data first
        if Path("Dataset/optimized/optimized_main.csv").exists():
            df = pd.read_csv("Dataset/optimized/optimized_main.csv")
            st.success("✅ Using optimized dataset")
        else:
            # Fallback to original with column selection
            df = pd.read_csv("Dataset/half_cleaned.csv", usecols=essential_columns + ['Body_question', 'Body_answer'])
            st.warning("⚠️ Using original dataset - consider running memory optimization")
            
        return df
        
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
    """Generate response - currently using simple string reversal"""
    if df is not None and len(df) > 0:
        # Simple search in titles for now (memory efficient)
        matching_rows = df[df['Title'].str.contains(query, case=False, na=False)]
        if len(matching_rows) > 0:
            best_match = matching_rows.loc[matching_rows['Score_answer'].idxmax()]
            
            # Load the actual answer text
            if 'answer_id' in best_match:
                # Using optimized data
                answer_text = load_text_by_id(answer_id=best_match['answer_id'])
            else:
                # Using original data
                answer_text = best_match.get('Body_answer', 'No answer available')
                
            return f"Found match: {answer_text[:200]}..." if len(answer_text) > 200 else answer_text
        else:
            return f"No matches found for '{query}'. Try a different search term."
    else:
        return query[::-1]  # Fallback to string reversal

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

# Display memory usage info
if df is not None:
    memory_usage = df.memory_usage(deep=True).sum() / 1024**2
    st.sidebar.metric("Dataset Memory Usage", f"{memory_usage:.1f} MB")
    st.sidebar.metric("Dataset Size", f"{len(df):,} rows")

# Main interface
st.title("StackBot - Memory Optimized")

user_query = st.text_input("Ask a question", key="user_query", on_change=text_update, placeholder="Ask a programming question!")

# Display bad query counter
if st.session_state["bad_query_count"] > 0:
    st.error(f"Bad queries: {st.session_state['bad_query_count']}")

# Display the response with feedback buttons
if st.session_state["response_text"]:
    col1, col2, col3 = st.columns([6, 1, 1])
    
    with col1:
        st.text(f"Response: {st.session_state['response_text']}")
    
    # Only show feedback buttons if user hasn't given feedback yet
    if not st.session_state["feedback_given"]:
        with col2:
            st.button("👍", key="thumbs_up", on_click=thumbs_up, help="Good response")
        
        with col3:
            st.button("👎", key="thumbs_down", on_click=thumbs_down, help="Bad response - regenerate")
    else:
        if st.session_state["user_satisfied"]:
            with col2:
                st.success("👍 Thanks!")
        else:
            with col2:
                st.info("Response regenerated")

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