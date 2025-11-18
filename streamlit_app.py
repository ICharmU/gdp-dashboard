import streamlit as st
import pandas as pd
import math
from pathlib import Path
from helper_functions import preprocess_text

# Set the title and favicon that appear in the Browser's tab bar.
st.set_page_config(
    page_title='StackBot WebPage Sandbox',
    page_icon=':earth_americas:', # This is an emoji shortcode. Could be a URL too.
)


def generate_response(query: str):
    return query[::-1]

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

user_query = st.text_input("user_query", key="user_query", on_change=text_update, placeholder="Ask a question!")

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
                processed_text = preprocess_text(user_query)
                st.success(processed_text)
        else:
            with col2:
                st.info("Response regenerated")