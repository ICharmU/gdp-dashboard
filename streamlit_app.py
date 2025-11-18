import streamlit as st
import pandas as pd
import math
from pathlib import Path

# Set the title and favicon that appear in the Browser's tab bar.
st.set_page_config(
    page_title='StackBot WebPage Sandbox',
    page_icon=':earth_americas:', # This is an emoji shortcode. Could be a URL too.
)


def generate_response(query: str):
    response = query[::-1]
    responses = [response] * k
    return responses

def upvote(response_id):
    current_vote = st.session_state.get(f"user_vote_{response_id}", None)
    
    # If user already upvoted, remove the upvote
    if current_vote == "up":
        st.session_state[f"votes_{response_id}"] -= 1
        st.session_state[f"user_vote_{response_id}"] = None
    # If user downvoted before, remove downvote and add upvote
    elif current_vote == "down":
        st.session_state[f"votes_{response_id}"] += 2  # Remove -1 and add +1
        st.session_state[f"user_vote_{response_id}"] = "up"
    # If user hasn't voted, add upvote
    else:
        st.session_state[f"votes_{response_id}"] += 1
        st.session_state[f"user_vote_{response_id}"] = "up"

def downvote(response_id):
    current_vote = st.session_state.get(f"user_vote_{response_id}", None)
    
    # If user already downvoted, remove the downvote
    if current_vote == "down":
        st.session_state[f"votes_{response_id}"] += 1
        st.session_state[f"user_vote_{response_id}"] = None
    # If user upvoted before, remove upvote and add downvote
    elif current_vote == "up":
        st.session_state[f"votes_{response_id}"] -= 2  # Remove +1 and add -1
        st.session_state[f"user_vote_{response_id}"] = "down"
    # If user hasn't voted, add downvote
    else:
        st.session_state[f"votes_{response_id}"] -= 1
        st.session_state[f"user_vote_{response_id}"] = "down"

def text_update():
    for i in range(1, k+1):
        if st.session_state.user_query:
            responses = generate_response(st.session_state.user_query)
            st.session_state[f"response_text_{i}"] = responses[i-1]

if "text" not in st.session_state:
    st.session_state["text"] = ""

k = 10 # number of results
for i in range(1, k+1):
    if f"response_text_{i}" not in st.session_state:
        st.session_state[f"response_text_{i}"] = ""
    if f"votes_{i}" not in st.session_state:
        st.session_state[f"votes_{i}"] = 0
    if f"user_vote_{i}" not in st.session_state:
        st.session_state[f"user_vote_{i}"] = None

user_query = st.text_input("user_query", key="user_query", on_change=text_update, placeholder="Ask a question!")

# Display the responses with voting
for i in range(1, k+1):
    if st.session_state[f"response_text_{i}"]:
        col1, col2 = st.columns([8, 1])
        
        with col1:
            st.text(f"Response {i}: {st.session_state[f'response_text_{i}']}")
        
        with col2:
            user_vote = st.session_state.get(f"user_vote_{i}", None)
            
            # Stack Overflow style voting layout
            st.button("▲", key=f"up_{i}", on_click=upvote, args=(i,), type="primary" if user_vote == "up" else "secondary")
            st.markdown(f"<div style='text-align: center; font-weight: bold; font-size: 18px;'>{st.session_state[f'votes_{i}']}</div>", unsafe_allow_html=True)
            st.button("▼", key=f"down_{i}", on_click=downvote, args=(i,), type="primary" if user_vote == "down" else "secondary")
        
        st.divider()










