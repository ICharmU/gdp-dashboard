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
    st.session_state[f"votes_{response_id}"] += 1

def downvote(response_id):
    st.session_state[f"votes_{response_id}"] -= 1

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

user_query = st.text_input("user_query", key="user_query", on_change=text_update, placeholder="Ask a question!")

# Display the responses with voting
for i in range(1, k+1):
    if st.session_state[f"response_text_{i}"]:
        col1, col2, col3, col4 = st.columns([6, 1, 1, 1])
        
        with col1:
            st.text(f"Response {i}: {st.session_state[f'response_text_{i}']}")
        
        with col2:
            st.button("👍", key=f"up_{i}", on_click=upvote, args=(i,))
        
        with col3:
            st.button("👎", key=f"down_{i}", on_click=downvote, args=(i,))
        
        with col4:
            st.text(f"{st.session_state[f'votes_{i}']}")
        
        st.divider()










