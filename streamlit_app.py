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

user_query = st.text_input("user_query", key="user_query", on_change=text_update, placeholder="Ask a question!")

# Display the response
for i in range(1, k+1):
    if st.session_state[f"response_text_{i}"]:
        st.text(f"Response {i}: {st.session_state[f'response_text_{i}']}")










