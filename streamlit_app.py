import streamlit as st
import pandas as pd
import math
from pathlib import Path

# Set the title and favicon that appear in the Browser's tab bar.
st.set_page_config(
    page_title='GDP dashboard',
    page_icon=':earth_americas:', # This is an emoji shortcode. Could be a URL too.
)


def generate_response(query: str):
    return query[::-1]

def text_update():
    if st.session_state.user_query:
        response = generate_response(st.session_state.user_query)
        st.session_state["response_text"] = response

if "text" not in st.session_state:
    st.session_state["text"] = ""
if "response_text" not in st.session_state:
    st.session_state["response_text"] = ""

user_query = st.text_input("user_query", key="user_query", on_change=text_update, placeholder="Ask a question!")

# Display the response
if st.session_state["response_text"]:
    st.text(f"Response: {st.session_state['response_text']}")







