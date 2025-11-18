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
    return str[::-1]

def text_update():
    text = generate_response(user_query)
    st.text(text)

def replace_text():
    st.session_state["text"] = text1.replace(before, after)

if "text" not in st.session_state:
    st.session_state["text"] = ""
text1 = st.text_area('Text : ', st.session_state["text"])
before = st.text_input('Before')
after = st.text_input('After')
button = st.button('Button', on_click=replace_text) 

user_query = st.text_input("user_query", on_change=text_update, placeholder="Ask a question!")







