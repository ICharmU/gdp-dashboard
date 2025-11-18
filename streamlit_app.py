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

user_query = st.text_input("user_query", on_change=text_update, placeholder="Ask a question!")







