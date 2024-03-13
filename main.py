import langchain_helper as lch
import streamlit as st
import textwrap

st.title('YouTube Assistant')

with st.sidebar:
    with st.form(key='my_form'):
        youtube_url = st.sidebar.text_area(
            label='Enter the Youtube URL:',
            max_chars=50,
        )
        query = st.sidebar.text_area(
            label='Ask about the video:',
            max_chars=50,
            key='query',
        )

        submit = st.form_submit_button(
            label='Submit'
        )

if query and youtube_url:
    db = lch.create_yt_vector_db(youtube_url)
    response, docs = lch.get_query_response(db, query)
    st.subheader('Answer:')
    st.text(
        textwrap.fill(
            response,
            width=80,
        )
    )