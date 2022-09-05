import streamlit as st

from pages.data_visualistaion_app import data_visualisation

def main_page():
    st.markdown("# Main page 🎈")
    st.sidebar.markdown("# Main page 🎈")

page_names_to_funcs = {
    "Main Page": main_page,
    "Visualisation": data_visualisation,
}

selected_page = st.sidebar.selectbox("Select a page", page_names_to_funcs.keys())
page_names_to_funcs[selected_page]()
