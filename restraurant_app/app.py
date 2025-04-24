import streamlit as st
import helper

st.title('🍽️ Restaurant Name & Menu Generator')

cuisine = st.sidebar.selectbox(
    "Pick a Cuisine",
    (
        "Indian", "Mexican", "Italian", "Arabian", "American",
        "Thai", "Japanese", "Korean", "Chinese", "French",
        "Greek", "Spanish", "Turkish", "Ethiopian", "Vietnamese"
    )
)

if cuisine:
    response = helper.generate_res_name_and_items(cuisine)

    st.header(response['restaurant_name'])
    menu_items = [item.strip() for item in response['menu_items'].split(',')]

    st.subheader("⭐ Menu Items")
    for item in menu_items:
        st.write("•", item)
