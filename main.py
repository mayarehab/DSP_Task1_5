import streamlit as st
import utils as utl
from views import home, generate, options, configuration

st.set_page_config(layout="wide", page_title='Navbar sample')
st.set_option('deprecation.showPyplotGlobalUse', False)
utl.inject_custom_css()
utl.navbar_component()


def navigation():
    route = utl.get_current_route()
    if route == None or route == "home":
        home.load_view()
    elif route == "generate":
        generate.load_view()
   

navigation()
