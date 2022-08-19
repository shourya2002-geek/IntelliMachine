# importing basic libraries
import streamlit as st

# defining a class multipage to act as a framework for the app
class MultiPage: 

    def __init__(self) -> None:
        # constructor to generate a list which will store all our applications as an instance variable.
        self.pages = []
    
    def add_page(self, title, func) -> None: 
        """
        Class Method to Add pages to the project
        Args:
            title ([str]): The title of page which we are adding to the list of apps 
            
            func: Python function to render this page in Streamlit
        """

        self.pages.append(
            {
                "title": title, 
                "function": func
            }
        )

    def run(self):

        # dropdown to select the page to run  
        page = st.sidebar.selectbox('App Navigation', self.pages, format_func=lambda page: page['title'])

        # run the app function 
        page['function']()