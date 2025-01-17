import streamlit as st
from langchain.llms import Ollama
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# Set page configuration
st.set_page_config(
    page_title="Book Information Assistant",
    page_icon="📚"
)

# Create prompt template
book_prompt = PromptTemplate(
    input_variables=["book_title"],
    template="""
    Provide detailed information about the book '{book_title}' in the following format:
    Author:
    Genre:
    Publication Year:
    Summary (200 words):
    Sales Information:
    - Approximate lifetime sales
    - Notable sales achievements (bestseller lists, records, etc.)
    Similar Books (3 recommendations):
    Other Books by This Author (3-5 notable works):
    
    Please ensure all information is accurate and well-formatted.
    If exact sales figures are not available, provide best estimates and indicate this.
    """
)

def get_book_information(book_title):
    """Function to get book information using LangChain with Ollama"""
    # Initialize Ollama
    llm = Ollama(
        model="dolphin-phi"
    )
    
    # Create chain
    chain = LLMChain(llm=llm, prompt=book_prompt)
    
    # Get response
    response = chain.run(book_title=book_title)
    return response

def main():
    # App title and description
    st.title("📚 Book Information Assistant")
    st.markdown("""
        Enter a book title to get detailed information including the author, genre,
        sales figures, summary, similar book recommendations, and other works by the same author.
        Using Ollama (dolphin-phi) model for enhanced responses.
    """)

    # Input field for book title
    book_title = st.text_input("Enter Book Title:", placeholder="e.g., 1984")

    if st.button("Get Information"):
        if book_title:
            try:
                with st.spinner("Fetching book information using Ollama..."):
                    # Get book information
                    book_info = get_book_information(book_title)
                    
                    # Display results in an expanded section
                    with st.expander("Book Information", expanded=True):
                        # Split the response into sections
                        sections = book_info.split('\n')
                        
                        current_section = ""
                        for section in sections:
                            if section.strip():
                                if ":" in section and not section.startswith("-"):
                                    title, content = section.split(":", 1)
                                    current_section = title.strip()
                                    st.markdown(f"**{current_section}**:{content}")
                                elif section.startswith("-"):
                                    # Handle bullet points in Sales Information
                                    st.markdown(section)
                                else:
                                    st.markdown(section)
                
                # Add refresh button
                if st.button("Search Another Book"):
                    st.experimental_rerun()
                    
            except Exception as e:
                st.error(f"""An error occurred: {str(e)}
                
                 The dolphin-phi model is pulled (run: 'ollama pull dolphin-phi')""")
        else:
            st.warning("Please enter a book title.")

    # Footer
    st.markdown("---")
    st.markdown("*Powered by LangChain and Ollama*")

if __name__ == "__main__":
    main()