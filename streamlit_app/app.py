import streamlit as st
import requests

st.title("LLM Search Application with Memory")

# Input field for the user query
query = st.text_input("Enter your query:", "")

if st.button("Search"):
    if query:
        try:
            # Step 1: Inform the user we're starting the process
            st.write("Step 1: Searching articles...")
            
            response = requests.post("https://own-llm-search-engine.onrender.com/query", json={"query": query})
            response.raise_for_status()  # Will raise an error if the response status is not 200

            result = response.json()
            answer = result.get("answer", "No answer found.")

            # Display the answer
            st.subheader("Answer:")
            st.write(answer)

        except requests.exceptions.RequestException as e:
            # Display any error encountered
            st.error(f"Error during the request: {e}")

    else:
        st.warning("Please enter a query.")
