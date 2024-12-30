import streamlit as st
import pandas as pd
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class GarageAssistant:
    def __init__(self):
        # Initialize Gemini
        genai.configure(api_key='AIzaSyCcMZPrzP5me7Rl4pmAc1Nn5vUDSan5Q6E')
        self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
        self.chat = self.model.start_chat()
        
        # Initialize search
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.garages_df = None
        self.embeddings = None

    def load_data(self, file):
        """Load garage data from CSV"""
        try:
            self.garages_df = pd.read_csv(file)
            # Generate embeddings for search
            texts = self.garages_df.apply(lambda x: f"{x['Garage Name']} {x['Location']}", axis=1)
            self.embeddings = self.embedding_model.encode(texts.tolist())
            return True
        except Exception as e:
            st.error(f"Error loading data: {e}")
            return False

    def search_garages(self, query, num_results=5):
        """Find relevant garages based on query"""
        if self.garages_df is None:
            return None
        
        query_embedding = self.embedding_model.encode([query])
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        top_indices = similarities.argsort()[-num_results:][::-1]
        return self.garages_df.iloc[top_indices]

    def get_response(self, query):
        """Get AI response for user query"""
        try:
            # Find relevant garages
            relevant_garages = self.search_garages(query)
            if relevant_garages is None:
                return "Please upload garage data first.", None

            # Create prompt with garage information
            garage_info = "\n".join([
                f"Name: {row['Garage Name']}\n"
                f"Location: {row['Location']}\n"
                f"Phone: {row['Phone']}\n"
                f"Email: {row.get('Email', 'N/A')}\n"
                f"Website: {row.get('Website', 'N/A')}\n"
                for _, row in relevant_garages.iterrows()
            ])

            prompt = f"""Help find a suitable garage based on this query: "{query}"
            
            Available garages:
            {garage_info}
            
            Provide a helpful response that:
            1. Suggests the most suitable garage(s)
            2. Includes relevant contact details
            3. Explains your recommendations"""

            # Get AI response
            response = self.chat.send_message(prompt)
            return response.text, relevant_garages

        except Exception as e:
            st.error(f"Error: {e}")
            return "Sorry, I encountered an error. Please try again.", None

def main():
    st.title("Garage Finder")

    # Initialize session state
    if 'assistant' not in st.session_state:
        st.session_state.assistant = GarageAssistant()
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Sidebar for data upload
    with st.sidebar:
        uploaded_file = st.file_uploader("Upload Garage Data (CSV)", type='csv')
        if uploaded_file:
            if st.session_state.assistant.load_data(uploaded_file):
                st.success("Data loaded successfully")

    # Chat interface
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            
            # Show garage details if available
            if "garages" in message and message["garages"] is not None:
                for _, garage in message["garages"].iterrows():
                    with st.expander(f"{garage['Garage Name']}"):
                        st.write(f"Location: {garage['Location']}")
                        st.write(f"Phone: {garage['Phone']}")
                        if pd.notna(garage.get('Email')):
                            st.write(f"Email: {garage['Email']}")
                        if pd.notna(garage.get('Website')):
                            st.write(f"Website: {garage['Website']}")

    # User input
    if prompt := st.chat_input("How can I help you find a garage?"):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        # Get and show assistant response
        with st.chat_message("assistant"):
            response, garages = st.session_state.assistant.get_response(prompt)
            st.write(response)
            st.session_state.messages.append({
                "role": "assistant", 
                "content": response,
                "garages": garages
            })

if __name__ == "__main__":
    main()
