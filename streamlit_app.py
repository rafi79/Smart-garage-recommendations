import streamlit as st
import pandas as pd
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class AutoServiceAgent:
    def __init__(self, api_key='AIzaSyCcMZPrzP5me7Rl4pmAc1Nn5vUDSan5Q6E'):
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.garages_df = None
        self.embeddings = None
        
        # Configure Gemini
        genai.configure(api_key=api_key)
        
        # Set up generation config
        self.generation_config = {
            "temperature": 1,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 8192,
            "response_mime_type": "text/plain",
        }
        
        # Initialize the model
        self.model = genai.GenerativeModel(
            model_name="gemini-2.0-flash-exp",
            generation_config=self.generation_config,
        )
        
        # Start chat session
        self.chat_session = self.model.start_chat(history=[])
        
    def load_data(self, file_path):
        """Load and prepare garage data"""
        self.garages_df = pd.read_csv(file_path)
        self.generate_embeddings()
        
    def generate_embeddings(self):
        """Generate embeddings for all garages"""
        text_for_embedding = self.garages_df.apply(
            lambda x: f"{x['Garage Name']} {x['Location']} {x['City']}", 
            axis=1
        )
        self.embeddings = self.embedding_model.encode(text_for_embedding.tolist())
        
    def find_relevant_garages(self, query, top_k=5):
        """Find most relevant garages for the query"""
        query_embedding = self.embedding_model.encode([query])
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        top_indices = similarities.argsort()[-top_k:][::-1]
        return self.garages_df.iloc[top_indices]
    
    def generate_prompt(self, query, relevant_garages):
        """Generate structured prompt for Gemini"""
        garage_context = "\n".join([
            f"Garage: {row['Garage Name']}\n"
            f"Location: {row['Location']}\n"
            f"City: {row['City']}\n"
            f"Postcode: {row['Postcode']}\n"
            f"Phone: {row['Phone']}\n"
            f"Email: {row['Email']}\n"
            f"Website: {row['Website']}\n"
            for _, row in relevant_garages.iterrows()
        ])
        
        prompt = f"""
        Act as an expert automotive service assistant. Analyze the following garage information to answer the user's query.
        
        Available Garages:
        {garage_context}
        
        User Query: {query}
        
        Think about this step by step:
        1. What specific needs does the user have?
        2. Which garages best match these needs?
        3. What are the key factors (location, services, contact options)?
        4. What additional information would be helpful?
        
        Provide a clear, detailed response that:
        - Answers the query directly
        - Recommends specific garages with reasons
        - Includes relevant contact details
        - Adds helpful context or suggestions
        """
        return prompt

    def process_query(self, query):
        """Process query and get response from Gemini"""
        try:
            relevant_garages = self.find_relevant_garages(query)
            prompt = self.generate_prompt(query, relevant_garages)
            
            # Get response from chat session
            response = self.chat_session.send_message(prompt)
            
            return response.text, relevant_garages
            
        except Exception as e:
            st.error(f"Error processing query: {str(e)}")
            return "I apologize, but I encountered an error processing your request. Please try again.", None

def init_session_state():
    """Initialize session state variables"""
    if 'messages' not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": """Welcome! I'm your automotive service assistant. I can help you with:
                - Finding suitable garages in your area
                - Getting service recommendations
                - Understanding garage specialties
                - Connecting with automotive experts
                
                How can I assist you today?"""
            }
        ]
    if 'agent' not in st.session_state:
        st.session_state.agent = AutoServiceAgent()

def main():
    st.set_page_config(page_title="Auto Service Assistant", layout="wide")
    
    st.title("🚗 Smart Garage Assistant")
    
    # Initialize session state
    init_session_state()
    
    # Sidebar for file upload and information
    with st.sidebar:
        st.header("Upload Data")
        uploaded_file = st.file_uploader("Upload Garage Data (CSV)", type=['csv'])
        
        if uploaded_file is not None:
            try:
                st.session_state.agent.load_data(uploaded_file)
                st.success("✅ Garage data loaded successfully!")
                total_garages = len(st.session_state.agent.garages_df)
                st.info(f"📊 Total garages loaded: {total_garages}")
            except Exception as e:
                st.error(f"Error loading data: {str(e)}")
        
        st.markdown("---")
        st.markdown("""
        ### How to Use
        1. Upload your garage data CSV
        2. Ask questions about:
           - Garage locations
           - Services offered
           - Contact information
           - Recommendations
        3. Get AI-powered responses
        """)
    
    # Main chat interface
    chat_container = st.container()
    
    with chat_container:
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
                # Display garage cards if available
                if "garages" in message and message["garages"] is not None:
                    for _, garage in message["garages"].iterrows():
                        with st.expander(f"🔍 {garage['Garage Name']}", expanded=True):
                            cols = st.columns([2, 1])
                            with cols[0]:
                                st.markdown(f"**Location:** {garage['Location']}, {garage['City']}")
                                st.markdown(f"**Postcode:** {garage['Postcode']}")
                            with cols[1]:
                                st.markdown(f"**Phone:** {garage['Phone']}")
                                if pd.notna(garage['Email']):
                                    st.markdown(f"**Email:** {garage['Email']}")
                                if pd.notna(garage['Website']):
                                    st.markdown(f"**Website:** [{garage['Website']}]({garage['Website']})")
    
    # Chat input
    if prompt := st.chat_input("Ask me about garage services, locations, or repairs..."):
        if st.session_state.agent.garages_df is None:
            st.error("⚠️ Please upload garage data first!")
            return
            
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get agent response
        with st.chat_message("assistant"):
            with st.spinner("Analyzing garages and generating response..."):
                response, relevant_garages = st.session_state.agent.process_query(prompt)
                st.markdown(response)
                
                # Add response to messages
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response,
                    "garages": relevant_garages
                })

if __name__ == "__main__":
    main()
