import streamlit as st
import pandas as pd
from vertexai.preview.generative_models import GenerativeModel
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os
from dotenv import load_dotenv
import requests
import json

# Load environment variables
load_dotenv()

class AutoServiceAgent:
    def __init__(self):
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.garages_df = None
        self.embeddings = None
        self.api_key = os.getenv('AIzaSyCcMZPrzP5me7Rl4pmAc1Nn5vUDSan5Q6E')
        
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
        You are an expert automotive service assistant. Use the following garage information to answer the user's query.
        
        Context:
        {garage_context}
        
        User Query: {query}
        
        Think through this step by step:
        1. Analyze the user's needs and the available garage information
        2. Consider location and accessibility factors
        3. Match services and specializations
        4. Evaluate contact options and unique advantages
        
        Provide a helpful response that:
        - Directly answers the user's query
        - Recommends specific garages when relevant
        - Includes contact information
        - Offers additional useful suggestions
        """
        return prompt

    def get_gemini_response(self, prompt):
        """Get response from Gemini API using direct REST API call"""
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={self.api_key}"
        
        headers = {
            'Content-Type': 'application/json'
        }
        
        data = {
            "contents": [{
                "parts": [{
                    "text": prompt
                }]
            }]
        }
        
        response = requests.post(url, headers=headers, json=data)
        
        if response.status_code == 200:
            response_json = response.json()
            try:
                generated_text = response_json['candidates'][0]['content']['parts'][0]['text']
                return generated_text
            except (KeyError, IndexError):
                return "I apologize, but I couldn't generate a proper response at the moment."
        else:
            return f"Error: Unable to get response from API (Status code: {response.status_code})"

    def process_query(self, query):
        """Process query and get response"""
        relevant_garages = self.find_relevant_garages(query)
        prompt = self.generate_prompt(query, relevant_garages)
        response = self.get_gemini_response(prompt)
        return response, relevant_garages

def init_session_state():
    """Initialize session state variables"""
    if 'messages' not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Hello! I'm your automotive service assistant. How can I help you today?"
            }
        ]
    if 'agent' not in st.session_state:
        st.session_state.agent = AutoServiceAgent()

def main():
    st.set_page_config(page_title="Auto Service Assistant", layout="wide")
    
    st.title("🚗 Auto Service Assistant")
    
    # Initialize session state
    init_session_state()
    
    # Sidebar for file upload
    with st.sidebar:
        st.header("Configuration")
        uploaded_file = st.file_uploader("Upload Garage Data (CSV)", type=['csv'])
        
        if uploaded_file is not None:
            st.session_state.agent.load_data(uploaded_file)
            st.success("Data loaded successfully!")
        
        st.markdown("---")
        st.markdown("""
        ### About
        This Auto Service Assistant helps you find and connect with automotive services.
        
        Features:
        - Smart garage recommendations
        - Location-based search
        - Service matching
        - Contact information
        """)
    
    # Main chat interface
    chat_container = st.container()
    
    with chat_container:
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
                # Display garage cards if available
                if "garages" in message:
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
    if prompt := st.chat_input("Ask about garage services, locations, or repairs..."):
        if st.session_state.agent.garages_df is None:
            st.error("Please upload garage data first!")
            return
            
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get agent response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
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
