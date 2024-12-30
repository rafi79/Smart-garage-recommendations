import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import requests
import json

class AutoServiceAgent:
    def __init__(self):
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.garages_df = None
        self.embeddings = None
        self.api_key = "AIzaSyCcMZPrzP5me7Rl4pmAc1Nn5vUDSan5Q6E"
        self.chat_history = []

    def call_gemini_api(self, prompt):
        """Call Gemini API using the provided API key"""
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={self.api_key}"
        
        headers = {
            "Content-Type": "application/json"
        }
        
        data = {
            "contents": [{
                "parts": [{
                    "text": prompt
                }]
            }]
        }

        try:
            response = requests.post(url, headers=headers, json=data)
            if response.status_code == 200:
                result = response.json()
                if 'candidates' in result and result['candidates']:
                    return result['candidates'][0]['content']['parts'][0]['text']
            return "I apologize, but I couldn't generate a proper response. Please try rephrasing your question."
        except Exception as e:
            st.error(f"Error calling Gemini API: {str(e)}")
            return "I encountered an error while processing your request. Please try again."
        
    def load_data(self, file_path):
        """Load and prepare garage data"""
        try:
            self.garages_df = pd.read_csv(file_path)
            self.generate_embeddings()
            return True
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            return False
        
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
            f"Phone: {row['Phone']}\n"
            f"Email: {row.get('Email', 'Not available')}\n"
            f"Website: {row.get('Website', 'Not available')}\n"
            for _, row in relevant_garages.iterrows()
        ])
        
        prompt = f"""You are an automotive service assistant. Help the user find the most suitable garage based on their query.

Available Garages:
{garage_context}

User Question: {query}

Please analyze:
1. User's specific requirements
2. Best matching garages
3. Location and accessibility
4. Contact options

Provide a clear response that:
- Directly answers the query
- Suggests the most suitable garage(s)
- Includes relevant contact info
- Offers any helpful additional information"""
        
        return prompt

    def process_query(self, query):
        """Process user query and get response"""
        try:
            relevant_garages = self.find_relevant_garages(query)
            prompt = self.generate_prompt(query, relevant_garages)
            response = self.call_gemini_api(prompt)
            return response, relevant_garages
        except Exception as e:
            st.error(f"Error processing query: {str(e)}")
            return "I apologize, but I encountered an error. Please try again.", None

# Initialize Streamlit interface
def init_session_state():
    if 'messages' not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "👋 Hello! I'm your garage finder assistant. Upload a CSV file with garage data, and I'll help you find the perfect garage for your needs!"
            }
        ]
    if 'agent' not in st.session_state:
        st.session_state.agent = AutoServiceAgent()

def main():
    st.set_page_config(page_title="Garage Finder Assistant", layout="wide")
    
    st.title("🚗 Garage Finder Assistant")
    
    init_session_state()
    
    # Sidebar
    with st.sidebar:
        st.header("📁 Data Upload")
        uploaded_file = st.file_uploader("Upload Garage Data (CSV)", type=['csv'])
        
        if uploaded_file is not None:
            if st.session_state.agent.load_data(uploaded_file):
                st.success("✅ Data loaded successfully!")
                st.info(f"📊 Total garages: {len(st.session_state.agent.garages_df)}")
        
        st.markdown("---")
        st.markdown("""
        ### 🔍 How to Use
        1. Upload your garage CSV
        2. Ask questions like:
           - "Find a garage in Bath"
           - "Who can fix my electric car?"
           - "Need a garage with good reviews"
        3. Get AI-powered suggestions
        """)
    
    # Chat interface
    chat_container = st.container()
    
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
                if "garages" in message and message["garages"] is not None:
                    for _, garage in message["garages"].iterrows():
                        with st.expander(f"🔧 {garage['Garage Name']}", expanded=True):
                            col1, col2 = st.columns([2,1])
                            with col1:
                                st.markdown(f"📍 **Location:** {garage['Location']}, {garage['City']}")
                            with col2:
                                st.markdown(f"📞 **Phone:** {garage['Phone']}")
                                if pd.notna(garage.get('Email')):
                                    st.markdown(f"📧 **Email:** {garage['Email']}")
                                if pd.notna(garage.get('Website')):
                                    st.markdown(f"🌐 **Website:** [{garage['Website']}]({garage['Website']})")
    
    # User input
    if prompt := st.chat_input("Ask about garages..."):
        if st.session_state.agent.garages_df is None:
            st.error("⚠️ Please upload garage data first!")
            return
            
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("🔍 Finding the best garages for you..."):
                response, relevant_garages = st.session_state.agent.process_query(prompt)
                st.markdown(response)
                
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response,
                    "garages": relevant_garages
                })

if __name__ == "__main__":
    main()
