import streamlit as st
import pandas as pd
import google.generativeai as genai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class AutoServiceAgent:
    def __init__(self, api_key='AIzaSyCcMZPrzP5me7Rl4pmAc1Nn5vUDSan5Q6E'):
        self.garages_df = None
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = None
        
        # Configure Gemini
        genai.configure(api_key=api_key)
        
        # Set generation config
        self.generation_config = {
            "temperature": 1,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 8192,
            "response_mime_type": "text/plain",
        }
        
        # Initialize model
        self.model = genai.GenerativeModel(
            model_name="gemini-2.0-flash-exp",
            generation_config=self.generation_config,
        )
        
        # Start chat session
        self.chat = self.model.start_chat(history=[])
        
    def load_data(self, file_path):
        """Load and prepare garage data"""
        try:
            self.garages_df = pd.read_csv(file_path)
            self._create_search_index()
            return True
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            return False
        
    def _create_search_index(self):
        """Create TF-IDF matrix for garage search"""
        search_texts = self.garages_df.apply(
            lambda x: f"{x['Garage Name']} {x['Location']} {x['City']}", 
            axis=1
        ).tolist()
        self.tfidf_matrix = self.vectorizer.fit_transform(search_texts)
        
    def find_relevant_garages(self, query, top_k=5):
        """Find most relevant garages using TF-IDF and cosine similarity"""
        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix)[0]
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
        
        prompt = f"""You are an automotive service assistant helping users find the right garage. 

Available Garages:
{garage_context}

User Question: {query}

Provide a helpful response that:
1. Addresses the user's specific needs
2. Recommends the most suitable garage(s) with reasons
3. Includes relevant contact details
4. Adds any useful suggestions

Format your response clearly with appropriate sections and bullet points."""
        
        return prompt

    def process_query(self, query):
        """Process user query and get Gemini response"""
        try:
            relevant_garages = self.find_relevant_garages(query)
            prompt = self.generate_prompt(query, relevant_garages)
            
            # Get response from Gemini
            response = self.chat.send_message(prompt)
            
            return response.text, relevant_garages
            
        except Exception as e:
            st.error(f"Error processing query: {str(e)}")
            return "I apologize, but I encountered an error. Please try again.", None

def init_session_state():
    """Initialize Streamlit session state"""
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
    """Main Streamlit application"""
    st.set_page_config(page_title="Smart Garage Finder", layout="wide")
    
    st.title("🚗 Smart Garage Finder")
    
    init_session_state()
    
    # Sidebar
    with st.sidebar:
        st.header("📁 Upload Data")
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
           - "I need an electrical specialist"
           - "Looking for a garage with good reviews"
        3. Get AI-powered recommendations
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
