# Auto Service Assistant

An intelligent automotive service assistant powered by Streamlit and Google's Gemini AI. This application helps users find and connect with automotive services using natural language processing and smart garage recommendations.

## Features

- 🤖 AI-powered conversations using Google Gemini
- 🔍 Smart garage recommendations using semantic search
- 📍 Location-based service matching
- 💬 Interactive chat interface
- 📊 Detailed garage information display
- 🔄 CSV data integration

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/auto-service-assistant.git
cd auto-service-assistant
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file with your Google API key:
```
GOOGLE_API_KEY=your_gemini_api_key_here
```

## Usage

1. Run the Streamlit app:
```bash
streamlit run app.py
```

2. Upload your garage data CSV file using the sidebar uploader

3. Start chatting with the assistant!

## Data Format

The application expects a CSV file with the following columns:
- Serial
- Garage Name
- Location
- City
- Postcode
- Phone
- Email
- Website

## Features in Detail

### Semantic Search
- Uses sentence transformers for encoding garage information
- Finds relevant garages based on user queries
- Employs cosine similarity for matching

### AI Conversation
- Powered by Google's Gemini AI model
- Structured prompting for better responses
- Context-aware recommendations

### User Interface
- Clean and intuitive Streamlit interface
- Real-time chat experience
- Expandable garage information cards
- Easy data upload through sidebar

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Google Generative AI
- Streamlit
- Sentence Transformers
- All contributors and users of this project
