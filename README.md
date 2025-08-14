# news_research_tool_project

🤖 RockyBot Pro: AI-Powered News Research Assistant
RockyBot Pro is a sophisticated news research assistant built with Streamlit and the LangChain framework. It leverages Google Gemini models to perform lightning-fast analysis of global news sources. The application provides a comprehensive platform for users to research a topic, analyze articles, and get intelligent, AI-powered answers.

✨ Features
Multi-Source News Search: Scans a wide range of global news outlets and RSS feeds, including BBC, CNN, Reuters, TechCrunch, Bloomberg, and more.

Intelligent Q&A Assistant: Once articles are analyzed, you can ask detailed questions and receive answers based on the gathered information.

Advanced AI Models: Utilizes various Gemini models, including the fast and powerful gemini-2.0-flash-exp, for superior processing and response quality.

Customizable Search: Adjust search depth, time range, and include or exclude specific keywords to refine your research.

Smart Analytics: Provides a dashboard with key metrics such as articles processed, unique sources, and source distribution to give you a clear overview of your research.

Export Functionality: Easily download your chat history and the list of analyzed articles in JSON format for future reference.

Session Management: Clear all data with a single click and review your recent search history directly in the sidebar.

Responsive and Thematic UI: A custom-styled, clean, and responsive user interface built with Streamlit, featuring a professional and engaging design.

🚀 Getting Started
Prerequisites
You will need Python installed on your system and an API key from Google AI Studio. A News API key is optional but highly recommended for enhanced search results.

Google Gemini API Key: Get your free key from Google AI Studio.

News API Key (Optional): Get a free API key from NewsAPI.org.

Installation
Clone the repository:

git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name

Create and activate a virtual environment (recommended):

python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate

Install the required packages:

pip install -r requirements.txt

Note: The requirements.txt file should contain the following packages:

streamlit
langchain
langchain-google-genai
faiss-cpu # or faiss-gpu
beautifulsoup4
requests
feedparser
unstructured

🤖 Usage
Run the Streamlit application:

streamlit run main.py

Enter your API Keys: Start by entering your Google API key in the sidebar's "API Configuration" section. If you have a News API key, you can enter it there as well.

Search & Analyze: In the main panel, enter your research topic (e.g., "artificial intelligence breakthrough"). You can fine-tune your search with the "Advanced Search Options" or the sidebar configurations. Click the "Search & Analyze News" button to begin.

Ask Questions: Once the analysis is complete, a knowledge base is created. You can then use the "Intelligent Q&A Assistant" to ask questions about the analyzed articles.

View and Export: Use the different sections to view your chat history, content analytics, and export your data for later use.

🤝 Credits
This project was built using the following technologies:

Streamlit: For creating the interactive web application.

LangChain: For orchestrating the calls to the language models and handling data processing.

Google Gemini: The powerful large language model providing the core AI capabilities.

FAISS: For efficient similarity search and retrieval of information from the processed articles.

Requests, BeautifulSoup, Feedparser: For web scraping and fetching content from various news sources.
