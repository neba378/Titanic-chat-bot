# Standard library imports
import io
import base64

# Third-party imports
from dotenv import load_dotenv
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from langchain_google_genai import GoogleGenerativeAI
from langchain_experimental.agents import create_pandas_dataframe_agent

# Load environment variables
load_dotenv()

# Load Titanic dataset
df = pd.read_csv("data/titanic.csv")

# Initialize Gemini model
gemini_model = GoogleGenerativeAI(model="gemini-pro", temperature=0.8)

# Create LangChain agent
agent = create_pandas_dataframe_agent(gemini_model, df, verbose=True, allow_dangerous_code=True)

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")  # Small, fast embedding model

# Dataset column descriptions (for better understanding)
column_descriptions = {
    "PassengerId": "Unique ID for each passenger",
    "Survived": "Survival status (1 = survived, 0 = did not survive)",
    "Pclass": "Passenger class (1st, 2nd, 3rd)",
    "Name": "Full name of the passenger",
    "Sex": "Gender of the passenger",
    "Age": "Age of the passenger",
    "SibSp": "Number of siblings/spouses aboard",
    "Parch": "Number of parents/children aboard",
    "Ticket": "Ticket number",
    "Fare": "Price of the ticket",
    "Cabin": "Cabin number",
    "Embarked": "Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)"
}

# Create embeddings for dataset columns
column_names = list(column_descriptions.keys())
column_texts = [f"{name}: {desc}" for name, desc in column_descriptions.items()]
column_embeddings = embedding_model.encode(column_texts)

def find_best_matching_column(question):
    """Finds the dataset column that best matches the user's question using embeddings."""
    question_embedding = embedding_model.encode([question])[0]
    similarities = cosine_similarity([question_embedding], column_embeddings)[0]
    
    best_match_index = similarities.argmax()
    best_column = column_names[best_match_index]
    
    return best_column

# Function to generate a matplotlib chart as a base64-encoded string
def generate_chart(column, chart_type='hist', **kwargs):
    fig, ax = plt.subplots()
    
    if chart_type == 'hist':
        df[column].dropna().plot(kind='hist', ax=ax, title=f"Distribution of {column}", **kwargs)
    elif chart_type == 'bar':
        df[column].value_counts().plot(kind='bar', ax=ax, title=f"Count of {column}", **kwargs)
    elif chart_type == 'pie':
        df[column].value_counts().plot(kind='pie', ax=ax, title=f"Distribution of {column}", autopct='%1.1f%%', **kwargs)
    elif chart_type == 'scatter':
        df.plot(kind='scatter', x=kwargs.get('x'), y=kwargs.get('y'), ax=ax, title=f"Scatter plot of {kwargs.get('x')} vs {kwargs.get('y')}")
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    
    return base64.b64encode(buf.read()).decode('utf-8')

def detect_chart_request(question):
    """Determines the best chart type and column based on the user's question."""
    column = find_best_matching_column(question)  # Find best column dynamically

    words = question.lower().split()
    
    if "histogram" in words or "distribution" in words:
        return column, "hist"
    elif "bar chart" in words or "count" in words:
        return column, "bar"
    elif "pie chart" in words or "percentage" in words:
        return column, "pie"
    elif "scatter plot" in words or "relationship" in words:
        return column, "scatter"

    return column, "hist"  # Default to histogram if chart type is unclear

def process_query(question):
    """Processes user queries, providing both text answers and visualizations, while handling errors gracefully."""
    try:
        full_prompt = f"Dataset: Titanic\n\nUser: {question}\n\nAnswer in a detailed and friendly way."

        column, chart_type = detect_chart_request(question)
        text_response = agent.run(full_prompt)

        if column:
            try:
                chart = generate_chart(column, chart_type)
                return text_response, chart
            except Exception as e:
                print(f"Chart generation error: {e}")  # Log the error
                return text_response, None  # Return text response even if chart fails

        return text_response, None  # Default return if no column found

    except Exception as e:
        print(f"Error processing query: {e}")  # Log the error
        return "I'm sorry, but I couldn't process your request at the moment. Feel free to try again or to ask another question!", None

