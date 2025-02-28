from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAI
from langchain_experimental.agents import create_pandas_dataframe_agent
import pandas as pd
import base64
import matplotlib.pyplot as plt
import io
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import gc

# Load environment variables
load_dotenv()

# Load Titanic dataset lazily
def load_data():
    return pd.read_csv("data/titanic.csv")

df = load_data()

# Initialize Gemini model
gemini_model = GoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.8)

# Create LangChain agent
agent = create_pandas_dataframe_agent(gemini_model, df, verbose=True, allow_dangerous_code=True)

# Load NLP models
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

# Cache embeddings for frequently asked questions
from functools import lru_cache

@lru_cache(maxsize=32)
def find_best_matching_column(question):
    """Finds the dataset column that best matches the user's question using embeddings."""
    question_embedding = embedding_model.encode([question])[0]
    similarities = cosine_similarity([question_embedding], column_embeddings)[0]
    best_match_index = similarities.argmax()
    return column_names[best_match_index]

# Function to generate a matplotlib chart as a base64-encoded string
def generate_chart(column, chart_type='hist', **kwargs):
    fig, ax = plt.subplots(figsize=(6, 4))  # Smaller figure size to save memory
    
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
    plt.close(fig)  # Close the plot to free memory
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

    finally:
        gc.collect()  # Force garbage collection after processing


# from dotenv import load_dotenv
# from langchain_google_genai import GoogleGenerativeAI
# from langchain_experimental.agents import create_pandas_dataframe_agent
# import pandas as pd
# import os
# import base64
# import matplotlib.pyplot as plt
# import io
# from sklearn.metrics.pairwise import cosine_similarity
# from sentence_transformers import SentenceTransformer
# import gc
# from functools import lru_cache

# # Load environment variables
# load_dotenv()

# # Load Titanic dataset (Optimized: Using Parquet format to reduce memory usage)
# def load_data():
#     parquet_path = "data/titanic.parquet"
#     csv_path = "data/titanic.csv"

#     if os.path.exists(parquet_path):
#         return pd.read_parquet(parquet_path)
#     else:
#         df = pd.read_csv(csv_path)
#         columns_to_keep = ["Survived", "Pclass", "Sex", "Age", "Fare", "Embarked"]
#         df = df[columns_to_keep]
#         df.to_parquet(parquet_path, compression="gzip")  # Convert to Parquet for efficiency
#         return df

# df = load_data()

# # Initialize Gemini model
# gemini_model = GoogleGenerativeAI(model="gemini-pro", temperature=0.8)

# # Create LangChain agent (Optimized: No verbose mode, disabled dangerous code execution)
# agent = create_pandas_dataframe_agent(gemini_model, df, verbose=False, allow_dangerous_code=False)

# # Lazy loading for embedding model to save memory
# def get_embedding_model():
#     """Load the embedding model only when needed."""
#     if not hasattr(get_embedding_model, "model"):
#         get_embedding_model.model = SentenceTransformer("paraphrase-MiniLM-L3-v2")  # Lighter model
#     return get_embedding_model.model

# embedding_model = get_embedding_model()

# # Dataset column descriptions
# column_descriptions = {
#     "Survived": "Survival status (1 = survived, 0 = did not survive)",
#     "Pclass": "Passenger class (1st, 2nd, 3rd)",
#     "Sex": "Gender of the passenger",
#     "Age": "Age of the passenger",
#     "Fare": "Price of the ticket",
#     "Embarked": "Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)"
# }

# # Create embeddings for dataset columns
# column_names = list(column_descriptions.keys())
# column_texts = [f"{name}: {desc}" for name, desc in column_descriptions.items()]
# column_embeddings = embedding_model.encode(column_texts)

# # Cache embeddings for frequently asked questions
# @lru_cache(maxsize=32)
# def find_best_matching_column(question):
#     """Finds the dataset column that best matches the user's question using embeddings."""
#     question_embedding = embedding_model.encode([question])[0]
#     similarities = cosine_similarity([question_embedding], column_embeddings)[0]
#     best_match_index = similarities.argmax()
#     return column_names[best_match_index]

# # Function to generate a matplotlib chart as a base64-encoded string (Optimized: Smaller size, lower DPI)
# def generate_chart(column, chart_type='hist', **kwargs):
#     fig, ax = plt.subplots(figsize=(4, 3), dpi=80)  # Smaller figure size, lower DPI
    
#     try:
#         if chart_type == 'hist':
#             df[column].dropna().plot(kind='hist', ax=ax, title=f"Distribution of {column}", **kwargs)
#         elif chart_type == 'bar':
#             df[column].value_counts().plot(kind='bar', ax=ax, title=f"Count of {column}", **kwargs)
#         elif chart_type == 'pie':
#             df[column].value_counts().plot(kind='pie', ax=ax, title=f"Distribution of {column}", autopct='%1.1f%%', **kwargs)
#         elif chart_type == 'scatter':
#             df.plot(kind='scatter', x=kwargs.get('x'), y=kwargs.get('y'), ax=ax, title=f"Scatter plot of {kwargs.get('x')} vs {kwargs.get('y')}")
        
#         buf = io.BytesIO()
#         plt.savefig(buf, format='png')
#         plt.close(fig)  # Close the plot to free memory
#         buf.seek(0)
        
#         return base64.b64encode(buf.read()).decode('utf-8')

#     except Exception as e:
#         print(f"Chart generation error: {e}")  # Log error
#         return None

# def detect_chart_request(question):
#     """Determines the best chart type and column based on the user's question."""
#     column = find_best_matching_column(question)  # Find best column dynamically
#     words = question.lower().split()
    
#     if "histogram" in words or "distribution" in words:
#         return column, "hist"
#     elif "bar chart" in words or "count" in words:
#         return column, "bar"
#     elif "pie chart" in words or "percentage" in words:
#         return column, "pie"
#     elif "scatter plot" in words or "relationship" in words:
#         return column, "scatter"

#     return column, "hist"  # Default to histogram if chart type is unclear

# def process_query(question):
#     """Processes user queries, providing both text answers and visualizations, while handling errors gracefully."""
#     try:
#         full_prompt = f"Dataset: Titanic\n\nUser: {question}\n\nAnswer in a detailed and friendly way."

#         column, chart_type = detect_chart_request(question)
#         text_response = agent.run(full_prompt)

#         chart = None
#         if column:
#             chart = generate_chart(column, chart_type)

#         return text_response, chart

#     except Exception as e:
#         print(f"Error processing query: {e}")  # Log error
#         return "I'm sorry, but I couldn't process your request at the moment. Feel free to try again or to ask another question!", None

#     finally:
#         gc.collect()  # Force garbage collection after processing
