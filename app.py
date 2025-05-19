# lida visualization with sql connection
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.utilities import SQLDatabase
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import AzureChatOpenAI
from sqlalchemy.engine import URL
from sqlalchemy import create_engine
import streamlit as st
import pandas as pd
import os
import re
from lida import Manager, TextGenerationConfig, llm
import base64
from io import BytesIO
from PIL import Image

os.environ["OPENAI_ENDPOINT"] = 'https://openai-models-for-poc.openai.azure.com/'
os.environ["OPENAI_API_KEY"] = '208d0bba3243411b814e4aa5d3db777b'
os.environ["OPENAI_API_VERSION"] = "2024-02-15-preview"

def init_database() -> SQLDatabase:
  engine_str = URL.create(
    drivername="mssql+pyodbc",
    username=os.getenv('SQL_DB_UID'),
    password=os.getenv('SQL_DB_PWD'),
    host=os.getenv('SQL_DB_HOST'),
    port=os.getenv('SQL_DB_PORT'),
    database=os.getenv('SQL_DB_NAME'),
    query={
        "driver": "ODBC Driver 18 for SQL Server",
        "TrustServerCertificate": "Yes",
        "Connection Timeout": "30",
        "Encrypt": "yes",
    },
  )
  engine = create_engine(engine_str)
  return SQLDatabase(engine,schema='dbo')

def get_sql_chain(db):
  template = """
    You are a data analyst at a company. You are interacting with a user who is asking you questions about the company's database.
    Based on the table schema below, write a SQL query that would answer the user's question. Take the conversation history into account.
    
    <SCHEMA>{schema}</SCHEMA>
    
    Conversation History: {chat_history}
    
    Write only the SQL query and nothing else. Do not wrap the SQL query in any other text, not even backticks.
    
    For example:
    Question: What is YTD for Investments KPI?
    SQL Query: SELECT SUM(KPI_ACTUAL) AS YTD_Investments FROM actuals WHERE KPI_ID = 1002;
    Question: What is the actual and target for ROE KPI?
    SQL Query: SELECT a.KPI_DATA_FOR, a.KPI_ACTUAL, t.KPI_TARGET 
                FROM actuals a 
                JOIN targets t ON a.KPI_ID = t.KPI_ID 
                WHERE a.KPI_ID = t.KPI_ID and a.KPI_DATA_FOR = t.KPI_DATA_FOR and a.KPI_ID = 1001
                ORDER BY a.KPI_DATA_FOR;
    Reason: I have hard coded KPI_ID = 1001 because this is the KPI_ID value for ROE KPI. So whenever you receive a different KPI you need to 
    pass the KPI ID for that KPI from {schema}
    Your turn:
    
    Question: {question}
    SQL Query:
    """
    
  prompt = ChatPromptTemplate.from_template(template)

  # Connect to Azure OpenAI
  llm = AzureChatOpenAI(
      model=os.getenv('AZURE_OPENAI_DEPLOYMENT'),
      azure_deployment=os.getenv('AZURE_OPENAI_DEPLOYMENT'),
      api_key=os.getenv('AZURE_OPENAI_API_KEY'),
      api_version=os.getenv('AZURE_OPENAI_API_VERSION'),
      azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT'),
      temperature=0
  )
  
  def get_schema(_):
    return db.get_table_info()
  
  return (
    RunnablePassthrough.assign(schema=get_schema)
    | prompt
    | llm
    | StrOutputParser()
  )
    
def get_response(user_query: str, db: SQLDatabase, chat_history: list):
  sql_chain = get_sql_chain(db)
  
  template = """
    You are a data analyst at a company. You are interacting with C-level executives who is asking you questions about their company's database.
    Based on the table schema below, question, sql query, and sql response, write a natural language response.
    
    <SCHEMA>{schema}</SCHEMA>

    Conversation History: {chat_history}
    SQL Query: <SQL>{query}</SQL>
    User question: {question}
    SQL Response: {response}

    Remember:
    Accuracy is paramount. Double-check all figures and calculations before including them in your response.
    Be objective and data-driven in your analysis.
    Use clear, professional language appropriate for high-level executive communication.
    If the user query is ambiguous or requires clarification, state your assumptions clearly.
    """
  
  prompt = ChatPromptTemplate.from_template(template)

  llm = AzureChatOpenAI(
      model = os.getenv('AZURE_OPENAI_DEPLOYMENT'),
      azure_deployment = os.getenv('AZURE_OPENAI_DEPLOYMENT'),
      api_key = os.getenv('AZURE_OPENAI_API_KEY'),
      api_version = os.getenv('AZURE_OPENAI_API_VERSION'),
      azure_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT'),
      temperature = 0
  )
  
  chain = (
    RunnablePassthrough.assign(query=sql_chain).assign(
      schema=lambda _: db.get_table_info(),
      response=lambda vars: db.run(vars["query"]),
    )
    | prompt
    | llm
    | StrOutputParser()
  )
  
  return chain.invoke({
    "question": user_query,
    "chat_history": chat_history,
  })

# # Initialize the Lida Manager
# lida = Manager(text_gen=llm("openai", api_type="azure"))
# textgen_config = TextGenerationConfig(n=1, temperature=0.5, model="gpt-4o-mini", use_cache=True)
# Initialize LIDA manager
lida = Manager(text_gen=llm("openai", api_type="azure"))
textgen_config = TextGenerationConfig(n=1, temperature=0.5, model="gpt-4o-mini", use_cache=True)


# Function to convert a base64 string to an image
def base64_to_image(base64_string):
    byte_data = base64.b64decode(base64_string)
    return Image.open(BytesIO(byte_data))

# Function to extract a simple markdown table from text
def extract_table(text):
    table_pattern = r'(\|[^\n]*\|(?:\n\|[-:]+[-|:]*\|)(?:\n\|[^\n]*\|)+)'
    match = re.search(table_pattern, text)
    if match:
        return match.group(0)
    return None

# Function to truncate header and data to fit the cell size
def truncate_string(s, max_length=50):
    if len(s) > max_length:
        return s[:max_length] + '...'  
    return s

# Function to convert markdown table to DataFrame
def markdown_table_to_dataframe(table_str):
    lines = table_str.strip().split('\n')
    header = [truncate_string(col.strip()) for col in lines[0].split('|')[1:-1]]
    data = []
    for line in lines[2:]:
        row = [truncate_string(cell.strip()) for cell in line.split('|')[1:-1]]
        data.append(row)
    return pd.DataFrame(data, columns=header)  
 

# Initialize Streamlit app
st.set_page_config(page_title="KPI Analysis Chat Bot", page_icon=":speech_balloon:")
st.title("KPI Analysis Chat Bot")

# Initialize session states
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hello! I'm a SQL assistant. Ask me anything about your database."),
    ]

if "table_df" not in st.session_state:
    st.session_state.table_df = None

if "db" not in st.session_state:
    st.session_state.db = None

# Create a container for chat messages
chat_container = st.container()

# Sidebar for database connection
with st.sidebar:
    st.subheader("Settings")
    
    if st.button("Connect"):
        with st.spinner("Connecting to database..."):
            try:
                db = init_database()
                st.session_state.db = db
                st.success("Connected to database!")
            except Exception as e:
                st.error(f"Failed to connect to database: {str(e)}")

# Function to reset the filtered table, column selection, and generated visualization
def reset_visualizations():
    st.session_state.table_df = None
    st.session_state.filtered_df = None
    st.session_state.selected_columns = []

# Display chat history in the container
with chat_container:
    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.markdown(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.markdown(message.content)

# Handle user input
user_query = st.chat_input("What would you like to ask?")
if user_query is not None and user_query.strip() != "":
    # Reset visualizations for each new query
    reset_visualizations()

    if not st.session_state.db:
        st.error("Please connect to the database first!")
    else:
        # Add user message to chat history
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        with st.chat_message("Human"):
            st.markdown(user_query)
        
        # Get and display AI response
        with st.chat_message("AI"):
            with st.spinner("Thinking..."):
                response = get_response(user_query, st.session_state.db, st.session_state.chat_history)
                st.markdown(response)
                st.session_state.chat_history.append(AIMessage(content=response))
                
                # Extract table data if present
                table_data = extract_table(response)
                if table_data:
                    try:
                        df = markdown_table_to_dataframe(table_data)
                        st.session_state.table_df = df
                    except Exception as e:
                        st.error(f"Error processing the table: {e}")

# Visualization options if table data exists
if st.session_state.table_df is not None:
    with st.sidebar:
        st.subheader("Column Selection")
        columns = st.session_state.table_df.columns.tolist()
        
        # Multi-select for column selection, stored in session state
        st.session_state.selected_columns = st.multiselect(
            "Select columns to display and visualize:",
            options=columns,
            default=columns  # Initially select all columns
        )
        
    # Filter DataFrame based on selected columns
    if st.session_state.selected_columns:
        st.session_state.filtered_df = st.session_state.table_df[st.session_state.selected_columns]
        
        # Display filtered data
        st.write("Filtered Data:", st.session_state.filtered_df)

        # Trigger visualization generation
        if st.button("Visualize Filtered Data"):
            with st.spinner("Generating visualization..."):
                try:
                    # Save filtered DataFrame as CSV for processing
                    path_to_save = "filtered_data.csv"
                    st.session_state.filtered_df.to_csv(path_to_save, index=False)

                    # Generate visualizations with LIDA
                    summary = lida.summarize(path_to_save, summary_method="default", textgen_config=textgen_config)
                    goals = lida.goals(summary, n=1, textgen_config=textgen_config)
                    charts = lida.visualize(summary=summary, goal=goals[0], textgen_config=textgen_config, library="seaborn")

                    if charts:
                        img_base64_string = charts[0].raster
                        img = base64_to_image(img_base64_string)
                        st.image(img, caption='Generated Visualization for Selected Columns')
                    else:
                        st.warning("No charts were generated from the summary.")

                    # Clean up the CSV
                    if os.path.exists(path_to_save):
                        os.remove(path_to_save)
                
                except Exception as e:
                    st.error(f"Error generating visualization: {str(e)}")
