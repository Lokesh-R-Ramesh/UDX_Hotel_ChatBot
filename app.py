import json
import pandas as pd
import streamlit as st
import snowflake.connector
import os
import boto3
import cv2
import easyocr
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from PIL import Image
from transformers import pipeline
from langchain.schema import HumanMessage, SystemMessage
from PyPDF2 import PdfReader
from langchain_experimental.agents import create_pandas_dataframe_agent
from fuzzywuzzy import process
from langchain.schema import AIMessage
from langchain_community.chat_models import BedrockChat
from langchain.chains import RetrievalQA
from langchain.chains.question_answering import load_qa_chain
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import tempfile
from langchain.prompts import ChatPromptTemplate
from langchain_core.exceptions import OutputParserException
from langchain_core.exceptions import OutputParserException
from langchain.agents import AgentExecutor
import re

from dotenv import load_dotenv

load_dotenv()

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file using PyPDF2."""
    try:
        reader = PdfReader(pdf_path)
        text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
        return text
    except Exception as e:
        return f"Error reading PDF: {str(e)}"

def catergory(query,query_history,LLM):
            prompt = ChatPromptTemplate.from_template('''
                                       You are a smart routing agent. Your task is to classify the user's question into one of the following categories:
                                        DAX, CALCULATION, MEASURE, TABLE, POWER BI TABLE, DATA, NUMBER, DICTIONARY, MEANING, DEFINITION.

                                        Use the keywords and their alternative forms to decide the best fit. Choose the **single most relevant category** based on the user's intent.
                                        User prompt: "{query}"  
                                        User history: "{query_history}"
                                                      
                                        Based on the give histroy and the lasted prompt decide the tool.

                                        Here are your mappings:

                                        DAX: dax query, dax formula, power bi dax, calculate measure  
                                        POWER BI TABLE: data model, bi table, table structure  
                                        Snowflack: info, database, dataset , Actual Data linnked to dashboard 
                                        Excel: lexicon, glossary  
                                        Excel: define, definition, meaning of, what is, what does it mean, explain, explanation, clarification, description, interpret, elaborate, describe, terminology, term
                                        None : If the doesnot falls in any of them

                                        Only return the category name. Do not explain or include anything else.

                                        
                                        ''')
            
            messages = prompt.format_messages(query=query, query_history=query_history)

    # Send to the LLM
            response = LLM.invoke(messages)
            
            return getattr(response, "content", str(response))
    
st.set_page_config(layout="wide")
if "messages" not in st.session_state:
    st.session_state.messages = []
if "summary" not in st.session_state:
    st.session_state.summary = None
if "Data" not in st.session_state:
    st.session_state.Data = pd.read_excel("UDX_Hotel_Data.xlsx")
st.markdown(
    """
    <style>
        [data-testid="stSidebar"] {
            background-color: #99F6F6;  /* Dark background */
            color: black;  /* White text */
        }
        [data-testid="stSidebar"] h1, 
        [data-testid="stSidebar"] h2, 
        [data-testid="stSidebar"] h3, 
        [data-testid="stSidebar"] p {
            color: black;  /* Ensures all text is white */
        }
    </style>
    """,
    unsafe_allow_html=True
)
with st.sidebar:
    API = os.getenv("ACCESS_KEY")
    Secure_Key = os.getenv("SECRET_ACCESS_KEY")

    
    st.session_state.API = API
    st.session_state.Secure_Key = Secure_Key

    bedrock_runtime = boto3.client(
        service_name="bedrock-runtime",
        region_name="us-east-1",
        aws_access_key_id=st.session_state.API,
        aws_secret_access_key=st.session_state.Secure_Key,
    )

    llm = BedrockChat(
        model_id="anthropic.claude-3-sonnet-20240229-v1:0",
        client=bedrock_runtime,
        model_kwargs={
            "max_tokens": 1000,
            "temperature": 0.1,
            "top_p": 0.9,
            "top_k": 50
        }
    )

    json_path = r"DataModelSchema.json"
    excel_path = r"Data DictionaryChat bot.xlsx"

    json_data = pd.read_json(json_path, encoding='utf-16')
    df = pd.DataFrame()

    # Process JSON data
    table_1 = list(json_data["model"]['tables'])
    for i in range(len(table_1)):
        table = table_1[i]
        if 'measures' in table:
            df = pd.concat([df, pd.DataFrame(table['measures'])], ignore_index=True)
    Measure_Table = df[["name", "expression"]]
    Measure_Table = Measure_Table.rename(columns={"expression": "DAX", "name": "Dax Name"})

    df_1 = pd.DataFrame(columns=['Table Name', 'Column Name'])
    tables = json_data["model"]['tables']
    for table in tables:
        if 'columns' in table:
            for column in table['columns']:
                df_1 = pd.concat([df_1, pd.DataFrame({'Table Name': [table['name']], 'Column Name': [column['name']], 'Data Type': [column['dataType']]})], ignore_index=True)

    # Process Excel data
    xls_data = pd.read_excel(excel_path)



    #PDF = RetrievalQA.from_chain_type(llm=llm,chain_type="stuff",retriever=retriever)
    raw_dax_agent = create_pandas_dataframe_agent(
    llm=llm,
    df=Measure_Table,
    verbose=True,
    allow_dangerous_code=True,
    number_of_head_rows=Measure_Table.shape[0]
    )

    # Step 2: Wrap it in AgentExecutor with error handling
    DAX = AgentExecutor.from_agent_and_tools(
        agent=raw_dax_agent.agent,
        tools=raw_dax_agent.tools,
        handle_parsing_errors=True,  # ✅ This finally works
        verbose=True
    )

    raw_table_agent = create_pandas_dataframe_agent(
    llm=llm,
    df=df_1,
    verbose=True,
    allow_dangerous_code=True,
    number_of_head_rows=df_1.shape[0]
    )
    Table = AgentExecutor.from_agent_and_tools(
        agent=raw_table_agent.agent,
        tools=raw_table_agent.tools,
        handle_parsing_errors=True,
        verbose=True
    )
    
    # Excel
    raw_excel_agent = create_pandas_dataframe_agent(
        llm=llm,
        df=xls_data,
        verbose=True,
        allow_dangerous_code=True,
        number_of_head_rows=xls_data.shape[0]
    )
    Excel = AgentExecutor.from_agent_and_tools(
        agent=raw_excel_agent.agent,
        tools=raw_excel_agent.tools,
        handle_parsing_errors=True,
        verbose=True
    )
    
    # Snowflack
    raw_snow_agent = create_pandas_dataframe_agent(
        llm=llm,
        df=st.session_state.Data,

        verbose=True,
        allow_dangerous_code=True
    )
    Snowflack = AgentExecutor.from_agent_and_tools(
        agent=raw_snow_agent.agent,
        tools=raw_snow_agent.tools,
        handle_parsing_errors=True,
        verbose=True
    )


    # Initialize session state for chat history

    st.sidebar.title("💡 How to Chat with the Bot")
    st.sidebar.write("""
    ✅ Use **keywords** in your query to select the right data source:
    - **DAX**: Questions about calculations, formulas, or Power BI measures.
    - **TABLE**: Structure of Power BI tables.
    - **DATA**: Queries related to numbers and stored data.
    - **DICTIONARY**: Definitions of Power BI terms.
    - **GUIDE**: User Manual or User Guide.

    📌 **Example Queries:**
    - `"Give me the % of deals based on each stage from the data"` → Uses **Data**
    - `"Give me the calculation for TCV"` → Uses **Dax**
    - `"What does TCV means ?"` → Uses **Dictionary**
    - `"Give the table names present"` → Uses **Table**
    - `"Analyze the data and identify all possible reasons why the TCV in February 2024 is higher compared to other months. Consider factors such as team performance, deal size, client activity, or any noticeable trends or anomalies."` → Uses **Data**
    """)


    st.title("Power BI Smart Bot")

    # Chat response container
    response_container = st.container()
    with st.container(border=True):
        response_container = st.container(height=450)

        if st.button('Clear Chat'):
            st.session_state.messages = []

        # User input prompt
        if prompt := st.chat_input("Ask your question here"):
            st.session_state.messages.append({"role": "user", "content": prompt})

            # Build conversation history safely
            conversation_history = "\n".join([
                msg.get("content", "") for msg in st.session_state.messages if isinstance(msg, dict)
            ])
            def clean_with_llm(e, llm):
                """Send the entire parsing error to the LLM and ask it to clean it."""
                error_text = str(e)
                cleaning_prompt = (
                                    "You are given a raw LLM output that failed to parse correctly.\n"
                                    "Your task is to clean and extract the correct human-readable answer **only** from this output.\n\n"
                                    "📌 Do not reference or include any part of the original conversation history.\n"
                                    "📌 Only use the information present in the malformed LLM output provided below.\n"
                                    "📌 Answer only the **latest user question** that this output was attempting to respond to.\n"
                                    "📌 Remove all parsing errors, stack traces, or formatting issues.\n\n"
                                    "🔽 Malformed LLM Output:\n"
                                    f"{error_text}\n\n"
                                    "✅ Final Answer (Cleaned and user-friendly):"
                                )

                cleaned = llm.invoke(cleaning_prompt)
                return getattr(cleaned, "content", str(cleaned))
            
            agent_map = {
                        "DAX": DAX,
                        "POWER BI TABLE": Table,
                        "Snowflack": Snowflack,
                        "Excel": Excel,
                        "DICTIONARY": Excel,
                        "TABLE": Table,
                        "DATA": Snowflack,
                        "MEASURE": DAX,
                        "CALCULATION": DAX,
                        "NUMBER": Snowflack,
                        "DEFINITION": Excel,
                        "MEANING": Excel,
                        "None": llm
                    }
            selected_agent = catergory(query = prompt , query_history= conversation_history , LLM=llm)
            try:
                agent_tool = agent_map.get(selected_agent, llm)

                if agent_tool:
                    raw_response = agent_tool.invoke(conversation_history)
                else:
                    raw_response = llm.invoke(conversation_history)
                    raw_response = getattr(raw_response, "content", str(raw_response))

                # Always pass through the LLM cleaner
                response_content = clean_with_llm(raw_response, llm)

                # Fallback cleanup
                if isinstance(response_content, dict):
                    response_content = response_content.get("output", "")

                if not str(response_content).strip():
                    response_content = "⚠️ No valid response received."

            except Exception as e:
                # If any unexpected error
                response_content = f"❌ Error occurred: {str(e)}"

            # Store final response
            st.session_state.messages.append({"role": "assistant", "content": response_content})


        # Display chat messages
        with response_container:
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(f'<div style="font-size: small;">{message["content"]}</div>', unsafe_allow_html=True)

with st.container(border=True):
    power_bi_url = "https://app.powerbi.com/reportEmbed?reportId=b6437c22-5b36-4b31-8a98-7b892c5a6511&autoAuth=true&ctid=b1aae949-a5ef-4815-b7af-f7c4aa546b28"
    st.markdown(
        """
        <style>
        /* Remove default Streamlit padding */
        [data-testid="stAppViewContainer"] {
            padding: 0 !important;
        }

        /* Make the iframe cover full width */
        iframe {
            height: 95vh !important; /* Covers 95% of the viewport height */
            width: 100% !important; /* Covers full width */
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

        # Embed Power BI Report using HTML
    st.markdown(
        f"""
        <iframe title="Power BI Report" width="100%" height="95vh" 
        src="{power_bi_url}" frameborder="0" allowFullScreen="true"></iframe>
        """,
        unsafe_allow_html=True,
    )

st.title("Image analyzer")
st.write("Upload an image of a dashboard to analyze its contents.")
image_path = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"], key="image_uploader")

if image_path:
    image = Image.open(image_path)
    st.image(image, caption="📸 Uploaded Dashboard Image", use_column_width=True)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
        image.save(tmp_file.name)
        image_path = tmp_file.name
        reader = easyocr.Reader(['en'])
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


        ocr_results = reader.readtext(gray)
        extracted_text = [text[1] for text in ocr_results]
        extracted_numbers = [word for text in extracted_text for word in text.split() if word.replace(",", "").replace(".", "").isdigit()]

        # Detect active filter (Assuming a placeholder function detect_active_filter)
        #active_filter = detect_active_filter(image_path)

        description = (f"Dashboard contains charts/tables. "
                        f"Detected text: {'; '.join(extracted_text)}. "
                        f"Numbers: {', '.join(extracted_numbers)}. ")
                        #f"Active filter: {active_filter}.")
        
        system_message = SystemMessage(content="You are a skilled Image analyst.")

# Define the user's prompt
        user_message = HumanMessage(content=
        f"""You're analyzing a dashboard based on the visual input. Summarize the dashboard content in a detailed yet user-friendly way, using only the information visible in the dashboard. Avoid technical or backend details—assume the user can only see what's presented on the screen.
Here         is the extracted text from the dashboard: {description}""")

        st.write("Analyzing the image...")
        response = llm.invoke([system_message, user_message])
        st.write(response.content)
