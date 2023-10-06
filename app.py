from dotenv import load_dotenv
import os
import re
import uvicorn


from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader
from langchain.chains.summarize import load_summarize_chain
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate


from genai.extensions.langchain import LangChainInterface
from ibm_watson_machine_learning.foundation_models import Model
from genai.schemas import GenerateParams
from genai.model import Credentials, Model
from langchain.prompts import PromptTemplate

from langchain.chains.question_answering import load_qa_chain

from fastapi import FastAPI

app = FastAPI(version="3.0.2")

load_dotenv()

chunk_size = 1000
chunk_overlap = 100

loader = PyPDFLoader("life-event-based-insurance.pdf")
data = loader.load()

print("Fetched " + str(len(data)) + " documents")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
docs = text_splitter.split_documents(data)

print("Split into " + str(len(docs)) + " chunks")

from InstructorEmbedding import INSTRUCTOR
embeddings = HuggingFaceInstructEmbeddings(
        model_name = "hkunlp/instructor-large",
        model_kwargs = {"device": "cpu"}
    )

db = FAISS.from_documents(docs, embeddings)

os.environ["GENAI_API"] = "https://workbench-api.res.ibm.com/v1/"




api_key = os.getenv("GENAI_KEY", None)  
api_url = os.getenv("GENAI_API", None)

query = "Can you suggest me a insurance policy"
resultrecommend = db.similarity_search(query, k=2)
print(resultrecommend)

api_key = os.getenv("GENAI_KEY", None) 
api_url = os.getenv("GENAI_API", None)
creds = Credentials(api_key, api_endpoint=api_url)

print("\n------------- Example (Model Talk)-------------\n")

params = GenerateParams(decoding_method="greedy", max_new_tokens=200, min_new_tokens=1, repetition_penalty=1.0)


langchain_model = LangChainInterface(model="meta-llama/llama-2-13b-chat", params=params, credentials=creds)

##Prompt Template

template_prefix = """You are a insurance policy recommender system, designed to offer precise insurance policies recommendations based on user-specific financial habits, life events and goals. 
Analyze the context provided and the detailed financial information of the user, and life evnets, bank policy document to make a personalized insurance recommendation, providing clear reasoning behind your choice. 
If uncertain, respond truthfully that you don't have enough information without attempting to generate a speculative answer..


context:{context}"""

user_info = """This is what we know about the user, and you can use this information to better tune your research:

User Information
User Profile: Emily Davis

Name: Emily Davis
Age: 32
Gender: Female
Marital Status: Married
Occupation: Marketing Manager
Monthly Income: $5,500
Spouse's Monthly Income: $4,000
Total Household Monthly Income: $9,500
Address: 456 Oak Street, Suburbia, USA
Email: emily.davis@email.com
Phone Number: (555) 123-4567
Life Event:

Recent Life Event: Emily recently gave birth to a baby girl named Lily.
Financial Snapshot:

Monthly Income (Household): $9,500
Monthly Expenses: $6,500
Savings: $20,000
Outstanding Loans: Mortgage ($180,000 remaining)
Insurance Coverage: Health insurance for the family.
Spending Habits:

Monthly Expenses Breakdown:
Mortgage: $1,500
Utilities: $200
Groceries: $400
Childcare: $800
Transportation: $300
Dining Out: $100
Entertainment: $100
Shopping: $300
Baby Supplies: $300
Goals and Preferences:

Family Protection: Emily's primary goal is to ensure the financial security and well-being of her family, especially her newborn daughter, Lily.
Education: She plans to save for Lily's future education expenses.
Savings: Emily values savings and wants to continue building her family's financial safety net.


"""

template_suffix = """Question: {question}
Your response:"""


COMBINED_PROMPT = template_prefix +'\n'+ user_info +'\n'+ template_suffix
print(COMBINED_PROMPT)

#output model

PROMPT = PromptTemplate(
    template=COMBINED_PROMPT, input_variables=["context", "question"])

chain_type_kwargs = {"prompt": PROMPT}
qa = RetrievalQA.from_chain_type(llm=langchain_model, 
    chain_type="stuff", 
    retriever=db.as_retriever(),
    return_source_documents=True, 
    chain_type_kwargs=chain_type_kwargs)

#query = "Can you suggest me insurance policy"
# result = qa({'query':query})
# result['result']
# print(result['result'])

def perform_qa(query):
    qa_result=qa({'query': query})['result']
    print('resultssssss', qa_result)
    return qa_result

@app.get("/recommendation/{query}")
async def get_qa_result(query: str):
    try:
        result_qa = perform_qa(query)
        return {"result_qa": result_qa}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/qa")
async def get_qa_result(query_body: dict):
    try:
        query = query_body.get("query")
        if not query:
            raise HTTPException(status_code=400, detail="Query parameter is missing in the request body.")
        
        result_qa = perform_qa(query)
        return {"result_qa": result_qa}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("app:app", host ='0.0.0.0', port=8080, log_level="info")