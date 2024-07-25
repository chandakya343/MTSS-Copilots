import os
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

def load_knowledge_base(file_path):
    loader = TextLoader(file_path)
    data = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=20000,
        chunk_overlap=100,
        strip_whitespace=True,
        separators=["#####", "*****", "@@@@@", "\n\n\n\n\n\n", "\n\n\n\n\n", "\n\n\n\n", "\n\n\n", "\n\n", "\n", " ", ""],
    )
    
    docs_processed = text_splitter.split_documents(data)
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    return FAISS.from_documents(docs_processed, embeddings)

def get_system_template(role):
    templates = {
        "School_Administrators": """You have been asked a question by School Administrators who job role is to ensures there is an MTSS team to design the school-wide implementation process, progress monitoring protocols, and data collection procedures. 
                                Below are the context:\nThis is context1:\n{context1}\n\nThis is context2:\n{context2}\n\nThis is context3:\n{context3}\n\nThis is context4:\n{context4}""",
        "Clinical_Staff": """You have been asked a question by Clinical_Staff who job role is to ensures, Clinical_Staff is a key member of both the MTSS team and the
                    school culture. Their visibility and voice on this team and throughout the building
                    communicates the importance of mental health across all tiers of the MTSS.  They
                    function as not only a referral source for students and families with more intensive
                    needs, but also as a consultant for wellness efforts across the tiers. 
                    School mental health clinicians:
                    - attend all MTSS meetings
                    -cprepare and report on qualitative and quantitative progress data on students and families receiving intensive support
                    - facilitate problem solving, offering key insights on the impacts and potential root causes of internalizing and externalizing behaviors
                    - identify and collaborate on research-based intervention strategies implemented by school staff
                    - support problem-solving and mediation for educators 
                    - lead and plan professional development related to individual and systems level mental health and wellness strategies 
                    - facilitate small groups related to prevention, intervention, and postvention support
                    - continuously interact with and engage families at both the community and individual level
                    Below are the context:\nThis is context1:\n{context1}\n\nThis is context2:\n{context2}\n\nThis is context3:\n{context3}\n\nThis is context4:\n{context4}""",
        # Add other roles here...
    }
    return templates.get(role, "")

def mtss_model(user_query, role, knowledge_base, openai_api_key):
    os.environ['OPENAI_API_KEY'] = openai_api_key
    chat = ChatOpenAI(model='gpt-4-turbo', temperature=0)
    
    retrieved_docs = knowledge_base.similarity_search(query=user_query, k=4)
    context1, context2, context3, context4 = [doc.page_content for doc in retrieved_docs]
    
    system_template = get_system_template(role)
    system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
    
    human_template = user_query
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    
    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
    request = chat_prompt.format_prompt(context1=context1, context2=context2, context3=context3, context4=context4, question=user_query).to_messages()
    
    result = chat(request)
    return result.content


# Usage example:
if __name__ == "__main__":
    knowledge_base = load_knowledge_base('/workspaces/MTSS-Copilots/knowledge_base.txt')
    response = mtss_model("Your question here", "School_Administrators", knowledge_base, "your-openai-api-key")
    print(response)