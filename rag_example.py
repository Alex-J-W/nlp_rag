"""This is an example of a RAG pipeline used in a short presentation for the NLP course."""
from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate

from util import load_documents, pretty_print as _print

"""
OPENAI

Model                   | Dimensions
====================================
text-embedding-3-large 	|   3,072
text-embedding-3-small  |   1,536
text-embedding-ada-002  |   1,536

GPT4All
Model                   | Dimensions
====================================
all-MiniLM-L6-v2        |   384
"""

OPENAI_API_KEY = ""
USER_QUERY = "What is LangChainPro?"
DIMENSIONS = 384


## Models
llm = None
embeddings_model = None

# Step 0: Initiate LLM and embeddings model
if OPENAI_API_KEY:
    from langchain_openai import OpenAI
    from langchain_openai import OpenAIEmbeddings

    _print(0, "Get Models", "Using OpenAI API")
    llm = OpenAI(api_key=OPENAI_API_KEY, model="gpt-4o-mini")
    embeddings_model = OpenAIEmbeddings(api_key=OPENAI_API_KEY, model="text-embedding-3-small")
else:
    """This will download an run models on your own machine. This might be a nice thing to do,
    but be wary of file sizes and worse results than using OpenAi.
    
    You can download gguf models to be used with gpt4all from HuggingFace or from nomic (gpt4all) directly.
    Find default GPT4All models here: https://github.com/nomic-ai/gpt4all/blob/main/gpt4all-chat/metadata/models3.json
    """
    from langchain_community.llms.gpt4all import GPT4All
    from langchain_community.embeddings.gpt4all import GPT4AllEmbeddings

    # The orca is just the smallest model, you can also run Llama3 for better results
    _print(0, "Get Models", "Using local GPT4All models")
    llm_name = "orca-mini-3b-gguf2-q4_0.gguf"
    embeddings_name = "all-MiniLM-L6-v2.gguf2.f16.gguf"
    kwargs={'allow_download': 'True'}

    llm = GPT4All(model=llm_name, allow_download=True)
    embeddings_model = GPT4AllEmbeddings(model_name=embeddings_name, gpt4all_kwargs=kwargs)

# Step 1: Get documents
folder_path = 'documents'
docs = load_documents(folder_path)
_print(1, "Load documents", docs)

# Step 2: Create an vector store with the documents
vector_store = Chroma(embedding_function=embeddings_model)
vector_store.add_documents(docs)
_print(2, "Created vector store from documents", vector_store)

# Step 3: Query document store
"""similarity_search_with_score uses cosine distance"""
retrieved_docs = vector_store.similarity_search_with_score(query=USER_QUERY, k=3)
_print(2, "Similarity Search Response", retrieved_docs) 

# Step 4: Prompt template
template = """Answer the question based on the provided documents:
{context}

Question: {question}
Answer:"""
prompt = PromptTemplate(input_variables=["context", "question"], template=template)
_print(4, "Prompt template", template)

# Step 5: Retrieve relevant documents and create prompt
context = "\n\n".join([doc.page_content for doc, _ in retrieved_docs])
formatted_prompt = prompt.format(context=context, question=USER_QUERY)
_print(5, "Formatted prompt", formatted_prompt)

# Step 6: Send to model
response = llm.invoke(formatted_prompt)
_print(6, "Response with RAG", response)

response = llm.invoke(USER_QUERY)
_print(6, "Response from model without RAF", response)
