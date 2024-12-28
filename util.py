import os
from langchain.schema import Document


## UTIL ##
def load_documents(folder_path: str) -> list[Document]:
    documents = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):  # Process only .txt files
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                # Create a Document object with content and metadata
                documents.append(Document(page_content=content, metadata={"source": filename}))
    return documents

def pretty_print(step_nr: int, meta_info:str, info) -> None:
    if isinstance(info, list):
        print(f"[Step {step_nr}: {meta_info}]")
        for item in info:
            print(f"{item}\n")
    else:
        print(f"[Step {step_nr}: {meta_info}]\n{info}\n\n")
