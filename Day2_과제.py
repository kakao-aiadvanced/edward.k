import requests
from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
import openai

# URL 목록
urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/"
]

# 각 URL의 내용을 텍스트로 추출
def fetch_text_from_url(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    paragraphs = soup.find_all('p')
    text = " ".join([para.text for para in paragraphs])
    return text

# 데이터 소스 로드
documents = [fetch_text_from_url(url) for url in urls]

# RecursiveTextSplitter 설정
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=256,
    chunk_overlap=50,
    length_function=len,
    is_separator_regex=False,
)

# 문서를 분할
split_documents = [text_splitter.split_text(document) for document in documents]

# 분할된 문서를 하나의 리스트로 합침
flattened_documents = [chunk for doc in split_documents for chunk in doc]

# OpenAI API 키 설정
openai_api_key = 'YOUR_API_KEY'
embeddings = OpenAIEmbeddings(api_key=openai_api_key)

# 문서 청크를 임베딩으로 변환
document_embeddings = embeddings.embed_documents(flattened_documents)

# Chroma 벡터 스토어 초기화
vector_store = Chroma(collection_name="my_collection")

# 임베딩 저장
vector_store.add_texts(flattened_documents, document_embeddings)

# 사용자 쿼리를 받아 관련된 청크를 검색
def retrieve_documents(query, top_k=5):
    # 쿼리 임베딩
    query_embedding = embeddings.embed_query(query)
    # 벡터 스토어에서 유사한 문서 검색
    results = vector_store.similarity_search(query_embedding, k=top_k)
    return [result['text'] for result in results], [result['source'] for result in results]

# OpenAI API 키 설정
openai.api_key = openai_api_key

def generate_text(prompt):
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=300
    )
    return response.choices[0].text.strip()

def evaluate_relevance(query, chunk):
    prompt = f"""
    [INST] Evaluate the relevance of the following text chunk to the given query. Provide your answer in the format {{'relevance': 'yes' or 'relevance': 'no'}}.
    
    Query: {query}
    Chunk: {chunk}
    [/INST]
    """
    response = generate_text(prompt)
    return response

def debug_relevance(query, retrieved_chunks):
    for i, chunk in enumerate(retrieved_chunks):
        relevance = evaluate_relevance(query, chunk)
        if relevance != "{'relevance': 'yes'}":
            print(f"Debugging chunk {i+1}...")
            print(f"Chunk: {chunk}")
            print(f"Relevance: {relevance}\n")
        else:
            print(f"Chunk {i+1} is relevant.")
            print(f"Chunk: {chunk}")
            print(f"Relevance: {relevance}\n")

def generate_response(query, relevant_chunks):
    combined_context = " ".join(relevant_chunks)
    prompt = f"""
    [INST] Given the following context, generate a detailed response to the query.

    Query: {query}
    Context: {combined_context}
    [/INST]
    """
    return generate_text(prompt)

def evaluate_hallucination(query, response, relevant_chunks):
    combined_context = " ".join(relevant_chunks)
    prompt = f"""
    [INST] Evaluate the following response for hallucinations based on the given context. Provide your answer in the format {{'hallucination': 'yes' or 'hallucination': 'no'}}.
    
    Query: {query}
    Context: {combined_context}
    Response: {response}
    [/INST]
    """
    return generate_text(prompt)

# 사용자 쿼리
user_query = "agent memory"

# 관련된 청크 검색
related_chunks, sources = retrieve_documents(user_query)

# 검색된 청크 출력 및 디버깅
if not related_chunks:
    print("No relevant chunks found. Debugging needed.")
else:
    all_no = True
    relevant_chunks = []
    relevant_sources = []
    for chunk, source in zip(related_chunks, sources):
        relevance = evaluate_relevance(user_query, chunk)
        print(f"Chunk relevance: {relevance}")
        if relevance == "{'relevance': 'yes'}":
            all_no = False
            relevant_chunks.append(chunk)
            relevant_sources.append(source)
    
    if all_no:
        print("All chunks are marked as 'no'. Starting debug...")
        debug_relevance(user_query, related_chunks)
    else:
        while True:
            response = generate_response(user_query, relevant_chunks)
            print(f"Response: {response}")
            hallucination_evaluation = evaluate_hallucination(user_query, response, relevant_chunks)
            print(f"Hallucination Evaluation: {hallucination_evaluation}")
            if hallucination_evaluation == "{'hallucination': 'no'}":
                print(f"Final Response: {response}")
                print(f"Sources: {', '.join(relevant_sources)}")
                break
