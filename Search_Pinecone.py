import os
import pinecone
from googletrans import Translator
import torch
from sentence_transformers import SentenceTransformer

def SearchSimilarity(question):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2', device=device)
    translator = Translator()
    PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY') or '6098a2ca-e9d9-4d67-a60a-9d62336d5cb6'
    PINECONE_ENV = os.environ.get('PINECONE_ENVIRONMENT') or 'us-east-1-aws'
    pinecone.init(
        api_key=PINECONE_API_KEY,
        environment=PINECONE_ENV
    )
    index_name = 'ninajung'
    if index_name not in pinecone.list_indexes():
        pinecone.create_index(
            name=index_name,
            dimension=768,
            metric='cosine'
        )
    index = pinecone.GRPCIndex(index_name)
    question = translator.translate(question[:1500], to_lang='en')
    xq = model.encode(question.text).tolist()
    documents = index.query(xq, top_k=2, include_metadata=True,)
    return documents,question

