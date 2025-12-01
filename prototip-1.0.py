import glob
import os
from dotenv import load_dotenv
import google.generativeai as genai

# 1. API Anahtarı
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("API Key bulunamadı! .env dosyanı kontrol et.")
genai.configure(api_key=api_key)

# 2. DOĞRU İMPORTLAR (Temiz Kurulum Sonrası)
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# Google (Beyin)
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

# Topluluk Paketleri (PDF, Veritabanı)
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader

# Ayrı Paket (Metin Bölücü)
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ANA PAKET (Zincirler)
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

# 1. Adım: Tüm PDF'leri yükle
data_folder_path = "data/"
pdf_dosyalari = glob.glob(data_folder_path + "*.pdf")

all_documents = []
for pdf_path in pdf_dosyalari:
    loader = PyPDFLoader(pdf_path)
    # PDF'i sayfa sayfa yükler
    all_documents.extend(loader.load()) 

print(f"Toplam {len(all_documents)} sayfa doküman yüklendi.")

# 2. Adım: Metinleri Parçala (Chunking)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # Her parçanın boyutu (karakter sayısı)
    chunk_overlap=200   # Parçalar arası kesişim (anlam bütünlüğü için)
)

# Yüklenen tüm sayfaları parçalara ayır
chunks = text_splitter.split_documents(all_documents)

#Embedding modelini yükledik
model_name = "sentence-transformers/all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(model_name=model_name)

#Vektör veritabanı oluşturuyoruz
#embeddings modelini kullanarak hepsini vektöre çevirir ve FAISS veritabanına kaydeder
print("Embeddingler oluşturuluyor ve vektör veritabanı indeksleniyor...")
vector_store = FAISS.from_documents(chunks, embeddings)
print("Veritabanı hazır.")

try:
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)
except Exception as e:
    print(f"Hata: Gemini modeli yüklenemedi. API anahtarını kontrol et. Hata detayı: {e}")
    # Hata alırsan burada dur
    exit()

#En alakalı 3 sonuç getir
retriever = vector_store.as_retriever(search_kwargs={"k": 10})

# 4. Prompt Şablonunu Oluştur
# Bu, LLM'e nasıl davranacağını söyleyen talimattır.
template = """
Sen bir üniversite asistan chatbotusun. 
Sana verilen 'Bağlam'ı kullanarak 'Soru'ya cevap ver. 
Eğer cevap 'Bağlam' içinde yoksa, "Bu konuda bilgim yok." de. 
Cevaplarını sadece verilen bağlama dayandır, dışarıdan bilgi ekleme.

Bağlam:
{context}

Soru:
{input}

Cevap:
"""

prompt = ChatPromptTemplate.from_template(template)

# 5. Zinciri Oluştur
# Bu, (LLM + Prompt) 'u birleştiren alt zincirdir.
document_chain = create_stuff_documents_chain(llm, prompt)

# Bu, (Retriever + document_chain) 'i birleştiren ana RAG zinciridir.
# Bu zincir, önce belgeleri bulur (retriever), sonra o belgeleri 
# kullanarak cevap üretir (document_chain).
retrieval_chain = create_retrieval_chain(retriever, document_chain)

print("RAG Zinciri hazır. Soru soruluyor...")

# 6. Zinciri Test Et!
question = "Yatay geçiş yapabilmek için genel not ortalamam ne olmalı ?"
response = retrieval_chain.invoke({"input": question})

print("\n--- RAG CEVABI ---")
print(response["answer"])

# İstersen hangi belgeleri kullandığına da bakabilirsin:
#print("\n--- KULLANILAN KAYNAKLAR ---")
for doc in response["context"]:
    #print(f"Kaynak: {doc.metadata['source']}, İçerik: {doc.page_content[:100]}...")
    print(f"Kaynak: {doc.metadata['source']}")
    print(f"İçerik: {doc.page_content}")
    print("----------------------------")
