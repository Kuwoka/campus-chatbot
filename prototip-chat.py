import os
import glob
from dotenv import load_dotenv

# --- Dünkü importlarımız (hepsi 'langchain_community' veya 'langchain'den) ---
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

# --- 1. AŞAMA: KURULUM FONKSİYONU ---
# Neden bir fonksiyon? Çünkü bu "ağır" işi ana koddan ayırmak ve 
# program başlarken SADECE BİR KEZ çalıştırmak istiyoruz.
def setup_rag_chain():
    print("--- RAG Asistanı Kuruluyor (Bu işlem 1 kez yapılır) ---")
    
    # === AŞAMA 1 & 2: VERİ YÜKLEME VE PARÇALAMA ===
    print("Aşama 1/2: PDF'ler yükleniyor ve parçalanıyor...")
    data_folder_path = "data/"
    pdf_dosyalari = glob.glob(data_folder_path + "*.pdf")

    if not pdf_dosyalari:
        print(f"Hata: '{data_folder_path}' klasöründe PDF dosyası bulunamadı.")
        return None # Fonksiyondan çık

    all_documents = []
    for pdf_path in pdf_dosyalari:
        loader = PyPDFLoader(pdf_path)
        all_documents.extend(loader.load()) 

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(all_documents)
    print(f"Toplam {len(chunks)} adet metin parçası (chunk) oluşturuldu.")

    # === AŞAMA 3: EMBEDDING VE VEKTÖR VERİTABANI ===
    print("Aşama 3: Embedding modeli ve FAISS veritabanı oluşturuluyor...")
    model_name = "paraphrase-multilingual-MiniLM-L12-v2"
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    vector_store = FAISS.from_documents(chunks, embeddings)
    print("Vektör veritabanı hazır.")

    # === AŞAMA 4: GENERATION (LLM, PROMPT, ZİNCİR) ===
    print("Aşama 4: LLM (Gemini) ve RAG Zinciri hazırlanıyor...")
    load_dotenv()
    if not os.getenv("GOOGLE_API_KEY"):
        print("Hata: GOOGLE_API_KEY .env dosyasında bulunamadı.")
        return None

    try:
        # Dünkü kota sorununu çözmek için 'flash' modelini kullanıyoruz
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)
    except Exception as e:
        print(f"Hata: Gemini modeli yüklenemedi. API anahtarınız doğru mu? Hata: {e}")
        return None

    # Dünkü çözüm: k=10 ile daha fazla sonuç getiriyoruz
    retriever = vector_store.as_retriever(search_kwargs={"k": 10})

    # Dünkü çözüm: Prompt'taki 'question' yerine 'input' kullanıyoruz
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

    # Zincirleri oluştur
    document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    
    print("--- Kurulum Tamamlandı. Chatbot Hazır! ---")
    
    # Neden zinciri döndürüyoruz? 
    # Çünkü bu ağır işlemle oluşturulan 'retrieval_chain' objesini,
    # hızlı çalışan sohbet döngüsünde tekrar tekrar kullanacağız.
    return retrieval_chain

# --- 2. AŞAMA: SOHBET DÖNGÜSÜ FONKSİYONU ---
# Neden bir fonksiyon? Bu, kodun "sohbet" kısmını temizce ayırır.
# Neden 'chain' parametresi alıyor? 
# Çünkü o yavaşça hazırlanan zincire (yukarıdaki fonksiyondan gelen) ihtiyacı var.
def start_chat_loop(chain):
    while True:
        # 1. Kullanıcıdan soru al
        question = input("\nSoru (çıkmak için 'çık' yazın): ")
        
        # 2. Çıkış komutunu kontrol et
        if question.lower() in ["çık", "exit", "quit"]:
            print("Görüşmek üzere! Kapatılıyor...")
            break
        
        # 3. Zinciri çalıştır (İşte HIZLI olan kısım burası)
        # Neden sadece 'invoke'? Çünkü PDF'ler, embedding'ler, her şey 
        # 'chain' objesinin içinde çoktan hazır.
        print("Cevap düşünülüyor...")
        response = chain.invoke({"input": question})
        
        # 4. Cevabı yazdır
        print("\n--- RAG CEVABI ---")
        print(response["answer"])

        # 5. (İsteğe bağlı) Dünkü gibi kaynakları göster (Hata ayıklama için)
        print("\n--- KULLANILAN KAYNAKLAR ---")
        for doc in response["context"]:
            file_name = os.path.basename(doc.metadata.get('source', 'Bilinmeyen'))
            print(f"Kaynak: {file_name}, İçerik: {doc.page_content[:100]}...")

# --- ANA ÇALIŞTIRMA KISMI ---
# Neden 'if __name__ == "__main__":'?
# Bu, Python'da bir script'in doğrudan çalıştırıldığını (import edilmediğini)
# kontrol etmenin standart yoludur.
if __name__ == "__main__":
    # 1. Ağır Kurulumu SADECE BİR KEZ yap
    rag_chain = setup_rag_chain()
    
    # 2. Eğer kurulum başarılıysa (rag_chain 'None' değilse)
    if rag_chain:
        # 3. Hızlı Sohbet Döngüsünü başlat
        start_chat_loop(rag_chain)
    else:
        print("Kurulum sırasında bir hata oluştu. Program kapatılıyor.")