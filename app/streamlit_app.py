# Intent Sınıflandırma (Embedding + LogisticRegression)
import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import joblib

from dotenv import load_dotenv
import os

# LangChain + Gemini + Llama
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# .env dosyasını yükle
load_dotenv()

# Başlık
st.set_page_config(page_title="Tıbbi Asistan Chatbot", layout="wide")

# Sol üst köşe başlık 
st.sidebar.markdown("""
    <style>
    .sidebar-title {
        font-size: 20px;
        font-weight: bold;
        padding: 10px 0 5px 5px;
        color: #ffffff;
    }

    section[data-testid="stSidebar"]::before {
        content: " ";
        display: block;
        margin-bottom: 10px;
        border-bottom: 1px solid #444;
    }
    </style>
    <div class="sidebar-title">🏥 Tıbbi Asistan</div>
""", unsafe_allow_html=True)

# Session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "model_comparisons" not in st.session_state:
    st.session_state.model_comparisons = []

if "show_comparison_analysis" not in st.session_state:
    st.session_state.show_comparison_analysis = False

import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Intent Model ve Label Encoder Yükleme
model_path = os.path.join(BASE_DIR, "..", "data", "medical_intent_classifier.joblib")
encoder_path = os.path.join(BASE_DIR, "..", "data", "medical_label_encoder.joblib")

if not os.path.exists(model_path):
    try:
        st.info("🔄 Model dosyası bulunamadı, yeni model eğitiliyor...")
        
        # Dataset dosyasının varlığını kontrol et
        dataset_path = os.path.join(BASE_DIR, "..", "medibot_dataset_complete.xlsx")
        if not os.path.exists(dataset_path):
            st.error(f"❌ Dataset dosyası bulunamadı: {dataset_path}")
            st.stop()
        
        # Data klasörünü oluştur (yoksa)
        data_dir = os.path.join(BASE_DIR, "..", "data")
        os.makedirs(data_dir, exist_ok=True)
        
        # Tamamlanmış dataset'i yükle
        st.write(f"📂 Dataset yükleniyor: {dataset_path}")
        df = pd.read_excel(dataset_path)
        df = df.dropna()
        
        if df.empty:
            st.error("❌ Dataset boş!")
            st.stop()
        
        # Sütun kontrolü
        required_columns = ['user_message', 'balanced_intent']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            st.error(f"❌ Dataset'te eksik sütunlar: {missing_columns}")
            st.write("Mevcut sütunlar:", list(df.columns))
            st.stop()
        
        # Dengeli veri setinde balanced_intent kullan
        st.write(f"📊 Dataset yüklendi: {len(df)} örnek")
        st.write(f"🏷️ Kategori sayısı: {len(df['balanced_intent'].unique())} adet")
        
        # Dengeli intent'leri kullan
        df['simplified_intent'] = df['balanced_intent']
        
        # Intent dağılımını göster
        intent_counts = df['simplified_intent'].value_counts()
        st.write("📈 Dataset Dağılımı:")
        st.dataframe(intent_counts)

        # Embedding modeli yükle
        st.write("🤖 Model eğitimi için embedding modeli hazırlanıyor...")
        import torch
        
        # Device belirleme
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
        
        try:
            embedding_model_training = SentenceTransformer(
                "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                device=device
            )
        except:
            st.warning("⚠️ Ana model yüklenemedi, alternatif model kullanılıyor...")
            embedding_model_training = SentenceTransformer(
                "sentence-transformers/all-MiniLM-L6-v2",
                device="cpu"
            )
        
        # Embedding oluştur
        st.write("🔄 Embeddings oluşturuluyor...")
        X = embedding_model_training.encode(df["user_message"].tolist())
        
        # Label encoder
        st.write("🏷️ Label encoding...")
        le = LabelEncoder()
        y = le.fit_transform(df["simplified_intent"])

        # Model Eğitim
        st.write("⚙️ Model eğitiliyor...")
        clf = LogisticRegression(max_iter=1000, random_state=42, C=1.0, class_weight='balanced')
        clf.fit(X, y)
        
        # Model performansını test et
        from sklearn.model_selection import cross_val_score
        scores = cross_val_score(clf, X, y, cv=5)
        st.success(f"✅ Model eğitildi! Doğruluk: {scores.mean():.3f}")

        # Kaydet
        st.write("💾 Model kaydediliyor...")
        joblib.dump(clf, model_path)
        joblib.dump(le, encoder_path)
        st.success(f"✅ Model başarıyla kaydedildi!")
        st.success(f"📁 Model dosyası: {model_path}")
        st.success(f"📁 Encoder dosyası: {encoder_path}")
        
    except Exception as e:
        st.error(f"❌ Model eğitimi sırasında hata: {str(e)}")
        st.error(f"Hata tipi: {type(e).__name__}")
        import traceback
        st.code(traceback.format_exc())
        st.stop()
else:
    try:
        # Mevcut modeli yükle
        clf = joblib.load(model_path)
        le = joblib.load(encoder_path)
        st.success("✅ Mevcut model başarıyla yüklendi!")
    except Exception as e:
        st.error(f"❌ Model yüklenirken hata: {str(e)}")
        st.stop()

# Embedding modeli güvenli yükleme
@st.cache_resource
def load_embedding_model():
    import torch
    try:
        # Device belirleme (Apple Silicon Mac için)
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
        
        st.info(f"🤖 Embedding modeli yükleniyor... (Device: {device})")
        
        # Türkçe destekli model - device belirtme
        model = SentenceTransformer(
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            device=device
        )
        st.success("✅ Embedding modeli başarıyla yüklendi!")
        return model
        
    except Exception as e:
        st.warning(f"⚠️ Ana model yüklenemedi: {e}")
        st.info("🔄 Alternatif model deneniyor...")
        
        # Fallback: Daha basit model
        try:
            model = SentenceTransformer(
                "sentence-transformers/all-MiniLM-L6-v2",
                device="cpu"  # CPU'da çalıştır
            )
            st.success("✅ Alternatif embedding modeli yüklendi!")
            return model
        except Exception as e2:
            st.error(f"❌ Hiçbir embedding model yüklenemedi: {e2}")
            return None

# Global embedding model
embedding_model = load_embedding_model()

# Intent Tahmin Fonksiyonu (Güçlendirilmiş)
def predict_intent(text):
    if embedding_model is None:
        # Fallback: güvenilir değil
        return "non_medical", 0.1
    
    try:
        # Metni encode et
        vec = embedding_model.encode([text])
        
        # Tahmin yap
        pred = clf.predict(vec)
        probabilities = clf.predict_proba(vec)[0]
        confidence = probabilities.max()
        intent = le.inverse_transform(pred)[0]
        
        # En yüksek 2 probability'yi karşılaştır (belirsizlik kontrolü)
        sorted_probs = sorted(probabilities, reverse=True)
        if len(sorted_probs) > 1:
            uncertainty = sorted_probs[0] - sorted_probs[1]
            # Eğer en yüksek ile ikinci yüksek arasında az fark varsa güven azalt
            if uncertainty < 0.2:
                confidence *= 0.8
        
        return intent, confidence
        
    except Exception as e:
        print(f"Intent tahmin hatası: {e}")
        return "non_medical", 0.1

# Akıllı acil durum kontrolü - Dengeli kategorilere göre
def is_emergency(intent, user_message="", urgency_level="UNKNOWN"):
    # GERÇEK ACİL DURUMLAR (112'lik)
    true_emergency_keywords = [
        'nefes alamıyorum', 'göğsümde ağrı', 'bayılıyorum', 'bayıldım',
        'bilinç kaybı', 'felç geçirdim', 'kalbim duruyor', 'şiddetli göğüs ağrısı',
        'kalp krizi', 'nefes darlığı çok şiddetli', 'öldürücü ağrı',
        'intihar etmek', 'kendimi öldürmek', 'çok yüksek ateş 40',
        'şuur kaybı', 'komada', 'kanama durmuyor', 'çok fazla kan kaybı',
        'zehirlendim', 'overdoz', 'aşırı doz'
    ]
    
    # Sözcük kontrolü (Türkçe küçük harf)
    message_lower = user_message.lower()
    has_true_emergency = any(keyword in message_lower for keyword in true_emergency_keywords)
    
    # Sadece GERÇEK emergency intent'i VE kritik kelimeler varsa acil
    is_emergency_intent = intent == 'emergency'
    is_critical_urgency = urgency_level == "CRITICAL"
    
    # ÇOCUK DOKTORU durumları acil değil
    child_non_emergency = any(word in message_lower for word in [
        'diken battı', 'çocuğ', 'kaşıntı', 'morarma', 'küçük yara'
    ])
    
    # Gerçek acil: (Emergency intent VE kritik kelime) VEYA kritik urgency
    return (is_emergency_intent and has_true_emergency) or is_critical_urgency

# Tıbbi PDF için RAG (Eğer tıbbi dökümanınız varsa)
faiss_index_path = os.path.join(BASE_DIR, "..", "data", "medical_faiss_index")
embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=os.getenv("GEMINI_API_KEY"))  # Gemini embedding kullan

if os.path.exists(os.path.join(faiss_index_path, "index.faiss")):
    vectorstore = FAISS.load_local(
        folder_path=faiss_index_path,
        embeddings=embedding,
        allow_dangerous_deserialization=True
    )
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})
else:
    # Eğer tıbbi PDF'niz varsa buraya ekleyin
    pdf_path = os.path.join(BASE_DIR, "..", "data", "medical_guide.pdf")  # PDF dosya yolunuz
    if os.path.exists(pdf_path):
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=50)
        chunks = splitter.split_documents(docs)
        vectorstore = FAISS.from_documents(chunks, embedding)
        vectorstore.save_local(faiss_index_path)
        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    else:
        retriever = None

# LLM Modeller
llm_gemini = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash-latest",
    temperature=0.3,
    max_tokens=500,
    google_api_key=os.getenv("GEMINI_API_KEY"),
    convert_system_message_to_human=True
)

# Llama Model (Local - Ollama)
llm_llama = ChatOllama(
    model="llama3.2:3b",
    temperature=0.3,
    num_predict=500,
)

# Tıbbi Prompt
system_prompt = (
    "Sen deneyimli bir tıbbi asistansın. Kullanıcıların sağlık sorularına yardımcı oluyorsun. "
    "ÖNEMLİ UYARILAR:\n"
    "1. Kesin teşhis koymayın, sadece genel bilgi verin\n"
    "2. Acil durumlarda mutlaka doktora başvurmasını söyleyin\n"
    "3. İlaç önerisi yapmayın, sadece genel tavsiyelerde bulunun\n"
    "4. Cevaplarınızı kısa ve anlaşılır tutun\n\n"
    "Verilen içerik: {context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])

# Zincir
qa_chain_gemini = create_stuff_documents_chain(llm_gemini, prompt)

# RAG zinciri
def run_rag_chain(question, model_choice):
    # Model seçimine göre LLM belirle
    if "Llama" in model_choice:
        llm = llm_llama
        qa_chain = create_stuff_documents_chain(llm, prompt)
    else:  # Gemini
        llm = llm_gemini
        qa_chain = qa_chain_gemini
    
    if retriever:
        chain = create_retrieval_chain(retriever, qa_chain)
        result = chain.invoke({"input": question})
        return result.get("answer")
    else:
        # RAG olmadan direkt LLM
        if "Llama" in model_choice:
            # Llama için sistem promptu dahil etme
            full_prompt = f"""Sen deneyimli bir tıbbi asistansın. Kullanıcıların sağlık sorularına yardımcı oluyorsun.

ÖNEMLİ UYARILAR:
1. Kesin teşhis koymayın, sadece genel bilgi verin
2. Acil durumlarda mutlaka doktora başvurmasını söyleyin
3. İlaç önerisi yapmayın, sadece genel tavsiyelerde bulunun
4. Cevaplarınızı kısa ve anlaşılır tutun

Soru: {question}

Cevap:"""
            response = llm.invoke(full_prompt)
            return response.content
        else:  # Gemini
            return llm.invoke(question).content

# Sidebar ayarları
with st.sidebar:
    st.header("⚙️ Ayarlar")
    model_choice = st.selectbox("🤖 Model Seçimi:", [
        "Gemini-1.5-flash", 
        "Llama-3.2-3B (Local)",
        "Karşılaştırma (Her İkisi)"
    ])
    
    # Model karşılaştırma özelliği
    if model_choice == "Karşılaştırma (Her İkisi)":
        st.info("🔄 Gemini ve Llama modelleri karşılaştırılacak!")
    elif model_choice == "Llama-3.2-3B (Local)":
        st.success("🦙 Local Llama modeli kullanılıyor - API key gerekmez!")
    
    # Acil durum uyarısı
    st.error("""
    🚨 **ACİL DURUM UYARISI**
    Bu chatbot tıbbi tavsiye vermez!
    Acil durumlar için:
    📞 112 - Ambulans
    🏥 En yakın hastaneye gidin
    """)
    
    if st.button("🗑️ Geçmişi Temizle"):
        st.session_state.chat_history = []
        st.rerun()
    
    # Model karşılaştırması geçmişi
    if len(st.session_state.model_comparisons) > 0:
        st.markdown("---")
        st.subheader("📊 Model Karşılaştırması")
        st.write(f"Toplam karşılaştırma: {len(st.session_state.model_comparisons)}")
        
        if st.button("📈 Karşılaştırma Analizi"):
            st.session_state.show_comparison_analysis = True
        
        if st.button("🗑️ Karşılaştırmaları Temizle"):
            st.session_state.model_comparisons = []
            st.rerun()

# Ana başlık
st.title("🏥 Tıbbi Asistan Chatbot")
st.markdown("*Sağlık sorularınızda size yardımcı olmak için buradayım. Acil durumlar için mutlaka 112'yi arayın!*")

# Geçmişi göster
if st.session_state.chat_history:
    st.markdown("## 💬 Sohbet Geçmişi")
    for i, (q, a, m) in enumerate(st.session_state.chat_history):
        st.markdown(f"**Soru {i+1}:** {q}")
        st.markdown(f"**Yanıt ({m}):** {a}")
        st.markdown("---")

# Kullanıcı input
user_input = st.text_input("💭 Sağlık sorunuzu yazın:", placeholder="Örn: Başım ağrıyor ne yapmalıyım?")
st.button("📤 Gönder", use_container_width=True)

if user_input:
    with st.spinner("🤔 Düşünüyor..."):
        # Intent tahmin et
        intent, confidence = predict_intent(user_input)
        
        # Sadece tıbbi kategoriler tanımla (greeting ve farewell hariç)
        medical_categories = {
            'general_consultation', 'emergency', 'ophthalmology', 'digestive', 
            'orthopedic', 'otolaryngology', 'respiratory', 'pediatric', 
            'neurology', 'cardiovascular', 'urology', 'dermatology', 
            'geriatric', 'endocrine', 'mental_health', 'medication', 
            'womens_health', 'pain_management'
        }
        
        # SIKI Confidence threshold kontrolü ve kategori filtreleme
        # Sadece yüksek güvenle ve tıbbi kategorilerde cevap ver
        if confidence < 0.6 or intent == "non_medical" or (intent not in medical_categories and intent not in ['greeting', 'farewell']):
            answer = f"""
            🏥 **Üzgünüm, bu soruyu yanıtlayamıyorum.**
            
            Ben sadece aşağıdaki tıbbi konularda yardımcı olabilirim:
            
            **🩺 Tıbbi Uzmanlık Alanlarım:**
            🫀 Kardiyoloji (Kalp ve damar hastalıkları)
            🧠 Nöroloji (Sinir sistemi hastalıkları) 
            👁️ Göz hastalıkları (Oftalmoloji)
            🦴 Ortopedi (Kemik ve eklem hastalıkları)
            👂 Kulak-Burun-Boğaz hastalıkları
            🫁 Solunum yolu hastalıkları
            👶 Çocuk hastalıkları (Pediatri)
            🍽️ Sindirim sistemi hastalıkları
            🩺 Genel tıbbi danışmanlık
            💊 İlaç bilgileri ve kullanımı
            👩‍⚕️ Kadın sağlığı
            🧓 Geriatri (Yaşlılık hastalıkları)
            🏥 Acil durumlar
            🧴 Deri hastalıkları
            ⚖️ Hormon hastalıkları (Endokrin)
            🧠 Ruh sağlığı
            💉 Ağrı yönetimi
            🚽 Üroloji (İdrar yolu hastalıkları)
            
            **Lütfen yukarıdaki konulardan biriyle ilgili soru sorun.**
            
            🚨 **Acil durumda 112'yi arayın!**
            """
            intent_display = f"KONU DIŞI - Intent: {intent} (Güven: {confidence:.2f})"
        
        # Eğer intent konu dışı ise sabit cevap (dataset'teki gerçek non_medical intent'i)
        elif intent == "greeting":
            greetings = [
                "Merhaba! Size sağlık konularında nasıl yardımcı olabilirim? 🏥",
                "Selam! Sağlık sorunlarınızla ilgili sorularınızı bekliyorum. 😊",
                "İyi günler! Tıbbi konularda size nasıl destek olabilirim? 🩺",
                "Hoş geldiniz! Sağlığınızla ilgili merak ettiklerinizi sorabilirsiniz. 💙",
                "Merhaba! Ben tıbbi asistan chatbot'uyum. Size nasıl yardımcı olabilirim? 🤖"
            ]
            import random
            answer = random.choice(greetings)
        # Farewell (Vedalaşma) mesajları  
        elif intent == "farewell":
            farewells = [
                "Görüşürüz! Sağlıklı günler dilerim. 🌟",
                "Hoşça kalın! Kendinize iyi bakın. 💚",
                "Güle güle! Başka sorularınız olursa buradayım. 👋",
                "İyi günler! Sağlığınızda kalın. 🙏",
                "Elveda! Her zaman sağlık konularında yardımcı olmaya hazırım. 😊",
                "Sağlıcakla kalın! Tekrar görüşmek dileğiyle. 🌈"
            ]
            import random
            answer = random.choice(farewells)
        elif is_emergency(intent, user_input):
            answer = f"""
            🚨 **ACİL DURUM TESPİT EDİLDİ!**
            
            Lütfen derhal:
            📞 112'yi arayın
            🏥 En yakın hastaneye gidin
            
            Bu ciddi bir durum olabilir ve profesyonel tıbbi müdahale gerektirir.
            """
        else:
            # Normal tıbbi sorular - Model seçimine göre çalıştır
            if model_choice == "Karşılaştırma (Her İkisi)":
                # Her iki modeli de çalıştır
                answers = {}
                models_to_test = ["Gemini-1.5-flash", "Llama-3.2-3B (Local)"]
                
                for model in models_to_test:
                    try:
                        current_answer = run_rag_chain(user_input, model)
                        
                        # Eğer çok kısa cevap geliyorsa, detaylandır
                        if len(current_answer) < 100:
                            enhanced_prompt = f"""
                            Kullanıcının sağlık sorusu: "{user_input}"
                            Tespit edilen kategori: {intent}
                            
                            Bu sağlık sorusuna detaylı, faydalı ve empati dolu bir yanıt ver.
                            Genel tavsiyeler, dikkat edilmesi gerekenler ve ne zaman doktora başvurulması gerektiğini belirt.
                            """
                            if "Llama" in model:
                                current_answer = llm_llama.invoke(enhanced_prompt).content
                            else:
                                current_answer = llm_gemini.invoke(enhanced_prompt).content
                        
                        # Güvenlik uyarısı ekle
                        current_answer += "\n\n⚠️ *Bu bilgi genel amaçlıdır. Kesin teşhis için doktora başvurun.*"
                        answers[model] = current_answer
                        
                    except Exception as e:
                        error_msg = str(e)
                        if "API key" in error_msg.lower() or "unauthorized" in error_msg.lower():
                            answers[model] = f"❌ API Key Hatası: {model} için API anahtarı geçersiz veya eksik."
                        else:
                            answers[model] = f"❌ Hata: {model} - {error_msg[:100]}"
                
                # Karşılaştırma sonucunu kaydet
                comparison_data = {
                    "timestamp": pd.Timestamp.now(),
                    "question": user_input,
                    "intent": intent,
                    "confidence": confidence,
                    "gemini_answer": answers.get("Gemini-1.5-flash", "Hata"),
                    "llama_answer": answers.get("Llama-3.2-3B (Local)", "Hata")
                }
                st.session_state.model_comparisons.append(comparison_data)
                
                # Karşılaştırmalı yanıtı hazırla
                answer = f"""
                ## 🤖 **Gemini-1.5-flash Yanıtı:**
                {answers.get('Gemini-1.5-flash', 'Hata oluştu')}
                
                ---
                
                ## 🦙 **Llama-3.2-3B (Local) Yanıtı:**
                {answers.get('Llama-3.2-3B (Local)', 'Hata oluştu')}
                """
                
            else:
                # Tek model çalıştır
                try:
                    answer = run_rag_chain(user_input, model_choice)
                    
                    # Eğer çok kısa cevap geliyorsa, detaylandır
                    if len(answer) < 100:
                        enhanced_prompt = f"""
                        Kullanıcının sağlık sorusu: "{user_input}"
                        Tespit edilen kategori: {intent}
                        
                        Bu sağlık sorusuna detaylı, faydalı ve empati dolu bir yanıt ver.
                        Genel tavsiyeler, dikkat edilmesi gerekenler ve ne zaman doktora başvurulması gerektiğini belirt.
                        """
                        if "Llama" in model_choice:
                            answer = llm_llama.invoke(enhanced_prompt).content
                        else:
                            answer = llm_gemini.invoke(enhanced_prompt).content
                    
                    # Güvenlik uyarısı ekle
                    answer += "\n\n⚠️ *Bu bilgi genel amaçlıdır. Kesin teşhis için doktora başvurun.*"
                    
                except Exception as e:
                    error_msg = str(e)
                    if "API key" in error_msg.lower() or "unauthorized" in error_msg.lower():
                        answer = f"""
                        ❌ **API Key Hatası**
                        
                        {model_choice} modeli için API anahtarı geçersiz veya eksik.
                        Lütfen .env dosyasında API anahtarınızı kontrol edin.
                        
                        Geçici çözüm:
                        🔹 Farklı bir model seçin
                        🔹 API anahtarınızı yenileyin
                        """
                    else:
                        answer = f"""
                        ❌ **Teknik Hata**
                        
                        Hata: {error_msg[:150]}
                        
                        Genel tavsiyem:
                        🔹 Ciddi belirtileriniz varsa doktora başvurun
                        🔹 Acil durumda 112'yi arayın
                        🔹 Sorunuzun kategorisi: {intent}
                        """
        
        # Sonucu göster
        st.markdown(f"**❓ Soru:** {user_input}")
        st.markdown(f"**🤖 Yanıt ({model_choice}):** {answer}")
        
        # Intent bilgisini göster
        if confidence < 0.6 or intent == "non_medical" or (intent not in medical_categories and intent not in ['greeting', 'farewell']):
            st.markdown(f"**🏷️ Intent:** {intent_display}")
        else:
            st.markdown(f"**🏷️ Intent:** {intent} (Güven: {confidence:.2f})")
        st.markdown("---")
        
        # Geçmişe ekle
        st.session_state.chat_history.append((user_input, answer, model_choice))

# Model Karşılaştırma Analizi
if st.session_state.get("show_comparison_analysis", False) and len(st.session_state.model_comparisons) > 0:
    st.markdown("## 📊 Model Karşılaştırma Analizi")
    
    # DataFrame oluştur
    df_comparisons = pd.DataFrame(st.session_state.model_comparisons)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Toplam Karşılaştırma", len(df_comparisons))
    
    with col2:
        avg_confidence = df_comparisons['confidence'].mean()
        st.metric("Ortalama Güven", f"{avg_confidence:.3f}")
    
    with col3:
        unique_intents = df_comparisons['intent'].nunique()
        st.metric("Farklı Intent", unique_intents)
    
    # Intent dağılımı
    st.subheader("🏷️ Intent Dağılımı")
    intent_counts = df_comparisons['intent'].value_counts()
    st.bar_chart(intent_counts)
    
    # Tablo görünümü
    st.subheader("📋 Karşılaştırma Detayları")
    
    for i, row in df_comparisons.iterrows():
        with st.expander(f"Soru {i+1}: {row['question'][:60]}... ({row['intent']})"):
            st.write(f"**Intent:** {row['intent']} (Güven: {row['confidence']:.3f})")
            st.write(f"**Zaman:** {row['timestamp']}")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 🤖 Gemini")
                st.write(row['gemini_answer'])
            
            with col2:
                st.markdown("### 🦙 Llama")
                st.write(row.get('llama_answer', 'Veri yok'))
    
    # Karşılaştırmayı JSON olarak kaydet
    if st.button("💾 Karşılaştırmayı JSON'a Kaydet"):
        comparison_file = f"data/model_comparison_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json"
        df_comparisons.to_json(comparison_file, orient='records', indent=2, force_ascii=False)
        st.success(f"✅ Karşılaştırma kaydedildi: {comparison_file}")
    
    if st.button("❌ Analizi Kapat"):
        st.session_state.show_comparison_analysis = False
        st.rerun()

# Footer
st.markdown("""
---
<div style='text-align: center; color: #666; padding: 20px;'>
    <p>🏥 Tıbbi Asistan Chatbot © 2024</p>
    <p><small>⚠️ Bu chatbot tıbbi tavsiye vermez. Ciddi durumlar için doktora başvurun.</small></p>
</div>
""", unsafe_allow_html=True)
