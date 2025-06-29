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
st.set_page_config(page_title="Ubuntu Help Chatbot", layout="wide")

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
    <div class="sidebar-title"> 💻 Ubuntu Helpdesk</div>
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
model_path = os.path.join(BASE_DIR, "..", "data", "intent_classifier.joblib")
encoder_path = os.path.join(BASE_DIR, "..", "data", "label_encoder.joblib")

if not os.path.exists(model_path):
    try:
        st.info("🔄  Model file not found, training a new model...")
        
        # Dataset dosyasının varlığını kontrol et
        dataset_path = os.path.join(BASE_DIR, "..", "ubuntu_chatbot_dataset.xlsx")
        if not os.path.exists(dataset_path):
            st.error(f"❌ Dataset file not found: {dataset_path}")
            st.stop()
        
        # Data klasörünü oluştur (yoksa)
        data_dir = os.path.join(BASE_DIR, "..", "data")
        os.makedirs(data_dir, exist_ok=True)
        
        # Tamamlanmış dataset'i yükle
        st.write(f"📂 Loading dataset: {dataset_path}")
        df = pd.read_excel(dataset_path)
        df = df.dropna()
        
        if df.empty:
            st.error("❌ Dataset is empty!")
            st.stop()
        
        # Sütun kontrolü
        required_columns = ['user_message', 'balanced_intent']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            st.error(f"❌ Missing columns on dataset: {missing_columns}")
            st.write("Existing Columns", list(df.columns))
            st.stop()
        
        # Dengeli veri setinde balanced_intent kullan
        st.write(f"📊 Dataset loaded: {len(df)} örnek")
        st.write(f"🏷️ Number of categories: {len(df['balanced_intent'].unique())} adet")
        
        # Dengeli intent'leri kullan
        df['simplified_intent'] = df['balanced_intent']
        
        # Intent dağılımını göster
        intent_counts = df['simplified_intent'].value_counts()
        st.write("📈 Dataset Distribution:")
        st.dataframe(intent_counts)

        # Embedding modeli yükle
        st.write("🤖 Preparing embedding model for training...")
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
            st.warning("⚠️ Main model could not be loaded, using alternative model...")
            embedding_model_training = SentenceTransformer(
                "sentence-transformers/all-MiniLM-L6-v2",
                device="cpu"
            )
        
        # Embedding oluştur
        st.write("🔧 Generating embeddings...")
        X = embedding_model_training.encode(df["user_message"].tolist())
        
        # Label encoder
        st.write("🏷️ Label encoding...")
        le = LabelEncoder()
        y = le.fit_transform(df["simplified_intent"])

        # Model Eğitim
        st.write("🏋️ Training the model...")
        clf = LogisticRegression(max_iter=1000, random_state=42, C=1.0, class_weight='balanced')
        clf.fit(X, y)
        
        # Model performansını test et
        from sklearn.model_selection import cross_val_score
        scores = cross_val_score(clf, X, y, cv=5)
        st.success(f"✅ Model trained! Accuracy: {scores.mean():.3f}")

        # Kaydet
        st.write("💾 Saving the model...")
        joblib.dump(clf, model_path)
        joblib.dump(le, encoder_path)
        st.success(f"✅ Model saved successfully!")
        st.success(f"📁 Model folder: {model_path}")
        st.success(f"📁 Encoder folder: {encoder_path}")
        
    except Exception as e:
        st.error(f"❌ Error during model training: {str(e)}")
        st.error(f"Error Type: {type(e).__name__}")
        import traceback
        st.code(traceback.format_exc())
        st.stop()
else:
    try:
        # Mevcut modeli yükle
        clf = joblib.load(model_path)
        le = joblib.load(encoder_path)
        st.success("🤖 Existing model loaded successfully!")
    except Exception as e:
        st.error(f"❌ Error while loading the model: {str(e)}")
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
        
        st.info(f"🤖 Embedding model is loading... (Device: {device})")
        
        # Türkçe destekli model - device belirtme
        model = SentenceTransformer(
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            device=device
        )
        st.success("✅ Embedding loaded successfully!")
        return model
        
    except Exception as e:
        st.warning(f"⚠️ Can't load the main model: {e}")
        st.info("🔄 Trying for alternate model...")
        
        # Fallback: Daha basit model
        try:
            model = SentenceTransformer(
                "sentence-transformers/all-MiniLM-L6-v2",
                device="cpu"  # CPU'da çalıştır
            )
            st.success("✅  Alternative embedding model loaded!")
            return model
        except Exception as e2:
            st.error(f"🚫 No embedding model could be loaded: {e2}")
            return None

# Global embedding model
embedding_model = load_embedding_model()

# Intent Tahmin Fonksiyonu (Güçlendirilmiş)
def predict_intent(text):
    if embedding_model is None:
        # Fallback: güvenilir değil
        return "not_related", 0.1
    
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
        return "not_related", 0.1

# LLM Modeller
llm_gemini = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash-latest",
    temperature=0.3,
    max_tokens=500,
    google_api_key="AIzaSyDuqyG98CCFaeoBo8zVfffyGR0XmQDyXtY",
    convert_system_message_to_human=True
)

# Llama Model (Local - Ollama)
llm_llama = ChatOllama(
    model="llama3.2:3b",
    temperature=0.3,
    num_predict=500,
)

# Prompt
system_prompt = (
    "You are an experienced technical assistant specializing in Ubuntu operating system support. You help users solve Ubuntu-related issues in a clear and user-friendly manner."

    "IMPORTANT GUIDELINES:"

    "1. Do NOT execute system commands or make changes directly — only explain steps clearly."
    "2. If the issue is critical (e.g., system crash or boot failure), advise the user to consult an experienced technician or support forum."
    "3. Do NOT recommend or install third-party scripts or software unless they are officially trusted."
    "4. Always keep your answers concise, beginner-friendly, and structured step-by-step."
    "5. When needed, explain terminal commands and their effects briefly."

    "Your role is to support users with Ubuntu desktop and server problems, including package issues, system settings, permissions, updates, and basic troubleshooting."
    "Content given: {context}"
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
    
        if "Llama" in model_choice:
            # Llama için sistem promptu dahil etme
            full_prompt = f"""You are an experienced technical assistant specializing in Ubuntu operating system support. You help users solve Ubuntu-related issues in a clear and user-friendly manner.

IMPORTANT GUIDELINES:

1. Do NOT execute system commands or make changes directly — only explain steps clearly.
2. If the issue is critical (e.g., system crash or boot failure), advise the user to consult an experienced technician or support forum.
3. Do NOT recommend or install third-party scripts or software unless they are officially trusted.
4. Always keep your answers concise, beginner-friendly, and structured step-by-step.
5. When needed, explain terminal commands and their effects briefly.

Question: {question}

Answer:"""
            response = llm.invoke(full_prompt)
            return response.content
        else:  # Gemini
            return llm.invoke(question).content

# Sidebar ayarları
with st.sidebar:
    st.header("⚙️ Settings")
    model_choice = st.selectbox("🤖 Model Selection:", [
        "Gemini-1.5-flash", 
        "Llama-3.2-3B (Local)",
        "Compare (Both)"
    ])
    
    # Model karşılaştırma özelliği
    if model_choice == "Compare (Both)":
        st.info("🔄 Gemini and Llama models are going to compare!")
    elif model_choice == "Llama-3.2-3B (Local)":
        st.success("🦙 Local Llama model is being use - No API key needed!")
    
    
    if st.button("🗑️ Clear History"):
        st.session_state.chat_history = []
        st.rerun()
    
    # Model karşılaştırması geçmişi
    if len(st.session_state.model_comparisons) > 0:
        st.markdown("---")
        st.subheader("📊 Model Comparison")
        st.write(f"Total Tomparison: {len(st.session_state.model_comparisons)}")
        
        if st.button("📈 Comparison Analysis"):
            st.session_state.show_comparison_analysis = True
        
        if st.button("🗑️ Clear the comparisons"):
            st.session_state.model_comparisons = []
            st.rerun()

# Ana başlık
st.title("💻 Ubuntu Help Chatbot")
st.markdown("*Ask anything about Ubuntu*")

# Geçmişi göster
if st.session_state.chat_history:
    st.markdown("## 💬 Chat History")
    for i, (q, a, m) in enumerate(st.session_state.chat_history):
        st.markdown(f"**Question {i+1}:** {q}")
        st.markdown(f"**Answer ({m}):** {a}")
        st.markdown("---")

# Kullanıcı input
user_input = st.text_input("💭 Ask your question:", placeholder="Example: How to update apt?")
st.button("📤 Send", use_container_width=True)

if user_input:
    with st.spinner("🤔 Thinking..."):
        # Intent tahmin et
        intent, confidence = predict_intent(user_input)
        
        # Sadece doğru kategoriler tanımla (greeting ve farewell hariç)
        ubuntu_categories = {
            "networking","terminal_usage","system_configuration","file_operations","package_management","user_management","process_management","disk_and_partition","security",'other'
        }
        
        # SIKI Confidence threshold kontrolü ve kategori filtreleme
        if confidence < 0.6 or intent == "not_related" or (intent not in ubuntu_categories and intent not in ['greeting', 'farewell']):
            answer = f"""
            I can only assist you with the following topics related to Ubuntu and Linux systems:

            🧠 My Expertise Areas:

            💻 General Ubuntu support (installations, updates, package issues)
            🔧 Terminal commands and usage (bash, apt, dpkg, etc.)
            📦 Software and package management
            🧱 System performance and monitoring
            🌐 Network configuration and troubleshooting
            👤 User and permission management
            🛡️ Security and firewall settings
            🎨 Desktop environment issues (GNOME, KDE, XFCE...)
            📁 File system and storage (mount, disk, partitions)
            🐧 Linux kernel and modules
            💡 Tips and best practices for Ubuntu users
            📥 Dual boot and virtualization issues
            📦 Snap, Flatpak, AppImage usage
            🧩 Troubleshooting installation errors
            ⚙️ System services and daemons (systemd, cron, etc.)

             Please ask a question related to one of the topics above.

            🔄 If you're unsure where to start, try asking something like:
            “How can I install a .deb file on Ubuntu?”
                or
            “Why is my Wi-Fi not working on Ubuntu 22.04?”


            """
            intent_display = f"Out of Scope - Intent: {intent} (Confidence: {confidence:.2f})"
        
        # Eğer intent konu dışı ise sabit cevap 
        elif intent == "greeting":
            greetings = [
                "Hello! How can I assist you with Ubuntu today? 🐧",
                "Hi there! Feel free to ask any Ubuntu-related questions. 💻",
                "Welcome! I'm here to help you with your Ubuntu system. 😊",
                "Greetings! Need help with commands or troubleshooting in Ubuntu? 🔧",
                "Hey! I’m your Ubuntu assistant bot. Ask me anything. 🤖"
            ]
            import random
            answer = random.choice(greetings)
        # Farewell (Vedalaşma) mesajları  
        elif intent == "farewell":
            farewells = [
                "See you later! Happy Ubuntu-ing! 🧡",
                "Goodbye! Feel free to return with more questions. 👋",
                "Take care! Wishing you smooth Linux experiences. 🐧",
                "Bye! I'm always here if you need Ubuntu support again. 💬",
                "Farewell! Don't forget to keep your system updated. 🔄"
            ]
            import random
            answer = random.choice(farewells)

        else:
            # Normal sorular - Model seçimine göre çalıştır
            if model_choice == "Comprasion (Both)":
                # Her iki modeli de çalıştır
                answers = {}
                models_to_test = ["Gemini-1.5-flash", "Llama-3.2-3B (Local)"]
                
                for model in models_to_test:
                    try:
                        current_answer = run_rag_chain(user_input, model)
                        
                        # Eğer çok kısa cevap geliyorsa, detaylandır
                        if len(current_answer) < 100:
                            enhanced_prompt = f"""
                            User's Ubuntu question: "{user_input}"
                            Detected category: {intent}

                            Provide a detailed, helpful, and user-friendly answer to this Ubuntu-related issue.
                            Include relevant terminal commands if necessary, explain key steps clearly,
                            and remind the user to take precautions such as backing up data before making major changes.
                            """
                            if "Llama" in model:
                                current_answer = llm_llama.invoke(enhanced_prompt).content
                            else:
                                current_answer = llm_gemini.invoke(enhanced_prompt).content
                        
                        # Güvenlik uyarısı ekle
                        current_answer += "\n\n⚠️ *Remember to back up your data before applying critical system changes. Use terminal commands carefully.*"
                        answers[model] = current_answer
                        
                    except Exception as e:
                        error_msg = str(e)
                        if "API key" in error_msg.lower() or "Unauthorized" in error_msg.lower():
                            answers[model] = f"❌ API Key Error: {model} invalid or missing API Key."
                        else:
                            answers[model] = f"❌ Error: {model} - {error_msg[:100]}"
                
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
                ## 🤖 **Gemini-1.5-flash Response:**
                {answers.get('Gemini-1.5-flash', 'Hata oluştu')}
                
                ---
                
                ## 🦙 **Llama-3.2-3B (Local) Response:**
                {answers.get('Llama-3.2-3B (Local)', 'Hata oluştu')}
                """
                
            else:
                # Tek model çalıştır
                try:
                    answer = run_rag_chain(user_input, model_choice)
                    
                    # Eğer çok kısa cevap geliyorsa, detaylandır
                    if len(answer) < 100:
                        enhanced_prompt = f"""
                            User's Ubuntu question: "{user_input}"
                            Detected category: {intent}
                            
                            Provide a detailed, helpful, and user-friendly answer to this Ubuntu-related issue.
                            Include relevant terminal commands if necessary, explain key steps clearly,
                            and remind the user to take precautions such as backing up data before making major changes.
                            """
                        if "Llama" in model_choice:
                            answer = llm_llama.invoke(enhanced_prompt).content
                        else:
                            answer = llm_gemini.invoke(enhanced_prompt).content
                    
                    # Güvenlik uyarısı ekle
                    answer += "\n\n⚠️ *Remember to back up your data before applying critical system changes. Use terminal commands carefully.*"
                    
                except Exception as e:
                    error_msg = str(e)
                    if "API key" in error_msg.lower() or "unauthorized" in error_msg.lower():
                        answer = f"""
                        The API key for the {model_choice} model is invalid or missing.
                        Please check your API key in the .env file.
    
                        Temporary solutions:
                        🔹 Select a different model
                        🔹 Renew your API key
                        """
                    else:
                        answer = f"""
                        ❌ **Technical Error**
                        
                        Error: {error_msg[:150]}
                        
                        My general advice:
                        🔹 If you have severe symptoms, please consult a doctor
                        🔹 In case of emergency, call your local emergency number (e.g., 112)
                        🔹 Detected category of your question: {intent}
                        """
        # Sonucu göster
        st.markdown(f"**❓ Question** {user_input}")
        st.markdown(f"**🤖 Answer ({model_choice}):** {answer}")
        
        # Intent bilgisini göster
        if confidence < 0.6 or intent == "not_related" or (intent not in ubuntu_categories and intent not in ['greeting', 'farewell']):
            st.markdown(f"**🏷️ Intent:** {intent_display}")
        else:
            st.markdown(f"**🏷️ Intent:** {intent} (Confidence: {confidence:.2f})")
        st.markdown("---")
        
        # Geçmişe ekle
        st.session_state.chat_history.append((user_input, answer, model_choice))

# Model Karşılaştırma Analizi
if st.session_state.get("show_comparison_analysis", False) and len(st.session_state.model_comparisons) > 0:
    st.markdown("## 📊 Model Cpmrasion Anaylsis")
    
    # DataFrame oluştur
    df_comparisons = pd.DataFrame(st.session_state.model_comparisons)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Comprasion", len(df_comparisons))
    
    with col2:
        avg_confidence = df_comparisons['confidence'].mean()
        st.metric("Average Confidence", f"{avg_confidence:.3f}")
    
    with col3:
        unique_intents = df_comparisons['intent'].nunique()
        st.metric("Different Intent", unique_intents)
    
    # Intent dağılımı
    st.subheader("🏷️ Intent Distribution")
    intent_counts = df_comparisons['intent'].value_counts()
    st.bar_chart(intent_counts)
    
    # Tablo görünümü
    st.subheader("📋 Comprasion Details")
    
    for i, row in df_comparisons.iterrows():
        with st.expander(f"Soru {i+1}: {row['question'][:60]}... ({row['intent']})"):
            st.write(f"**Intent:** {row['intent']} (Confidence: {row['confidence']:.3f})")
            st.write(f"**Time:** {row['timestamp']}")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 🤖 Gemini")
                st.write(row['gemini_answer'])
            
            with col2:
                st.markdown("### 🦙 Llama")
                st.write(row.get('llama_answer', 'No Data'))
    
    # Karşılaştırmayı JSON olarak kaydet
    if st.button("💾 Save the comprasion to JSON"):
        comparison_file = f"data/model_comparison_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json"
        df_comparisons.to_json(comparison_file, orient='records', indent=2, force_ascii=False)
        st.success(f"✅ Comprasion Saved: {comparison_file}")
    
    if st.button("❌ Close Analysis"):
        st.session_state.show_comparison_analysis = False
        st.rerun()

# Footer
st.markdown("""
---
<div style='text-align: center; color: #666; padding: 20px;'>
    <p>💻 Ubuntu Chatbot © 2025</p>
    <p><small>⚠️ Please backup your data before any process </small></p>
</div>
""", unsafe_allow_html=True)
