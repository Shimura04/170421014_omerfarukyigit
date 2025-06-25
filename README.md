# 🏥 Tıbbi Asistan Chatbot - Proje Ödevi

## 📋 Proje Bilgileri

**Üniversite:** Marmara Üniversitesi  
**Bölüm:** Bilgisayar Mühendisliği  
**Dönem:** 2025 Bahar Dönemi  
**Öğrenci:** Semih Semerci  
**Öğrenci No:** 170422824  
**Proje Türü:** Dönem Sonu Proje Ödevi  

---

## 🎯 Proje Açıklaması

Bu proje, Türkçe tıbbi sorulara yanıt verebilen, intent sınıflandırması yapabilen ve güvenli bir şekilde sadece tıbbi konularla ilgilenen akıllı bir chatbot sistemidir. Sistem, kullanıcı sorularını analiz ederek tıbbi kategorilere ayırır ve uygun yanıtlar üretir.

## 🛠️ Kullanılan Teknolojiler

### **1. Machine Learning & AI**
- **SentenceTransformers** - Metin embeddings için
- **Scikit-learn** - Intent sınıflandırma (LogisticRegression)
- **LangChain** - LLM entegrasyonu ve prompt yönetimi
- **FAISS** - Vektör veritabanı (RAG sistemi için)

### **2. Large Language Models**
- **Google Gemini 1.5 Flash** - Cloud-based LLM
- **Llama 3.2 3B** - Local LLM (Ollama ile)
- **GoogleGenerativeAI Embeddings** - Doküman embeddings

### **3. Web Framework & UI**
- **Streamlit** - Web arayüzü ve kullanıcı etkileşimi
- **Pandas** - Veri işleme ve analiz
- **Matplotlib/Plotly** (gelecek geliştirmeler için)

### **4. Data Processing**
- **PyPDFLoader** - PDF doküman işleme
- **RecursiveCharacterTextSplitter** - Metin bölümleme
- **Excel/XLSX** - Dataset formatı

### **5. Model Evaluation**
- **Sklearn Metrics** - Accuracy, Precision, Recall, F1-Score
- **Classification Report** - Detaylı performans analizi
- **Confusion Matrix** - Sınıf karışıklık matrisi

### **6. Development Tools**
- **Python 3.13** - Ana programlama dili
- **dotenv** - Çevre değişkenleri yönetimi
- **Joblib** - Model serileştirme
- **JSON** - Sonuç kaydetme formatı

---

## 📊 Dataset ve Model Detayları

### **Dataset Özellikleri**
- **Dosya:** `medibot_dataset_complete.xlsx`
- **Toplam Örnek:** 1,590 adet
- **Kategori Sayısı:** 21 adet (18 tıbbi + 3 genel)
- **Dil:** Türkçe
- **Dengelenmiş:** Her kategoriden eşit örnek (~75 örnek/kategori)

### **Dataset Yapısı**
Dataset iki ana sütundan oluşur:

| Sütun Adı | Açıklama | Örnek |
|-----------|----------|-------|
| `user_message` | Kullanıcı sorusu | "Başım ağrıyor ne yapmalıyım?" |
| `balanced_intent` | İntent kategorisi | "pain_management" |

**🔍 Veri Kalitesi:**
- ✅ **Temizlenmiş:** Boş değerler kaldırıldı
- ✅ **Dengeli:** Her kategoriden eşit sayıda örnek
- ✅ **Çeşitli:** Farklı soru formatları ve cümleler
- ✅ **Gerçekçi:** Hastalardan gelebilecek sorular

### **Dataset Oluşturma Süreci**

#### **📝 Veri Toplama:**
1. **Tıbbi Forum Verileri** - Sağlık forumlarından sorular
2. **Hasta Diyalogları** - Gerçek hasta-doktor görüşmeleri  
3. **Tıbbi Kaynak Kitaplar** - Semptom ve hastalık açıklamaları
4. **Yapay Veri Artırımı** - Mevcut verilerin genişletilmesi

#### **🔧 Veri İşleme:**
1. **Temizleme:** Boş ve hatalı kayıtların kaldırılması
2. **Normalizasyon:** Türkçe karakter ve format düzeltmeleri
3. **Kategorilendirme:** Uzman doktor onayıyla intent ataması
4. **Dengeleme:** Her kategoriden eşit sayıda örnek seçimi
5. **Validasyon:** Kalite kontrol ve doğrulama

#### **📊 İstatistiksel Bilgiler:**
- **Ortalama Soru Uzunluğu:** 8-12 kelime
- **En Uzun Soru:** 25 kelime
- **En Kısa Soru:** 3 kelime
- **Tekrar Oranı:** %0 (benzersiz sorular)

### **Intent Kategorileri**

#### **🏥 Tıbbi Kategoriler (18 adet):**
| Kategori | Türkçe Adı | Örnek Soru |
|----------|------------|------------|
| `cardiovascular` | Kalp-damar hastalıkları | "Kalp çarpıntım var" |
| `dermatology` | Deri hastalıkları | "Cildimdeki lekelerin sebebi?" |
| `digestive` | Sindirim sistemi | "Mide yanması için ne yapmalı?" |
| `emergency` | Acil durumlar | "Göğsümde şiddetli ağrı var" |
| `endocrine` | Hormon hastalıkları | "Şeker hastalığı belirtileri" |
| `general_consultation` | Genel danışmanlık | "Doktor önerisi istiyorum" |
| `geriatric` | Yaşlılık hastalıkları | "Yaşlılık problemleri" |
| `medication` | İlaç bilgileri | "Bu ilacın yan etkileri" |
| `mental_health` | Ruh sağlığı | "Depresyon belirtilerim var" |
| `neurology` | Sinir sistemi | "Baş ağrısı çok şiddetli" |
| `ophthalmology` | Göz hastalıkları | "Gözüm bulanık görüyor" |
| `orthopedic` | Kemik-eklem hastalıkları | "Dizim ağrıyor" |
| `otolaryngology` | KBB | "Kulağım tıkalı" |
| `pain_management` | Ağrı yönetimi | "Ağrı kesici önerisi" |
| `pediatric` | Çocuk hastalıkları | "Çocuğumun ateşi yüksek" |
| `respiratory` | Solunum yolu | "Öksürük geçmiyor" |
| `urology` | Üroloji | "İdrar yolu enfeksiyonu" |
| `womens_health` | Kadın sağlığı | "Regl düzensizliği" |

#### **💬 Genel Kategoriler (3 adet):**
| Kategori | Açıklama | Durum |
|----------|----------|-------|
| `greeting` | Selamlama mesajları | ✅ Kabul edilir |
| `farewell` | Vedalaşma mesajları | ✅ Kabul edilir |
| `non_medical` | Tıbbi olmayan sorular | ❌ Reddedilir |

---

## 🏗️ Sistem Mimarisi

### **1. Intent Classification Pipeline**
```
Kullanıcı Sorusu → SentenceTransformer Embedding → 
LogisticRegression → Intent + Confidence → 
Güvenlik Filtresi → Yanıt Üretimi
```

### **2. Güvenlik Katmanları**
- **Confidence Threshold:** 0.6 minimum güven skoru
- **Category Whitelist:** Sadece tıbbi kategoriler kabul
- **Non-medical Rejection:** Tıbbi olmayan sorular reddedilir
- **Emergency Detection:** Acil durum tespiti ve yönlendirme

### **3. Model Karşılaştırma Sistemi**
- Gemini vs Llama performans karşılaştırması
- Gerçek zamanlı yanıt kalitesi analizi
- JSON formatında sonuç kaydetme

---

## 📈 Model Performans Sonuçları

### **Intent Sınıflandırma Başarımı**
- **Accuracy:** 95.9%
- **Precision:** 96.3%
- **Recall:** 95.9%
- **F1-Score:** 95.7%
- **Test Örnekleri:** 147 adet

### **LLM Karşılaştırması**
| Model | Başarı Oranı | Ortalama Yanıt Uzunluğu | Maliyet |
|-------|---------------|-------------------------|---------|
| Llama 3.2 3B | %100 | 211 karakter | Ücretsiz |
| Gemini 1.5 Flash | %100 | 241 karakter | Ücretli |

**Detaylı performans analizi:** `model_performance_summary.md` dosyasında mevcuttur.

---

## 🚀 Kurulum ve Çalıştırma

### **1. Python Virtual Environment Oluşturma**
```bash
# Projeyi klonlayın veya indirin
cd chatbot

# Virtual environment oluşturun
python3 -m venv venv

# Virtual environment'ı aktifleştirin
# macOS/Linux:
source venv/bin/activate
# Windows:
# venv\Scripts\activate
```

### **2. Python Bağımlılıklarını Yükleme**
```bash
# Gerekli paketleri yükleyin
pip install -r requirements.txt

# Streamlit performansı için (opsiyonel)
pip install watchdog
```

### **3. Ollama Kurulumu (Llama için)**
```bash
# macOS
brew install ollama

# Ollama servisini başlatın
ollama serve

# Yeni terminal açıp Llama modelini indirin
ollama pull llama3.2:3b
```

### **4. API Keys (.env dosyası)**
```env
# .env dosyası oluşturun
GEMINI_API_KEY=your_gemini_api_key_here
```

### **5. Uygulamayı Başlatma**
```bash
# Streamlit uygulamasını çalıştırın
streamlit run app/streamlit_app.py

# Veya belirli port ile:
streamlit run app/streamlit_app.py --server.port 8501

# Tarayıcıda otomatik açılacak: http://localhost:8501
# Eğer otomatik açılmazsa manuel olarak bu URL'ye gidin
```

**🎯 Başarılı Başlatma Kontrolü:**
- Terminal'de "You can now view your Streamlit app in your browser" mesajı görülmeli
- URL: http://localhost:8501 çalışır durumda olmalı
- Sidebar'da "🏥 Tıbbi Asistan" başlığı görünmeli

### **6. Model Değerlendirmesi (Opsiyonel)**
```bash
# Performans testleri çalıştırın
python evaluate_models.py
```

### **7. Troubleshooting (Sorun Giderme)**

#### **Python 3.13 PyTorch Sorunu:**
```bash
# Apple Silicon Mac'te PyTorch hatası alırsanız:
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

#### **Ollama Bağlantı Sorunu:**
```bash
# Ollama servisinin çalıştığından emin olun:
ollama list
# Eğer çalışmıyorsa:
ollama serve
```

#### **Streamlit Port Sorunu:**
```bash
# Farklı port kullanın:
streamlit run app/streamlit_app.py --server.port 8502
```

#### **Streamlit Cache Sorunu (Eski Kod Gözüküyor):**
```bash
# Tüm Streamlit proseslerini öldürün:
pkill -f streamlit

# Cache'i temizleyin:
streamlit cache clear
rm -rf ~/.streamlit/

# Tarayıcı cache'ini temizleyin (Cmd+Shift+R veya Ctrl+Shift+R)
# Ardından yeniden başlatın:
streamlit run app/streamlit_app.py
```

#### **API Key Sorunu:**
- `.env` dosyasının proje kök dizininde olduğundan emin olun
- API key'in doğru formatta olduğunu kontrol edin
- Gemini API key için: https://makersuite.google.com/app/apikey

---

## 📁 Proje Dosya Yapısı

```
chatbot/
├── app/
│   └── streamlit_app.py          # Ana uygulama
├── data/
│   ├── medical_intent_classifier.joblib  # Eğitilmiş model
│   ├── medical_label_encoder.joblib      # Label encoder
│   └── model_evaluation_*.json           # Değerlendirme sonuçları
├── medibot_dataset_complete.xlsx         # Ana dataset
├── evaluate_models.py                    # Model değerlendirme scripti
├── model_performance_summary.md          # Performans raporu
├── requirements.txt                      # Python bağımlılıkları
├── .env                                  # API anahtarları
└── README.md                             # Bu dosya
```

---

## 🔬 Teknik Yenilikler

### **1. Hibrit Model Yaklaşımı**
- Local + Cloud LLM karşılaştırması
- Maliyet-performans optimizasyonu
- API bağımlılığını azaltma

### **2. Akıllı Intent Filtreleme**
- Multi-layered güvenlik sistemi
- Confidence-based rejection
- Medical category enforcement

### **3. Kapsamlı Değerlendirme Sistemi**
- Cross-validation ile model doğrulama
- Real-time response quality metrics
- JSON-based result tracking

---

## 📊 Başarı Metrikleri

- ✅ **%95.9 Intent Classification Accuracy**
- ✅ **%100 LLM Response Success Rate**
- ✅ **0 Error Rate** (her iki model için)
- ✅ **21 Farklı Tıbbi Kategori** desteği
- ✅ **Gerçek Zamanlı** performans karşılaştırması

---

## 🔮 Gelecek Geliştirmeler

1. **Voice Interface** - Sesli komut desteği
2. **Mobile App** - Mobil uygulama versiyonu  
3. **Multi-language** - İngilizce destek ekleme
4. **Advanced RAG** - Daha kapsamlı tıbbi doküman desteği
5. **Fine-tuning** - Türkçe tıbbi model özelleştirmesi

---

## 📞 İletişim

**Öğrenci:** Semih Semerci  
**Email:** semihsemerci@marun.edu.tr  
**Öğrenci No:** 170422824  
**Proje Tarihi:** Haziran 2025

---


---

**🎓 Marmara Üniversitesi Bilgisayar Mühendisliği - 2025 Bahar Dönemi** 