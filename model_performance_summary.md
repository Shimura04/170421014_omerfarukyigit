# 🏥 Tıbbi Chatbot Model Performans Değerlendirmesi

## 📊 Değerlendirme Özeti

**Tarih:** 24 Haziran 2025  
**Test Örnekleri:** 147 adet (21 farklı intent)  
**Model Karşılaştırması:** Llama 3.2 3B vs Gemini 1.5 Flash  

---

## 🧠 Intent Sınıflandırma Performansı

### 🎯 Genel Metrikler
| Metrik | Değer | Açıklama |
|--------|-------|----------|
| **Accuracy** | **95.9%** | Toplam doğru tahmin oranı |
| **Precision** | **96.3%** | Pozitif tahminlerin doğruluk oranı |
| **Recall** | **95.9%** | Gerçek pozitif örneklerin yakalanma oranı |
| **F1-Score** | **95.7%** | Precision ve Recall'un harmonik ortalaması |
| **Ortalama Güven** | **75.3%** | Model tahminlerindeki ortalama güven skoru |

### 🏷️ Intent Kategorisi Performansı

#### ✅ Mükemmel Performans (F1=1.0)
- **dermatology** (Deri hastalıkları)
- **farewell** (Vedalaşma)
- **geriatric** (Yaşlılık hastalıkları)
- **greeting** (Selamlama)
- **medication** (İlaç bilgileri)
- **mental_health** (Ruh sağlığı)
- **non_medical** (Tıbbi olmayan)
- **ophthalmology** (Göz hastalıkları)
- **otolaryngology** (KBB)
- **pediatric** (Çocuk hastalıkları)
- **urology** (Üroloji)
- **womens_health** (Kadın sağlığı)

#### 🟡 Yüksek Performans (F1=0.92-0.93)
- **cardiovascular** (F1: 0.933)
- **digestive** (F1: 0.933)
- **endocrine** (F1: 0.933)
- **neurology** (F1: 0.923)
- **orthopedic** (F1: 0.923)
- **pain_management** (F1: 0.933)
- **respiratory** (F1: 0.933)

#### 🔴 Geliştirilebilir Alan
- **general_consultation** (F1: 0.727) - En düşük performans
- **emergency** (F1: 0.857) - Acil durum tespiti

---

## 💬 Yanıt Kalitesi Karşılaştırması

### 📏 Temel Metrikler
| Model | Ortalama Yanıt Uzunluğu | Hata Oranı | Başarı Oranı |
|-------|-------------------------|-------------|---------------|
| **🦙 Llama 3.2 3B** | 211 karakter | 0% | **100%** |
| **🤖 Gemini 1.5 Flash** | 241 karakter | 0% | **100%** |

### 🔍 Yanıt Kalitesi Analizi

#### 🦙 Llama 3.2 3B (Local)
**Güçlü Yanları:**
- ✅ %100 başarı oranı (hiç hata yok)
- ✅ Lokal çalışma (API key gerektirmez)
- ✅ Hızlı yanıt süresi
- ✅ Tıbbi terminolojiyi doğru kullanıyor

**Zayıf Yanları:**
- ⚠️ Bazen Türkçe-İngilizce karışımı (örn: "caused", "possible")
- ⚠️ Bazı cevaplarda gramer hataları
- ⚠️ Daha kısa ve basit yanıtlar

#### 🤖 Gemini 1.5 Flash (API)
**Güçlü Yanları:**
- ✅ %100 başarı oranı (hiç hata yok)
- ✅ Daha düzgün ve akıcı Türkçe
- ✅ Daha detaylı ve yapılandırılmış yanıtlar
- ✅ Daha profesyonel dil kullanımı

**Zayıf Yanları:**
- ⚠️ API key gerektirir (maliyetli)
- ⚠️ İnternet bağlantısı zorunlu
- ⚠️ Rate limiting kısıtlamaları

---

## 📈 Örnek Yanıt Karşılaştırmaları

### Örnek 1: Acil Durum
**Soru:** "Göğsümde şiddetli ağrı var nefes alamıyorum"

**🦙 Llama:** "Görgüs ağrısı ve nefes alamazlık gibi symptoms, acil durumlar için doktora başvurmayı gerektirir..."

**🤖 Gemini:** "Göğüs ağrısı ve nefes darlığı ciddi olabilir. Hemen 112'yi arayın veya acil servise gidin. Acil tıbbi yardım almanız gerekmektedir."

### Örnek 2: Trigeminal Nevralji
**Soru:** "Trigeminal neuralji yüz ağrılarım dayanılmaz"

**🦙 Llama:** "Trigeminal neuraljisi, facial nerve ile ilgilidir. Bu durumun neden olduğu ağrılar, her günün bir günü olabilir..."

**🤖 Gemini:** "Dayanılmaz yüz ağrınız için hemen bir doktora veya diş hekimine başvurmalısınız. Trigeminal nevralji şüphesiyle detaylı muayene ve uygun tedavi için randevu almanız önemlidir..."

---

## 🏆 Sonuç ve Öneriler

### 🥇 Intent Sınıflandırma: Mükemmel
- Model %95.9 accuracy ile çok başarılı
- Sadece `general_consultation` kategorisi geliştirilmeli
- Confidence threshold 0.6 optimal seviye

### 🥈 Model Karşılaştırması: Her İkisi de Başarılı

#### 🦙 Llama 3.2 3B - **Önerilen**
- ✅ **Ücretsiz ve lokal**
- ✅ Güvenilir performans
- ✅ API bağımlılığı yok
- ⚠️ Türkçe dil kalitesi geliştirilebilir

#### 🤖 Gemini 1.5 Flash - **Premium**
- ✅ **Daha iyi dil kalitesi**
- ✅ Daha profesyonel yanıtlar
- ⚠️ Maliyetli (API key)
- ⚠️ İnternet bağımlılığı

### 📋 Geliştirme Önerileri
1. **General Consultation** kategorisi için ek eğitim örnekleri
2. Llama için Türkçe dil modelini iyileştirme
3. Hybrid yaklaşım: Llama (ana), Gemini (yedek)
4. Acil durum tespiti için ek güvenlik katmanları

---

**💾 Detaylı veriler:** `data/model_evaluation_20250624_134848.json`  
**🔄 Tekrarlanabilirlik:** Random seed 42 kullanıldı 