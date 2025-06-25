#!/usr/bin/env python3
"""
Model Performans Değerlendirme - Llama vs Gemini
Dataset'ten örnekleri test edip accuracy, precision, recall, F1-score hesaplar
"""

import os
import pandas as pd
import json
from datetime import datetime
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from langchain_ollama import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import time
import random

# .env dosyasını yükle
load_dotenv()

# App modülünden intent prediction fonksiyonunu import et
import sys
sys.path.append('app')
from streamlit_app import predict_intent

def setup_models():
    """Modelleri hazırla"""
    print("🤖 Modeller hazırlanıyor...")
    
    # Llama (Local)
    llama_model = ChatOllama(
        model="llama3.2:3b",
        temperature=0.3,
        num_predict=100,
    )
    
    # Gemini (API)
    gemini_model = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash-latest",
        temperature=0.3,
        max_tokens=100,
        google_api_key=os.getenv("GEMINI_API_KEY"),
        convert_system_message_to_human=True
    )
    
    return llama_model, gemini_model

def load_test_dataset(sample_size=100):
    """Dataset'ten test örnekleri yükle"""
    print("📂 Test dataset'i yükleniyor...")
    
    df = pd.read_excel("medibot_dataset_complete.xlsx")
    df = df.dropna()
    
    # Her intent'ten eşit sayıda örnek al
    test_samples = []
    intents = df['balanced_intent'].unique()
    samples_per_intent = max(1, sample_size // len(intents))
    
    for intent in intents:
        intent_samples = df[df['balanced_intent'] == intent].sample(
            n=min(samples_per_intent, len(df[df['balanced_intent'] == intent])),
            random_state=42
        )
        test_samples.append(intent_samples)
    
    test_df = pd.concat(test_samples, ignore_index=True)
    print(f"✅ {len(test_df)} test örneği yüklendi")
    print(f"📊 {len(test_df['balanced_intent'].unique())} farklı intent")
    
    return test_df

def evaluate_intent_classification(test_df):
    """Intent classification performansını değerlend"""
    print("\n🧠 Intent Classification değerlendiriliyor...")
    
    true_intents = test_df['balanced_intent'].tolist()
    predicted_intents = []
    confidences = []
    
    for i, row in test_df.iterrows():
        user_message = row['user_message']
        true_intent = row['balanced_intent']
        
        # Intent tahmin et
        pred_intent, confidence = predict_intent(user_message)
        predicted_intents.append(pred_intent)
        confidences.append(confidence)
        
        if i % 20 == 0:
            print(f"📝 {i+1}/{len(test_df)} işlendi...")
    
    # Metrikleri hesapla
    accuracy = accuracy_score(true_intents, predicted_intents)
    precision = precision_score(true_intents, predicted_intents, average='weighted', zero_division=0)
    recall = recall_score(true_intents, predicted_intents, average='weighted', zero_division=0)
    f1 = f1_score(true_intents, predicted_intents, average='weighted', zero_division=0)
    
    # Detaylı rapor
    report = classification_report(true_intents, predicted_intents, output_dict=True, zero_division=0)
    
    intent_results = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "average_confidence": float(np.mean(confidences)),
        "classification_report": report,
        "test_samples": len(test_df)
    }
    
    print(f"✅ Intent Classification - Accuracy: {accuracy:.3f}, F1: {f1:.3f}")
    return intent_results

def evaluate_response_quality(test_df, llama_model, gemini_model, sample_size=20):
    """Yanıt kalitesini değerlendir"""
    print(f"\n💬 Response Quality değerlendiriliyor ({sample_size} örnek)...")
    
    # Sadece tıbbi intent'leri al
    medical_intents = ['general_consultation', 'emergency', 'cardiovascular', 'respiratory', 
                      'neurology', 'pain_management', 'digestive', 'orthopedic']
    
    medical_df = test_df[test_df['balanced_intent'].isin(medical_intents)]
    if len(medical_df) < sample_size:
        sample_size = len(medical_df)
    
    test_samples = medical_df.sample(n=sample_size, random_state=42)
    
    results = {
        "llama_responses": [],
        "gemini_responses": [],
        "questions": [],
        "true_intents": []
    }
    
    medical_prompt_template = """Sen deneyimli bir tıbbi asistansın. Aşağıdaki sağlık sorusuna kısa ve net bir yanıt ver.

ÖNEMLİ:
- Kesin teşhis koymayın
- Genel tavsiyeler verin
- Acil durumlarda doktora başvurmayı söyleyin
- Maksimum 2-3 cümle

Soru: {question}

Yanıt:"""
    
    for i, row in test_samples.iterrows():
        question = row['user_message']
        true_intent = row['balanced_intent']
        
        print(f"🔄 {len(results['questions'])+1}/{sample_size}: {question[:50]}...")
        
        # Llama yanıtı
        try:
            llama_prompt = medical_prompt_template.format(question=question)
            llama_response = llama_model.invoke(llama_prompt).content
            results["llama_responses"].append(llama_response)
        except Exception as e:
            print(f"❌ Llama hatası: {e}")
            results["llama_responses"].append(f"HATA: {str(e)}")
        
        # Gemini yanıtı
        try:
            gemini_prompt = medical_prompt_template.format(question=question)
            gemini_response = gemini_model.invoke(gemini_prompt).content
            results["gemini_responses"].append(gemini_response)
        except Exception as e:
            print(f"❌ Gemini hatası: {e}")
            results["gemini_responses"].append(f"HATA: {str(e)}")
        
        results["questions"].append(question)
        results["true_intents"].append(true_intent)
        
        # Rate limiting için kısa bekleme
        time.sleep(0.5)
    
    print(f"✅ {len(results['questions'])} response karşılaştırması tamamlandı")
    return results

def calculate_response_metrics(response_results):
    """Yanıt metriklerini hesapla"""
    print("📊 Response metrikleri hesaplanıyor...")
    
    llama_responses = response_results["llama_responses"]
    gemini_responses = response_results["gemini_responses"]
    
    # Basit metrikler
    llama_metrics = {
        "avg_length": np.mean([len(r) for r in llama_responses if not r.startswith("HATA")]),
        "error_count": sum(1 for r in llama_responses if r.startswith("HATA")),
        "success_rate": sum(1 for r in llama_responses if not r.startswith("HATA")) / len(llama_responses)
    }
    
    gemini_metrics = {
        "avg_length": np.mean([len(r) for r in gemini_responses if not r.startswith("HATA")]),
        "error_count": sum(1 for r in gemini_responses if r.startswith("HATA")),
        "success_rate": sum(1 for r in gemini_responses if not r.startswith("HATA")) / len(gemini_responses)
    }
    
    return {
        "llama_metrics": llama_metrics,
        "gemini_metrics": gemini_metrics
    }

def save_results(intent_results, response_results, response_metrics):
    """Sonuçları JSON'a kaydet"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"data/model_evaluation_{timestamp}.json"
    
    # Sonuçları birleştir
    final_results = {
        "evaluation_info": {
            "timestamp": datetime.now().isoformat(),
            "models_tested": ["Llama-3.2-3B", "Gemini-1.5-flash"],
            "evaluation_type": "Intent Classification + Response Quality"
        },
        "intent_classification": intent_results,
        "response_quality": {
            "response_metrics": response_metrics,
            "sample_responses": response_results
        }
    }
    
    # Dosyayı kaydet
    os.makedirs("data", exist_ok=True)
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)
    
    print(f"💾 Sonuçlar kaydedildi: {filename}")
    return filename

def main():
    print("🚀 Model Performans Değerlendirmesi Başlıyor!")
    print("=" * 60)
    
    try:
        # 1. Dataset yükle
        test_df = load_test_dataset(sample_size=150)
        
        # 2. Intent classification değerlendir
        intent_results = evaluate_intent_classification(test_df)
        
        # 3. Modelleri hazırla
        llama_model, gemini_model = setup_models()
        
        # 4. Response quality değerlendir
        response_results = evaluate_response_quality(test_df, llama_model, gemini_model, sample_size=25)
        
        # 5. Response metrikleri hesapla
        response_metrics = calculate_response_metrics(response_results)
        
        # 6. Sonuçları kaydet
        result_file = save_results(intent_results, response_results, response_metrics)
        
        # 7. Özet göster
        print("\n" + "="*60)
        print("📈 SONUÇ ÖZETİ")
        print("="*60)
        print(f"🎯 Intent Classification Accuracy: {intent_results['accuracy']:.3f}")
        print(f"🎯 Intent Classification F1-Score: {intent_results['f1_score']:.3f}")
        print(f"🦙 Llama Success Rate: {response_metrics['llama_metrics']['success_rate']:.3f}")
        print(f"🤖 Gemini Success Rate: {response_metrics['gemini_metrics']['success_rate']:.3f}")
        print(f"💾 Detaylı sonuçlar: {result_file}")
        print("✅ Değerlendirme tamamlandı!")
        
    except Exception as e:
        print(f"❌ Hata oluştu: {e}")
        import traceback
        print(traceback.format_exc())

if __name__ == "__main__":
    main() 