
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
    """Prepare Models"""
    print("🤖 Preparing Models...")
    
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
    """Load the test"""
    print("📂 Test Dataset Loading...")
    
    df = pd.read_excel("ubuntu_chatbot_dataset.xlsx")
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
    print(f"✅ {len(test_df)} test example loaded")
    print(f"📊 {len(test_df['balanced_intent'].unique())} farklı intent")
    
    return test_df

def evaluate_intent_classification(test_df):
    """Rate Intent classification"""
    print("\n🧠 Rating Intent Classification...")
    
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
    """RateResponse Quality"""
    print(f"\n💬 Rating Response Quality ({sample_size} example)...")
    
    # Sadece related intent'leri al
    ubuntu_intents = ['generic_help', 'terminal_command', 'not_in_ubuntu']
    
    ubuntu_df = test_df[test_df['balanced_intent'].isin(ubuntu_intents)]
    if len(ubuntu_df) < sample_size:
        sample_size = len(ubuntu_df)
    
    test_samples = ubuntu_df.sample(n=sample_size, random_state=42)
    
    results = {
        "llama_responses": [],
        "gemini_responses": [],
        "questions": [],
        "true_intents": []
    }
    
    ubuntu_prompt_template = """You are an experienced technical assistant specializing in Ubuntu operating system support. You help users solve Ubuntu-related issues in a clear and user-friendly manner.

"IMPORTANT GUIDELINES:"

    "1. Do NOT execute system commands or make changes directly — only explain steps clearly."
    "2. If the issue is critical (e.g., system crash or boot failure), advise the user to consult an experienced technician or support forum."
    "3. Do NOT recommend or install third-party scripts or software unless they are officially trusted."
    "4. Always keep your answers concise, beginner-friendly, and structured step-by-step."
    "5. When needed, explain terminal commands and their effects briefly."

Soru: {question}

Reaponse:"""
    
    for i, row in test_samples.iterrows():
        question = row['user_message']
        true_intent = row['balanced_intent']
        
        print(f"🔄 {len(results['questions'])+1}/{sample_size}: {question[:50]}...")
        
        # Llama yanıtı
        try:
            llama_prompt = ubuntu_prompt_template.format(question=question)
            llama_response = llama_model.invoke(llama_prompt).content
            results["llama_responses"].append(llama_response)
        except Exception as e:
            print(f"❌ Llama Error: {e}")
            results["llama_responses"].append(f"HATA: {str(e)}")
        
        # Gemini yanıtı
        try:
            gemini_prompt = ubuntu_prompt_template.format(question=question)
            gemini_response = gemini_model.invoke(gemini_prompt).content
            results["gemini_responses"].append(gemini_response)
        except Exception as e:
            print(f"❌ Gemini Error: {e}")
            results["gemini_responses"].append(f"HATA: {str(e)}")
        
        results["questions"].append(question)
        results["true_intents"].append(true_intent)
        
        # Rate limiting için kısa bekleme
        time.sleep(0.5)
    
    print(f"✅ {len(results['questions'])}")
    return results

def calculate_response_metrics(response_results):
    """Calculate response metrics"""
    print("📊 Calculating response metrics...")
    
    llama_responses = response_results["llama_responses"]
    gemini_responses = response_results["gemini_responses"]
    
    # Basit metrikler
    # Basit metrikler
    if len(llama_responses) > 0:
        successful_responses = [r for r in llama_responses if not r.startswith("ERROR")]
        error_count = len(llama_responses) - len(successful_responses)

        llama_metrics = {
            "avg_length": np.mean([len(r) for r in successful_responses]) if successful_responses else 0.0,
            "error_count": error_count,
            "success_rate": len(successful_responses) / len(llama_responses)
        }
    else:
        llama_metrics = {
            "avg_length": 0.0,
            "error_count": 0,
            "success_rate": 0.0
        }
    
    if len(gemini_responses) > 0:
        successful_responses = [r for r in gemini_responses if not r.startswith("ERROR")]
        error_count = len(gemini_responses) - len(successful_responses)

        gemini_metrics = {
            "avg_length": np.mean([len(r) for r in successful_responses]) if successful_responses else 0.0,
            "error_count": error_count,
            "success_rate": len(successful_responses) / len(gemini_responses)
        }
    else:
        gemini_metrics = {
            "avg_length": 0.0,
            "error_count": 0,
            "success_rate": 0.0
        }
    
    return {
        "llama_metrics": llama_metrics,
        "gemini_metrics": gemini_metrics
    }

def save_results(intent_results, response_results, response_metrics):
    """Save responses as JSON"""
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
    
    print(f"💾 Responses Saved: {filename}")
    return filename

def main():
    print("🚀 Model Performance Evaluating Start!")
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
        print("📈 Result")
        print("="*60)
        print(f"🎯 Intent Classification Accuracy: {intent_results['accuracy']:.3f}")
        print(f"🎯 Intent Classification F1-Score: {intent_results['f1_score']:.3f}")
        print(f"💾 Detailed Results: {result_file}")
        print("✅ Evaluation Completed!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        print(traceback.format_exc())

if __name__ == "__main__":
    main() 