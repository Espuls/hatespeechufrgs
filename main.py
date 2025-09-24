from datasets import load_dataset
from data_loader import load_and_vectorize_data
from model_training import train_random_forest_80_20
from evaluation import evaluate_model
from bert_baseline import train_bert_model
from bert_shap_explainer import explain_bert_with_shap
from report_generator import generate_report
from explainability import explain_with_shap
from transformers import BertTokenizer

def main():
    # Random Forest
    X, y, vectorizer = load_and_vectorize_data()
    rf_model, X_test, y_test = train_random_forest_80_20(X, y)
    report_rf, acc_rf, prec_rf, rec_rf, f1_rf = evaluate_model(rf_model, X_test, y_test)

    print("Random Forest:")
    print(report_rf)
    print(f"Acurácia: {acc_rf:.4f} | Precisão: {prec_rf:.4f} | Recall: {rec_rf:.4f} | F1: {f1_rf:.4f}")

    # SHAP para Random Forest
    explain_with_shap(rf_model, X_test, vectorizer, output_path="shap_rf_summary.png")

    # BERT/BERTimbau
    bert_model, bert_metrics = train_bert_model()
    print("\nBERT/BERTimbau:")
    print(f"Acurácia: {bert_metrics['eval_accuracy']:.4f} | Precisão: {bert_metrics['eval_precision']:.4f} | Recall: {bert_metrics['eval_recall']:.4f} | F1: {bert_metrics['eval_f1']:.4f}")

    # SHAP para BERT
    dataset = load_dataset("vpmoreira/offcombr", split="train")
    texts = dataset['text']
    tokenizer = BertTokenizer.from_pretrained("neuralmind/bert-base-portuguese-cased")
    explain_bert_with_shap(bert_model, tokenizer, texts, output_path="shap_bert_summary.png")

    # Relatório final
    metrics_rf = {
        "accuracy": acc_rf,
        "precision": prec_rf,
        "recall": rec_rf,
        "f1": f1_rf
    }
    metrics_bert = {
        "accuracy": bert_metrics['eval_accuracy'],
        "precision": bert_metrics['eval_precision'],
        "recall": bert_metrics['eval_recall'],
        "f1": bert_metrics['eval_f1']
    }

    generate_report(metrics_rf, metrics_bert, "shap_rf_summary.png", "shap_bert_summary.png")

if __name__ == "__main__":
    main()