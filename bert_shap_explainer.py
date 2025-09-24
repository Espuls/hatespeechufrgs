import shap
import torch
import matplotlib.pyplot as plt

def explain_bert_with_shap(model, tokenizer, texts, output_path="shap_bert_summary.png", num_samples=50):
    sample_texts = texts[:num_samples]

    def predict_proba(texts):
        inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        return probs.numpy()

    explainer = shap.KernelExplainer(predict_proba, sample_texts)
    shap_values = explainer.shap_values(sample_texts, nsamples=100)

    # Salvar gráfico SHAP
    plt.figure()
    shap.summary_plot(shap_values[1], sample_texts, show=False)
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
