import shap
import matplotlib.pyplot as plt

def explain_with_shap(model, X_test, vectorizer, output_path="shap_rf_summary.png"):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # Salvar gráfico SHAP
    plt.figure()
    shap.summary_plot(shap_values[1], X_test, feature_names=vectorizer.get_feature_names_out(), show=False)
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()