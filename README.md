# Detecção de Discurso de Ódio com Aprendizado de Máquina Explicável
Este projeto tem como objetivo detectar discursos de ódio em língua portuguesa utilizando diferentes modelos de classificação: Random Forest com Bag of Words e BERTimbau. Ambos os modelos são interpretados com SHAP para explicar suas decisões.

# Estrutura do Projeto
hate_speech_ufrgs/
│
├── data_loader.py              # Carrega e vetorização do dataset OFFCOMBR
├── model_training.py           # Treinamento dos modelos Random Forest
├── evaluation.py               # Avaliação dos modelos com métricas completas
├── explainability.py           # SHAP para Random Forest
├── bert_baseline.py            # Treinamento do modelo BERTimbau
├── bert_shap_explainer.py      # SHAP para BERTimbau via KernelExplainer
├── report_generator.py         # Geração de relatório final
├── requirements.txt            # Dependências do projeto
└── main.py                     # Pipeline principal de execução

# Instalação
1. Clone o repositório: git clone
2. Instale as dependências: pip install -r requirements.txt

# Ordem de Execução
1. Executar o pipeline principal: python main.py
2. O script main.py:
   * Carregamento e vetorização do dataset
   * Treinamento do modelo Random Forest (80/20)
   * Avaliação com acurácia, precisão, recall e F1
   * Treinamento do modelo BERTimbau
   * Avaliação do BERTimbau
   * Aplicação do SHAP para ambos os modelos
   * Geração de relatório final com métricas e caminhos dos gráficos SHAP
  
# Resultados
* Os gráficos SHAP serão salvos como:
  * shap_rf_summary.png → Random Forest
  * shap_bert_summary.png → BERTimbau
* O relatório final será impresso no console com todas as métricas comparativas.

# Requisitos
* Python 3.13
* Compatível com PyTorch e HuggingFace Transformers
* Dataset: OFFCOMBR → https://huggingface.co/datasets/vpmoreira/offcombr
