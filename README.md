## Carregamento do Dataset, Pré-processamento e Normalização refinados para português
* Normalização de texto:
    * Conversão para minúsculas.
    * Remoção de acentos (unidecode).
    * Substituição de caracteres especiais.
* Limpeza:
    * Remoção de URLs, menções (@usuario), hashtags, espaços, números, pontuações e outros acentos e emojis.
    * Remoção de Stopwords.
* Tokenização:
    * Segmentação em palavras ou subpalavras (wordtokenizer), especialmente útil em português por causa da morfologia rica.

## Random Forest + Feature Engineering Tradicional
* Bag-of-Words (BoW): baseline simples.
* TF-IDF: mais robusto para capturar relevância.
* N-grams: bigramas e trigramas são úteis em discurso de ódio (ex.: “vai morrer”, “seu lixo”):
    * Utilizado unigrama.
* Levantamento de erros na classificação.

## Regressão Logística
* Vetorizador híbrido: palavra (1–2) + caractere (3–5).
* Dois vetores são concatenados via FeatureUnion-like simples por Pipeline separado.

## BERTimbau (Transformer - variação do BERT)
* Usa transformers (Hugging Face).
* Estratégia: fine-tuning em BERTimbau-base.
* Entrada: texto cru (sem lematização/stemming, apenas limpeza mínima).
* Saída: classificação binária.
* Word embeddings pré-treinados:
    * Word2Vec, FastText (bom para português por lidar com morfologia e palavras raras).
* Contextual embeddings:
    * BERTimbau e GPT-like models ajustados para português.
* Fine-tuning de Transformers:
    * Ajuste fino em datasets anotados de discurso de ódio.
 
## Meta Llama3 8B Instruct
* Utiliza o PEFT LoRa.
 
## Observações
* Random Forest e Regressão Logística: baseline clássico, útil para entender ganhos relativos.
* BERTimbau: modelo contextual já pré-treinado em português, se adapta bem ao domínio devido ao ajuste via fine-tuning.
* LLaMA3 + LoRA: não utilizado apenas como um modelo genérico em prompting, foi adaptado ao OFFCOMBR, aprendendo padrões ofensivos específicos do português.
    * Com isso, o LLaMA3 ganha competitividade real contra o BERTimbau.
    * A comparação passa a ser mais justa, porque ambos (BERTimbau e LLaMA3) estão sendo fine-tunados no mesmo dataset.
    * A diferença é que, com LoRA, o LLaMA3 deixa de ser apenas um baseline de prompting e passa a ser um concorrente direto do BERTimbau.
    * Isso enriquece a análise, porque podemos ver quanto o fine-tuning eficiente (LoRA) melhora um LLM genérico em relação a um modelo já especializado em português (BERTimbau).

## Avaliação dos modelos
* Comparação de métricas.
* Curvas ROC e AUC.
* Curvas Precision-Recall (PRC).
* Barplot comparativo.
* Matriz de confusão
* Heatmap de correlação

# Visão geral do notebook
O notebook implementa um pipeline completo para detecção de discurso de ódio em português com o corpus OFFCOMBR (OffComBR2.arff e OffComBR3.arff), cobrindo: carregamento direto do GitHub, normalização de rótulos para binário (`label_int`), pré-processamento, exploração de TF-IDF (com visualização e redução opcional via SVD/UMAP), treinamento de quatro famílias de modelos (Random Forest, Regressão Logística, BERTimbau e LLaMA 3 8B com LoRA/quantização 4-bit), explicabilidade via SHAP (nível de token/feature) e avaliação consolidada (métricas, matrizes de confusão, curvas ROC/PR e análise de erros).

# Pré-processamento e preparação de dados
* Limpeza: lowercasing; remoção de URLs, menções, hashtags, dígitos e pontuação; normalização com *unidecode*; colapso de espaços.
* Tokenização (NLTK/PT), remoção de stopwords personalizadas e de tokens vazios/curtos; deduplicação de tokens por documento; geração opcional de n-grams (padrão: unigramas).
* Remoção de documentos vazios/duplicados após limpeza.
* *Splits* estratificados de treino/teste são persistidos para reprodutibilidade entre execuções e modelos.
* Estatísticas auxiliares: contagens de tokens, distribuição de sentenças e visualizações 2D de TF-IDF.

# Modelos e treinamento
**Random Forest (baseline clássico)**
* *Pipeline*: `TfidfVectorizer` (unigramas, vocabulário até 20k) → `RandomForestClassifier` com balanceamento.
* Saídas: métricas (accuracy, precisão/recall/F1 macro), relatório de classificação, exemplos mal previstos.

**Regressão Logística (baseline clássico)**
* *Pipeline*: `TfidfVectorizer híbrido` (n-grams para caracter(3-5) e para palavra(1-2)) → `LogisticRegression` com balanceamento.
* Saídas: métricas (accuracy, precisão/recall/F1 macro), relatório de classificação, exemplos mal previstos.

**BERTimbau (Transformer em PT)**
* *Checkpoint*: `neuralmind/bert-base-portuguese-cased` com cabeça de classificação binária.
* Treino com Hugging Face `Trainer` (épocas, *batch sizes* e *collator* configuráveis; GPU com FP16/BF16 quando disponível).
* Métricas macro, relatório de erros e matriz de confusão.

**LLaMA 3 8B + LoRA Instruct (quantização 4-bit)**
* *Checkpoint*: `meta-llama/Meta-Llama-3-8B-Instruct`.
* Quantização NF4 (bitsandbytes) + PEFT/LoRA (r=16) com *gradient checkpointing*; *datasets* HF tokenizados para *max_length* fixo.
* Treino/avaliação via `Trainer`, salvando *tokenizer* e *adapters*; relatório de métricas, matriz de confusão e curva ROC/AUC.

# Explicabilidade (SHAP)
* **RF**: `shap.TreeExplainer` com *background* do treino; ranking global de tokens por |SHAP| médio e gráficos locais (contribuições assinadas) por instância.
* **BERTimbau e LLaMA**: *wrapper* `predict_proba` + `shap.maskers.Text`; explicações por token (globais: |SHAP| médio e sinal; locais: contribuições por amostra), alinhadas visualmente ao RF.

# Avaliação consolidada e interpretação de erros
* Consolidação das probabilidades de cada modelo para calcular métricas macro (accuracy, precisão, recall, F1) e da classe positiva; geração de tabelas comparativas, *barplots* (accuracy/F1), curvas ROC e Precision–Recall, matrizes de confusão e *heatmaps* de correlação entre métricas.
* Rotina de análise de erros: seleciona amostras em que pelo menos um modelo errou e mostra, para cada modelo, as previsões e gráficos SHAP locais (TF-IDF para Random Forest e Regressão Logística; tokens para BERTimbau/LLaMA), facilitando a inspeção de divergências entre abordagens.

## Experimentos
* Experimento 1 - explicable_hatespeech_pt.ipynb (baseline completo).
* Experimento 2 - explicable_hatespeech_pt_refactored.ipynb (refatoração com melhorias de features).
* Experimento 3 - explicable_hatespeech_pt_refactored_test.ipynb (com acentos e spaCy).
