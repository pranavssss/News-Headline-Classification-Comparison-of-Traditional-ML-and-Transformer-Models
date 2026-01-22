# Abstract:

News headline classification is an important task in Natural Language Processing (NLP) that involves assigning
short text sequences to predefined categories. This project addresses a multiclass text classification problem using
the AG News dataset, which contains news headlines labeled into four main categories: World, Sports, Business,
and Science/Technology.

The study begins by establishing strong baseline models using traditional machine learning techniques, including
Logistic Regression and Linear Support Vector Machines (SVM) with TF-IDF feature representations. These
baseline approaches provide a meaningful reference point and demonstrate that classical models can achieve
competitive performance on short text data.

Building on these baselines, the project explores encoder-based transformer architectures, specifically
DistilBERT and RoBERTa, which use bidirectional contextual representations to capture semantic and syntactic
information in news headlines. The models are trained using standard supervised fine-tuning, where all model
parameters (model weights) are updated for the classification task.

To improve training efficiency and generalization, parameter-efficient fine-tuning (PEFT) is implemented using
Low-Rank Adaptation (LoRA). The Q+V (Query and Value) projection approach is used. The key projection
is excluded, as it increases the parameter count by approximately 30–40% while providing minimal or no
accuracy improvement for classification tasks.

In addition, dropout regularization is applied to both the classifier head and LoRA layers to reduce overfitting
and improve model stability. Hyperparameter optimization is performed by tuning learning rates, batch sizes,
number of epochs, and regularization settings to achieve optimal performance.

Experimental results show that RoBERTa with LoRA and dropout regularization delivers the highest accuracy
and macro F1 score compared to all other models, while training only ~1.4% of the full model parameters. The
results demonstrate that parameter-efficient fine-tuning not only reduces computational cost but also improves
classification performance, making it a practical and effective approach for real-world NLP problems.

-------------------------------------------------------------------------------------------------------------------
Keywords: TF-IDF, Logistic Regression, Support Vector Machine, Encoder Transformers, DistilBERT,
RoBERTa, PEFT, LoRA, Q+V Attention, Key, Dropout Regularization, Hyperparameter Optimization.
