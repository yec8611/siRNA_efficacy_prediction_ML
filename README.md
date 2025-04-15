# siRNA_efficacy_prediction_ML

This repository aims to establish advanced machine learning models for predicting siRNA efficacy.

To train any ML model, we need to have training data. This study utilizes a small dataset from [Huesken et al 2005](https://www.nature.com/articles/nbt1118), where over 2000 siRNAs were tested for inhibition efficiency. We framed this siRNA efficacy prediction task as either a regression problem (gene expression inhibtion 0-1), or a classification problem (1 for efficient inhibtion, 0 for inefficient inhibition). We used the [RNA-FM model](https://arxiv.org/abs/2204.00300) to generate rich sequence representation and trained a prediction head with diverse architectures. Further, we adopted a parameter efficient fine tuning (PEFT) strategy to enhance model prediction capability. We also tested a static embedding method where the embedded squence features were used for classical machine learning. Final comparison highlights the supurior performance of PEFT of RNA-FM on siRNA efficacy prediction.

## Train a prediction head (regressor) while freezing model parameters
We tested three architectures with different complexity (simple, medium, deep), and trained using 5 fold cross validation. The medium head structure performs better than simple head in terms of R2, and more complexity (deep head) did not improve performance.

![head comparison 1](https://github.com/user-attachments/assets/22896007-6011-49b1-8341-826bc0e5dea7)

We move forward with the medium structure (a small MLP) and used grid search to identify the best hyperparameters (learning rate, weight decay). The best set of hyperparamters (set 0) led to significant improvement over baseline (set 5).

![image](https://github.com/user-attachments/assets/dcb64f6f-5067-43b1-bb52-9f725f57a488)

We retrained the medium head after hyperparameter tuning. However, we only observed a minor increase in R2, propably due to overfitting during CV.

![hyperparameter tuning](https://github.com/user-attachments/assets/0709d7d7-3fca-4701-be68-bd9713233ec3)

We then applied low rank adaptation (LoRA), one of the parameter efficient fine tuning strategies. This method yields great improvement (R2 of ~0.4, 2 folds over baseline) on siRNA efficacy prediction. We did not perform hyperparameter tuning for LoRA due to computation cost but this would be really interesting to explore.

![LORA regressor](https://github.com/user-attachments/assets/7dec162a-1580-4768-9635-c97e6bc40fce)

In addition, we extracted the sequence embeddings (both siRNA sequences and target mRNA sequences) from RNA-FM, and trained 6 classical machine learning models. The best performing one is SVR-RBF, producing an R2 of 0.2027, worse than the LoRA fine tuning model but better than just fine-tuning the head. These machine learning methods can potentially model complex interactions and non-linearities than a simple MLP which may also be limited by the pooling strategy (we used mean pooling here). However, LoRA can allow the pretrained representations to be adapted specifically for the siRNA prediction task, hence outperforming all other methods.

![regressor_classical_ML](https://github.com/user-attachments/assets/e80e68da-964f-48c9-8808-cd58f04c0df8)


## Train a classification head
We want to see whether we can train a model to tell us whether a siRNA sequence is good or bad. We used inhibition efficiency > 0.5 as the threshold (>0.5 -> good, or 1; <0.5 -> bad, or 0). Similarly, we tested three architecutres for training a classification head attached to RNA-FM with 5 fold cross validation. We found the medium head to be slightly better than the simple head in ROC_AUC, while the deep head is higher but the F1 score is unstable.

![image](https://github.com/user-attachments/assets/cf4b4c96-9115-4bc6-9977-50c7d92e0c7a)

Again, we testd LoRA fine-tuning on RNA-FM for this classification task. We found a significant improvement in prediction (F1 = 0.7544 vs 0.6215; ROC AUC = 0.8231 vs 0.6540), compared to fine-tuning the head (medium)

![image](https://github.com/user-attachments/assets/ef87a0b2-f415-45b8-a30c-9941dc7cd5e9)

Lastly, we applied classical machine learning on the joined embeddings of siRNA/mRNA sequences. The best performing model is SVC-RBF with F1 of 0.6366 and ROC AUC of 0.6911.

![image](https://github.com/user-attachments/assets/cdc4af2d-3f63-40d7-9bb6-e0d12f970fb2)

The mechanisms of performance difference among the three classification strategies are similarly discussed above. Of note, a ROC AUC of 0.8231 with LoRA is really impressive considering our model hadn't gone through hyperparameter tuning and extensive structure optimization. As a reference, [A recent paper](https://academic.oup.com/bioinformatics/article/40/10/btae577/7775419) utilized static RNA-FM embeddings in combination with sequence thermodynamics features resulted in a ROC AUC of 0.86, only slightly better than our approach with sequence-only input.


## Further optimization
Moving forward, we plan to enhance our siRNA efficacy prediction model from the following aspects:
1. Optimize the pooling strategy. We can try more sophisticated methods such as attention pooling to minimize information loss.
2. Optimize the prediction head structure: add additional attention layers, use 1D convolutional layers, or recurrent layers.
3. Once base structure is determined, perform hyperparameter tuning for LoRA
4. Try inputing RNA features outside of RNA-FM such as thermodynamics, GC ratio, etc., to provide comprehensive aspects for modeling
5. Test different foundation models, such as DNA-BERT2, NT, etc.
6. More broadly, consider cellular background such as trancriptome for bulding the final model. Utilize single cell RNA-seq data, or gene/cell embeddings from scRNA-seq foundation models (such as scGPT), in a cross-attention fusion fashion.
