# Improving-Credit-Card-Fraud-Detection-using-Generative-Adversarial-Networks
6501 Capstone Group 4 

Team Member: Hao Ning, Jun Ying

## Project Description 
Link to the data:  <a href="https://www.kaggle.com/mlg-ulb/creditcardfraud">Credit Card Fraud Detection</a>  

As credit card transactions have become the mainstream consumption pattern, the number of credit card frauds has increased dramatically. Even if users are finally compensated, they often spend a lot of time and even money in the process. Therefore, it becomes important to distinguish whether the transaction is fraud or not from the beginning. 

In this project, we will first use traditional machine learning algorithms for classification training. The data is extremely imbalanced, regular machine learning has limited performanc, therefore we use Generative Adversarial Networks (GAN) to generate more credit card fraud data to oversample the data set. We evaluate the model performance with statistical metrics such as precision, recall, f1 score, ROC AUC score.   

We find out that, Vanilla GAN slightly improved fraud detection since it only generate lw spectrum data. Improved GAN models showed better fraud detection performance
because gnerated data have wider range & better overlap with the original data. **WGAN_GP** and **BEGAN** perform best among all GANs.  

Using GAN as an oversampling strategy has great potential in credit card fraud detection and extremely imbalanced dataset

|               | Base     | GAN      | WGAN     |**WGAN_GP**| GAN + AE |**BEGAN**| BAGAN    |
|---------------|----------|----------|----------|----------|----------|----------|----------|
| Accuracy      | 0.999491 | 0.999491 | 0.999544 | 0.999596 | 0.999544 | 0.999596 | 0.999544 |
| Precision     | 0.855670 | 0.864583 | 0.867347 | 0.894737 | 0.882979 | 0.903226 | 0.867347 |
| Recall        | 0.846939 | 0.846939 | 0.867347 | 0.867347 | 0.846939 | 0.857143 | 0.867347 |
| F1 score      | 0.851282 | 0.855670 | 0.867347 | 0.880829 | 0.864583 | 0.879581 | 0.867347 |
| ROC AUC score | 0.923346 | 0.923355 | 0.933559 | 0.933586 | 0.923373 | 0.928492 | 0.933559 |

## Project Structure
This section describes the structure of the repoistory.

## Code
The code for data download, preprocessing, modeling and evaluation.  
Please check code folder for detailed instruction.   

## Proposal Report
The group Proposal Report in pdf. 

## Fina Report
The final report for our project in pdf.

## Final Group Presentation
The group presentation slides in pdf.  

## Pre_trained_models
Pretained models (generators) that can generate fraud transcations.
