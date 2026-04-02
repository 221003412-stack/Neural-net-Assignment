Chest X-Ray Pneumonia Classifier
 Overview

This project presents a Convolutional Neural Network (CNN) designed to analyze chest 
X-ray images and classify them as either **NORMAL** or **PNEUMONIA**. The primary objective 
of this model is to maximize the detection of pneumonia cases, as missing a positive case 
can have serious clinical consequences. At the same time, the study also explores the trade-offs 
involved in optimizing medical AI systems, particularly between sensitivity and specificity.



 Dataset

The dataset consists of pediatric chest X-ray images divided into training, validation, and testing sets:

Training Set: 1,341 NORMAL | 3,875 PNEUMONIA
Validation Set: 8 NORMAL | 8 PNEUMONIA
Test Set: 234 NORMAL | 390 PNEUMONIA

A significant class imbalance exists, with pneumonia cases nearly three times more frequent than normal cases.
This imbalance introduces bias in model learning and required careful handling to avoid skewed predictions.


Methodology

1. Data Loading
The dataset was loaded from Google Drive. Directory paths were configured, and image counts per class were verified to ensure correct data distribution and integrity.

2. Handling Class Imbalance
To address imbalance, class weights were computed using `compute_class_weight`. The NORMAL class was assigned approximately 1.94× higher importance,
encouraging the model to pay more attention to underrepresented samples.

3. Preprocessing and Augmentation

* Images were normalized to the range [0,1]
* Light augmentation was applied only to training data:

  * Small rotations (±15°)
  * Minor horizontal and vertical shifts
  * Slight zoom and shear transformations
  * Horizontal flipping

Augmentation was intentionally kept minimal to maintain clinical realism, as excessive distortion can negatively affect medical image interpretation.

4. Model Architecture

* Four convolutional blocks with increasing filters: 32 → 64 → 128 → 256
* Each block includes Batch Normalization, Max Pooling, and 25% Dropout
* GlobalAveragePooling2D used instead of Flatten to reduce parameter count and overfitting
* Two fully connected layers (256 and 128 neurons) with L2 regularization and 50% Dropout
* Tanh activation function used in hidden layers
* Output layer: Single neuron with sigmoid activation for binary classification

5. Compilation

* Optimizer: Adam (learning rate = 0.001)
* Loss Function: Binary Crossentropy
* Metrics: Accuracy, Precision, Recall, AUC, and F1 Score

6. Training Strategy

* Early stopping, model checkpointing, and learning rate reduction were implemented
* Validation Recall (**val_recall**) was used as the primary monitoring metric
* Training was conducted for up to 30 epochs with class weights applied

7. Evaluation
The model was evaluated on the test dataset using:

* Confusion Matrix
* ROC AUC Score
* Sensitivity (Recall)
* Specificity

---

 Approach

Handling Class Imbalance
Class weighting ensured that the minority class (NORMAL) contributed more significantly to the loss function, helping reduce bias toward the majority class.

Focus on Recall
Recall was prioritized because, in medical diagnosis, failing to detect a disease (false negative) 
is often more dangerous than a false positive. This design choice intentionally biases the model toward detecting as many pneumonia cases as possible.


 Libraries Used

* TensorFlow / Keras** – Model building and training
* NumPy – Numerical computations
* Pandas – Data manipulation
* OpenCV & PIL – Image preprocessing
* Matplotlib & Seaborn – Visualization of results
* Scikit-learn – Evaluation metrics, ROC analysis, and class weighting



Results

| Metric               | Score |
| -------------------- | ----- |
| Accuracy             | 62.5% |
| Recall (Sensitivity) | 100%  |
| Specificity          | 0%    |
| AUC                  | 0.72  |
| F1 Score             | 0.63  |

The model predicted PNEUMONIA for all test images, resulting in:

* True Positives (TP): 390
* False Positives (FP): 234
* True Negatives (TN): 0
* False Negatives (FN): 0

 Discussion

The results reveal a strong bias toward predicting pneumonia for every input. This occurred due to the combined influence of 
class weighting and recall-based optimization during training. By prioritizing recall, the model effectively learned a shortcut
strategy—predicting all images as pneumonia ensures zero false negatives.

Despite this limitation, the **AUC score of 0.72** indicates that the model has learned useful underlying patterns and retains 
some ability to distinguish between classes. The issue lies primarily in the classification threshold (default = 0.5), which 
caused all predictions to fall into the positive class.

This highlights an important concept in machine learning: **model performance is not only determined by learning but also by 
decision thresholds and evaluation strategy**.


 Limitations

* The validation set contains only **16 images**, which is insufficient for reliable performance evaluation
* Small validation size leads to unstable metrics and unreliable early stopping
* Overemphasis on recall caused the model to ignore specificity entirely
* The model lacks practical usability in its current form due to excessive false positives

 Future Improvements

* Increase validation set size or apply **Stratified K-Fold Cross-Validation**
* Tune the classification threshold using ROC or Precision-Recall curves
* Monitor more balanced metrics such as **AUC or validation loss** instead of only recall
* Experiment with alternative loss functions such as **Focal Loss**
* Replace tanh activation with **ReLU or LeakyReLU** for better convergence
* Apply more advanced architectures (e.g., transfer learning with pre-trained models like ResNet or EfficientNet)



 Conclusion

This project demonstrates both the strengths and limitations of CNN-based medical image classification.
While the model successfully achieved perfect sensitivity, it failed to maintain balance across other important metrics.
The findings emphasize that in real-world medical applications, achieving a balance between sensitivity and specificity is crucial.
Overall, this work provides a strong foundation and valuable insights into handling imbalanced datasets, optimizing evaluation strategies,
and improving deep learning models for healthcare applications.
