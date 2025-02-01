# Blood-Cells-Classification
This project aims to develop a machine learning model for the classification of blood cells. Accurate classification of blood cells plays a crucial role in healthcare, especially in hematology, by enabling the rapid diagnosis 
of diseases and infections.

![kan_hücreleri](https://github.com/user-attachments/assets/d1b20eda-0c0f-441c-9f69-b2236d9a5b97)

## Dataset  
The **[Blood Cells Image Dataset](https://www.kaggle.com/datasets/unclesamulus/blood-cells-image-dataset)** from Kaggle was used in this study. It contains **17,092 images** of blood cells categorized into **eight different classes**.

- Neutrophils  
- Eosinophils  
- Basophils  
- Lymphocytes  
- Monocytes  
- Immature Granulocytes  
- Erythroblasts  
- Platelets

## Data Distribution  
To analyze the dataset, we examined a batch of **16 randomly selected images** and the **class distribution in the training set**.

#### Sample Batch (Batch Size: 16): 
![simple_cnn_16grid](https://github.com/user-attachments/assets/7068cfa1-71c8-4283-8307-3e9ca5ba5ca3) 
#### Class Distribution in Training Data: 
![class_distribution](https://github.com/user-attachments/assets/88a4607b-0915-4d0c-84eb-11b2e4490fea)

## Model Development  
During the development phase, we experimented with **four different CNN architectures**:  

1. **Basic CNN**: A simple convolutional model with minimal layers.  
2. **CNN with Data Augmentation**: Included random flips and rotations to improve generalization.  
3. **Complex CNN**: A deeper network with multiple convolutional and fully connected layers.  
4. **Final Model - Enhanced Complex CNN**: An optimized version of the third model with adjusted hyperparameters and extended training.  

The **final implementation** uploaded to this repository is **Model 4 - Enhanced Complex CNN**, as it achieved the highest accuracy and the most stable performance.

## Model Details  
- **Image Size**: 224x224  
- **Batch Size**: 16  
- **Optimizer**: Adam  
- **Loss Function**: Sparse Categorical Crossentropy  
- **Epochs**: 50 (with Early Stopping at 10 epochs)  
- **Dropout Rate**: 50% in fully connected layers  
- **Normalization**: Batch Normalization after each Conv2D layer  
- **Activation Functions**:  
  - ReLU for feature extraction layers  
  - Softmax for classification

## Performance Graph  
![model4_sonuc](https://github.com/user-attachments/assets/c1af1abd-fe1d-4be9-b321-4c51d936fa9d)

## Results  
The final model achieved a **classification accuracy of 93.07%**, demonstrating its effectiveness in distinguishing different blood cell types.  

## Future Work  
- Expanding the dataset to improve model generalization.  
- Optimizing model performance using hybrid algorithms or transfer learning.  
- Integrating the model into real-time clinical systems.  
- Developing a mobile or web-based application for accessibility.  

## Contributors  
- **Eren Akıncı**  
- **Yunus Emre Maral**  

This project showcases how deep learning can significantly enhance medical image classification, potentially reducing human errors and improving diagnostic efficiency.  
