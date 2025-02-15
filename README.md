# ResNet-AnimalEmotion
This project implements a pretrained ResNet-18 model for classifying pet facial expressions into three categories: Angry, Sad, and Happy. The model is fine-tuned using PyTorch and Torchvision, leveraging transfer learning for improved accuracy.

### Features：
- Pretrained Model: Uses ResNet-18 trained on ImageNet, with the final layer modified for three-class classification.
- Transfer Learning: Efficient training by adapting ResNet-18’s features to a new dataset.
- Data Augmentation: Resizes images, normalizes them, and applies transformations for better generalization.
- Training & Evaluation: Implements a custom training loop with performance tracking.
- Performance Visualization: Generates loss and accuracy plots for analysis.

### Example Console Output
```EPOCH: 24
Loss=0.017485028132796288, Batch_id=46 Accuracy=99.87%
current Learning Rate: 1.0000000000000002e-06
Test set: Average loss: 0.1414, Accuracy: 18/32 (56.25%)

### Training Results
![Training and Test Results](results.png)
