# Intel Image Classification Project

## Overview
This project implements a deep learning image classifier that categorizes natural scene images into six classes:
- Buildings
- Forest
- Glacier
- Mountain
- Sea
- Street

The model achieves a test accuracy of 93.7% using a fine-tuned Wide ResNet-50-2 architecture with PyTorch.

## Dataset
The project uses the Intel Image Classification dataset, which contains around 14,000 training images and 3,000 test images. Each image is 150x150 pixels in RGB format.

Dataset structure:
```
- seg_train/
  - buildings/ (2191 images)
  - forest/ (2271 images)
  - glacier/ (2404 images)
  - mountain/ (2512 images)
  - sea/ (2274 images)
  - street/ (2382 images)
- seg_test/
  - buildings/ (437 images)
  - forest/ (474 images)
  - glacier/ (553 images)
  - mountain/ (525 images)
  - sea/ (510 images)
  - street/ (501 images)
- seg_pred/ (prediction folder)
```

## Technical Architecture

### Model
- Base Model: Wide ResNet-50-2 (pretrained on ImageNet)
- Transfer Learning: Freezing pretrained weights and retraining only the final fully connected layer
- Output Layer: Modified to output 6 classes instead of the original 1000

### Data Preprocessing
- Resizing: All images are resized to 150x150 pixels
- Data Augmentation (Training):
  - Random horizontal flips
  - Random crops
  - Color jitter (brightness, contrast, saturation, hue)
- Normalization:
  - Training: mean=(0.425, 0.415, 0.405), std=(0.205, 0.205, 0.205)
  - Testing: mean=(0.425, 0.415, 0.405), std=(0.255, 0.245, 0.235)

### Training Setup
- Loss Function: Cross-Entropy Loss
- Optimizer: SGD with initial learning rate of 0.01
- Learning Rate Scheduler: MultiStepLR with gamma=0.055
- Batch Size: 32
- Epochs: 8
- Training/Validation Split: 90%/10%
- Hardware: NVIDIA Tesla P100 GPU

## Performance
- Test Accuracy: 93.7%
- Training Time: ~1250 seconds (8 epochs)
- Model Size: 267.39 MB


## Requirements
- Python 3.x
- PyTorch 2.0.0
- torchvision 0.15.1
- numpy
- pandas
- matplotlib
- seaborn
- torchinfo (for model summary)
- Pillow (PIL)
- tqdm

## Usage

### Training
The model training process is contained in the Jupyter notebook. To retrain the model:
1. Set up your environment with the required dependencies
2. Adjust paths to your dataset locations
3. Run the notebook cells in order
4. The best model weights will be saved as 'My_Model.pt'

### Inference
To use the model for inference on new images:

```python
# Load model
model = torchvision.models.wide_resnet50_2(pretrained=False)
num_features = model.fc.in_features
model.fc = nn.Linear(in_features=num_features, out_features=6)
model.load_state_dict(torch.load('My_Model.pt'))
model.eval()
model.to(device)

# Define classes
classes = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']

# Prediction function
def predict_image(image_path):
    img = Image.open(image_path)
    img_tensor = test_transforms(img)
    img_tensor = img_tensor.unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(img_tensor)
        _, predicted = torch.max(output, 1)
        class_idx = predicted.item()
    return classes[class_idx]
```

## Implementation Details

### Data Loading
- Custom implementation of `ImageFolderCustom` class for dataset handling
- DataLoader with SubsetRandomSampler for batch processing and validation splitting

### Training Process
- Progressive training with learning rate scheduling
- Early stopping based on validation loss improvement
- Checkpointing of best model weights

### Visualization
- Training/validation loss curves
- Sample predictions visualization
- Data augmentation visualization

## Future Improvements
- Experiment with different architectures (EfficientNet, Vision Transformer)
- Implement test-time augmentation
- Add model interpretability (GradCAM, feature visualization)
- Explore mixed-precision training for speed improvements
- Deploy model as a web service or mobile application


## Acknowledgements
- Intel for providing the dataset
- PyTorch team for the deep learning framework
- NVIDIA for GPU acceleration
