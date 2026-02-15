# Training Guide: Balanced Deepfake Detection on ElevenLabs Dataset

This guide provides a comprehensive overview of the modifications made to the voice-deepfake detection project to support the ElevenLabs dataset, handle class imbalance, and improve model performance. It includes detailed instructions on how to train the models on your local system.

## 1. Project Modifications

To address the challenges of the imbalanced ElevenLabs dataset and to improve the overall training pipeline, several key modifications have been implemented:

### 1.1. Balanced Dataset Loader (`balanced_elevenlabs_dataset.py`)

A new dataset loader has been created to specifically handle the imbalanced nature of the ElevenLabs dataset. This loader incorporates two primary techniques:

*   **Weighted Random Sampler**: This ensures that each batch of data presented to the model during training contains a more balanced representation of both "real" and "fake" audio samples. It does this by oversampling the minority class (in this case, the "real" audio) and undersampling the majority class ("fake" audio).
*   **Class Weights**: In addition to the sampler, the loader calculates class weights that can be used directly in the loss function. This tells the model to pay more attention to the minority class during training, further mitigating the effects of the imbalance.

### 1.2. Optimized Training Script (`train_balanced_elevenlabs.py`)

A new training script has been developed to leverage the balanced dataset loader and to incorporate a more robust training configuration. The key improvements in this script include:

*   **Optimized Hyperparameters**: The learning rate, weight decay, and other hyperparameters have been fine-tuned to improve model convergence and prevent overfitting.
*   **Advanced Augmentation**: The script now utilizes a more aggressive data augmentation strategy, including Mixup and CutMix, to create a more diverse training set and improve the model's ability to generalize.
*   **OneCycleLR Scheduler**: This learning rate scheduler helps to accelerate convergence and often leads to better model performance.
*   **Expanded Model Training**: The script is now configured to train all available model architectures, including the `enhanced`, `lightweight`, and all `ensemble` models (`standard`, `multiscale`, and `adaptive`).

## 2. How to Train the Models

Follow these steps to train the deepfake detection models on your local system. It is assumed that you have already cloned your GitHub repository and have the ElevenLabs dataset available in a directory named `elevenlabs_dataset` at the root of your project.

### 2.1. Prerequisites

Ensure you have all the necessary Python packages installed. You can install them using the following command:

```bash
sudo pip3 install torch torchvision torchaudio audiomentations soundfile librosa scikit-learn pyyaml psutil matplotlib seaborn pandas tqdm
```

### 2.2. Training Commands

To start the training process, you will use the `train_balanced_elevenlabs.py` script. You can choose to run a full training session or a quick test to verify your setup.

#### Full Training (Recommended)

This command will train all models for the number of epochs specified in the configuration file (default is 100). This process can be time-consuming and requires a GPU for optimal performance.

```bash
python3 train_balanced_elevenlabs.py
```

#### Quick Training Test

This command will run a short training session for only 3 epochs. This is useful for quickly verifying that your environment is set up correctly and that the training process can start without errors.

```bash
python3 train_balanced_elevenlabs.py --quick_test
```

### 2.3. Customizing the Training

You can customize the training process by modifying the `get_custom_config()` function in the `train_balanced_elevenlabs.py` script. This allows you to experiment with different hyperparameters, such as the number of epochs, batch size, learning rate, and augmentation strategies.

## 3. Expected Outcome

Upon successful completion of the training, you will find the trained model files in the `model/` directory. Each model will be saved with a filename that indicates its architecture and that it was trained on the balanced ElevenLabs dataset (e.g., `enhanced_balanced_elevenlabs.pth`).

These models should exhibit improved performance and higher confidence scores in detecting deepfakes, thanks to the balanced training approach and optimized training optimizations implemented.
