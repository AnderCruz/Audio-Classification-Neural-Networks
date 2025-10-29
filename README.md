# Audio Classification with Neural Networks

This project provides a comprehensive exploration of various neural network approaches for audio classification using TensorFlow and Keras. The implementation covers the complete machine learning pipeline from data loading and preprocessing to model deployment and inference.

## Project Overview

The notebook demonstrates multiple state-of-the-art techniques for audio classification, comparing different architectural approaches and their performance on speech command recognition tasks.

## Key Features & Techniques

### Data Handling & Preprocessing
- **Audio Loading**: Reading WAV files from compressed archives (.gz, .tar formats)
- **Signal Processing**: Resampling audio to consistent 16kHz sample rate using SciPy
- **Normalization**: Padding/trimming audio to fixed length (16,000 samples)
- **Efficient Pipelines**: Using `tf.data.Dataset` for optimized data loading with shuffling, batching, and prefetching
- **Label Encoding**: Converting string labels to integers using scikit-learn's LabelEncoder

### Model Architectures

#### 1. Time-Domain Classification (1D CNN)
- **Input**: Raw audio waveforms (16,000 samples × 1 channel)
- **Architecture**:
  - Conv1D (16 filters, kernel_size=3, ReLU activation)
  - MaxPooling1D (pool_size=2)
  - Flatten layer
  - Dense (64 units, ReLU)
  - Output (36 units, Softmax)
- **Parameters**: ~4.1 million trainable parameters
- **Use Case**: Direct learning from raw audio signals

#### 2. Frequency-Domain Classification (2D CNN & Spectrograms)
- **Input**: Spectrograms generated via Short-Time Fourier Transform (STFT)
- **Processing**: Converting time-domain signals to frequency-domain representations
- **Architecture**: 2D convolutional layers adapted for spectrogram input
- **Enhancements**: Custom normalization layers and attention mechanisms

#### 3. Attention Mechanisms
- **Custom Implementation**: ChannelAttention layer for Keras
- **Integration**: Enhanced 2D CNN architecture with attention gates
- **Benefits**: Improved feature focus and model interpretability

#### 4. Transfer Learning with YAMNet
- **Base Model**: Pre-trained YAMNet audio event classification model from TensorFlow Hub
- **Feature Extraction**: Using YAMNet embeddings (1,024 dimensions)
- **Custom Head**: Training new classification layers on top of frozen embeddings
- **Dataset Adaptation**: Applied to ESC-50 environmental sound classification dataset

## Dataset Information

### Speech Commands Dataset
- **36 Audio Classes**: 
  - Basic commands: "yes", "no", "stop", "go", "up", "down", "left", "right"
  - Numbers: "zero" through "nine"
  - Animals: "bird", "dog", "cat"
  - Household: "bed", "house", "tree"
  - Miscellaneous: "happy", "wow", "follow", "learn", "visual", etc.
  - Background noise category

### ESC-50 Dataset (for Transfer Learning)
- Environmental Sound Classification dataset
- 50 classes of environmental recordings
- Used for YAMNet transfer learning experiments

## Technical Implementation

### Preprocessing Pipeline
```python
def load_and_process_audio(filename, max_length=16000):
    # Read and decode WAV file
    # Resample to 16kHz using SciPy
    # Pad/trim to fixed length
    # Return normalized tensor
```

### Training Configuration
- **Optimizer**: Adam
- **Loss Function**: Sparse Categorical Crossentropy
- **Metrics**: Accuracy
- **Batch Size**: 32
- **Validation Split**: 20% stratified split
- **Epochs**: 10+ with early stopping potential

### Model Evaluation
- Training/validation accuracy and loss tracking
- Visualization of learning curves
- Confusion matrix analysis
- Performance comparison across architectures

## Usage Examples

### Basic Training
```python
# Time-domain model training
history_time_domain = model_time_domain.fit(
    train_dataset,
    epochs=10,
    batch_size=32,
    validation_data=val_dataset
)
```

### Spectrogram Generation
```python
# Convert audio to spectrograms
spectrogram = tf.signal.stft(audio, frame_length=255, frame_step=128)
```

### Transfer Learning
```python
# Load pre-trained YAMNet
yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')
# Extract embeddings and train custom classifier
```

## Performance Metrics

The project includes comprehensive evaluation of:
- **Training Accuracy**: Model performance on training data
- **Validation Accuracy**: Generalization capability
- **Loss Curves**: Training stability and convergence
- **Inference Speed**: Real-time classification potential
- **Model Size**: Parameter efficiency

## 🎮 Inference & Deployment

### Single Audio Prediction
```python
def predict_audio_class(model, audio_path):
    # Preprocess audio
    # Run model inference
    # Return class probabilities and predicted label
```

### Model Saving
```python
model_spectrogram.save('audio_classification_model.h5')
```

## Requirements

### Core Dependencies
```
tensorflow>=2.8.0
numpy>=1.21.0
scipy>=1.7.0
scikit-learn>=1.0.0
matplotlib>=3.5.0
librosa>=0.9.0
```

### Optional Dependencies
```
tensorflow-hub  # For YAMNet transfer learning
ipython         # For notebook visualization
```

## Project Structure

```
audio_classification/
├── Audio_Classification.ipynb      # Main notebook
├── data_audio/
│   └── dataset_commands.gz         # Compressed dataset
├── models/
│   ├── time_domain_model.h5        # Saved 1D CNN model
│   └── spectrogram_model.h5        # Saved 2D CNN model
└── utils/
    └── audio_processing.py         # Helper functions
```

## Tags

`audio-classification` `neural-networks` `tensorflow` `keras` `cnn` `spectrogram-analysis` `transfer-learning` `yamnet` `speech-recognition` `machine-learning` `deep-learning` `audio-processing` `python` `1d-cnn` `2d-cnn` `attention-mechanism` `signal-processing` `audio-ml` `environmental-sound-classification` `speech-commands` `esc-50` `data-augmentation` `tf-data-pipeline`

## Research Applications

This project demonstrates practical implementations of:
- Multi-modal neural network architectures for audio
- Comparative analysis of time-domain vs frequency-domain approaches
- Effective transfer learning strategies for audio tasks
- Attention mechanisms for improved feature learning
- Production-ready data preprocessing pipelines

## Potential Extensions

- Real-time audio classification
- Mobile deployment with TensorFlow Lite
- Multi-label audio classification
- Audio generation and style transfer
- Cross-modal learning (audio + text)

## References

- TensorFlow Audio Recognition Tutorials
- YAMNet: Pre-trained audio event classifier
- Speech Commands Dataset (Google)
- ESC-50 Dataset for environmental sound classification
- Attention mechanisms in audio processing literature

*This project serves as both an educational resource and a practical foundation for building production audio classification systems.*
