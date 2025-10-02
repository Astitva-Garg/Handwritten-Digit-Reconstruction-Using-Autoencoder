# Autoencoders on MNIST

This repository contains two implementations of **Autoencoders** trained on the [MNIST dataset](http://yann.lecun.com/exdb/mnist/) using **PyTorch**:

1. **Linear Autoencoder** – uses fully connected layers to compress and reconstruct the 28×28 MNIST digits.  
2. **Convolutional Autoencoder** – uses convolutional and transposed convolutional layers for better spatial feature extraction and reconstruction.

Both models learn to encode handwritten digits into a compressed latent space and decode them back to images.

---

## What is an Autoencoder?
An **Autoencoder** is an unsupervised neural network that learns to reconstruct its input.  
It consists of:
- **Encoder** – reduces the input dimensionality (compression).
- **Decoder** – reconstructs the input from the compressed representation.

The learning objective is to minimize the reconstruction loss (here, **Mean Squared Error**).

---

## Repository Structure
```
├── linear_autoencoder.py        # Autoencoder with fully connected layers
├── conv_autoencoder.py          # Autoencoder with convolutional layers
├── data/                        # MNIST dataset will be downloaded here
└── README.md
```

---

## Requirements
Install the dependencies with:

```bash
pip install torch torchvision matplotlib
```

---

##  Usage

### 1. Train the Linear Autoencoder
```bash
python ae_torch.py
```

### 2. Train the Convolutional Autoencoder
```bash
python conv_ae_torch.py
```

During training, the script prints the **epoch loss** and visualizes reconstructed images after training.  

---

## 🏗 Model Architectures

### 🔹 Linear Autoencoder
- **Encoder**:  
  `784 → 128 → 64 → 12 → 5`  
- **Decoder**:  
  `5 → 12 → 64 → 128 → 784`  

### 🔹 Convolutional Autoencoder
- **Encoder**:  
  `Conv2d(1→16) → Conv2d(16→32) → Conv2d(32→64)`  
- **Decoder**:  
  `ConvTranspose2d(64→32) → ConvTranspose2d(32→16) → ConvTranspose2d(16→1)`  

---

## Training Details
- **Dataset**: MNIST (28×28 grayscale images)  
- **Loss Function**: Mean Squared Error (MSE)  
- **Optimizer**: Adam (lr=1e-3, weight_decay=1e-5)  
- **Batch Size**: 64  
- **Epochs**: 13  

---

## Results
After training, the models generate reconstructions of MNIST digits.  
For every 4 epochs, the code visualizes:

- **Top row**: Original images  
- **Bottom row**: Reconstructed images  

Example (illustrative layout):

```
+---------+---------+---------+     +---------+---------+---------+
| Original| Original| Original| ... | Reconst | Reconst | Reconst |
+---------+---------+---------+     +---------+---------+---------+
```

The convolutional autoencoder typically achieves sharper reconstructions compared to the linear one.

---

##  Notes
- The **linear autoencoder** flattens the images into vectors.  
- The **convolutional autoencoder** preserves spatial information using Conv2d layers, making it better at capturing digit structure.  

---

## License
This project is open-source and available under the **MIT License**.
