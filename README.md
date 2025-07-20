# ANN from Scratch - FashionMNIST Classifier Using PyTorch
Welcome! This project demonstrates how to build an **Artificial Neural Network (ANN)** from scratch using **PyTorch** to classify images from the **FashionMNIST** dataset. This project was developed and executed entirely on Google Colab and showcases a classic deep learning pipeline from data loading to evaluation.

### 📌 Table of Contents

- [Overview](#overview)
- [Technologies Used](#technologies-used)
- [How to Run the Code](#how-to-run-the-code)
- [Results](#results)
- [What's Inside the Model](#whats-inside-the-model)
- [Future Improvements](#future-improvements)
  
### 📖 Overview

This notebook implements a feedforward **multi-layer neural network** (also called a Multi-Layer Perceptron or MLP) to classify grayscale fashion items like shirts, shoes, bags, etc., from the [FashionMNIST dataset](https://github.com/zalandoresearch/fashion-mnist).

Core tasks covered:
- Data preprocessing and visualization
- Custom PyTorch `Dataset` and `DataLoader`
- Neural network design from scratch using `torch.nn.Sequential`
- Model training with backpropagation using the SGD optimizer
- Evaluation of model performance on unseen (test) data

### 🧰 Technologies Used

| Library      | Purpose                              |
|--------------|---------------------------------------|
| `PyTorch`    | Neural network building & training    |
| `Pandas`     | CSV data loading and preprocessing    |
| `Matplotlib` | Visualization of image data           |
| `scikit-learn` | Train-test split                   |

### ▶️ How to Run the Code

1. 🔗 Open the notebook in **Google Colab**:  

2. ⬆️ Upload the dataset file `fmnist_small.csv` (if not included in Colab drive)

3. 🏁 Run each cell step-by-step:
   - Includes data loading, visualization, model training and test evaluation.

4. 📊 After training, the notebook prints accuracy and plots sample outputs.

### ✅ Results

- **Model Type:** Fully connected ANN
- **Input Features:** 784 (28x28 images flattened)
- **Accuracy Achieved:** ~85–90% (depending on training epochs and parameters)

### 🧬 What's Inside the Model?

```python
model = nn.Sequential(
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 10)
)
```

- **Loss Function:** CrossEntropyLoss
- **Optimizer:** Stochastic Gradient Descent (SGD)
- **Epochs:** 100
- **Batch Size:** 32

### 🚀 Future Improvements

- Add **early stopping** or **learning rate schedulers**
- Experiment with **dropout** or **batch normalization**
- Convert to a **Convolutional Neural Network (CNN)** for accuracy boost
- Use the full FashionMNIST dataset or load directly via `torchvision.datasets`

### 🙌 Acknowledgement

- FashionMNIST Dataset by Zalando Research
- Deep learning community tutorials for inspiration
