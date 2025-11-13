---
layout: default
title: "Comparative Analysis of Neural Network Architectures for ECG Classification"
description: "A Comprehensive Study of Seven Deep Learning Approaches"
---

## Abstract

This project presents a comprehensive comparative analysis of seven deep learning architectures for electrocardiogram (ECG) classification: a traditional feedforward neural network (FFNN), a Transformer-based model, a Three-Stage Hierarchical Transformer (3stageFormer), a 1D Convolutional Neural Network (CNN), a Long Short-Term Memory (LSTM) network, a Hopfield Network, and a Variational Autoencoder (VAE). The feedforward architecture is based on the seminal work by Lloyd et al. (2001) for ischemia detection, the Transformer model follows the approach by Ikram et al. (2025) for early detection of cardiac arrhythmias, the 3stageFormer implements the hierarchical multi-scale approach by Tang et al. (2025), the Hopfield Network is based on energy-based associative memory approaches for ECG analysis (ETASR, 2013), and the VAE implements the FactorECG approach by van de Leur et al. (2022) for explainable ECG analysis. We additionally implement CNN and LSTM models, which represent alternative approaches using convolution and recurrent connections respectively. We implement all seven models from scratch and conduct extensive benchmarking on synthetic ECG data. Our results demonstrate that Transformer-based models achieve superior classification accuracy by effectively capturing temporal dependencies, with the Three-Stage Hierarchical Transformer providing additional benefits through multi-scale feature extraction. The CNN model offers an excellent balance between accuracy and efficiency, effectively capturing local morphological patterns. The LSTM model provides strong sequential modeling capabilities. The Hopfield Network demonstrates unique energy-based pattern recognition capabilities. The VAE provides explainable latent representations that enable both reconstruction and classification tasks. The feedforward neural network offers significant advantages in computational efficiency, making it more suitable for real-time applications. This study provides comprehensive insights into the trade-offs between model complexity and performance, guiding the selection of appropriate architectures for different ECG classification scenarios.

## Seven Deep Learning Architectures

<div class="model-grid">
    <div class="model-card">
        <h4>1. Feedforward NN</h4>
        <ul>
            <li><span class="metric-label">Type:</span> Feature-based MLP</li>
            <li><span class="metric-label">Input:</span> Statistical features</li>
            <li><span class="metric-label">Speed:</span> Fastest</li>
            <li><span class="metric-label">Best For:</span> Real-time, edge devices</li>
        </ul>
    </div>
    <div class="model-card">
        <h4>2. Transformer</h4>
        <ul>
            <li><span class="metric-label">Type:</span> Single-scale Attention</li>
            <li><span class="metric-label">Input:</span> Raw signals</li>
            <li><span class="metric-label">Speed:</span> Moderate</li>
            <li><span class="metric-label">Best For:</span> High-accuracy, research</li>
        </ul>
    </div>
    <div class="model-card">
        <h4>3. 3stageFormer</h4>
        <ul>
            <li><span class="metric-label">Type:</span> Multi-scale Attention</li>
            <li><span class="metric-label">Input:</span> Raw (3 resolutions)</li>
            <li><span class="metric-label">Speed:</span> Slowest</li>
            <li><span class="metric-label">Best For:</span> Multi-scale patterns</li>
        </ul>
    </div>
    <div class="model-card">
        <h4>4. 1D CNN</h4>
        <ul>
            <li><span class="metric-label">Type:</span> Convolutional</li>
            <li><span class="metric-label">Input:</span> Raw signals</li>
            <li><span class="metric-label">Speed:</span> Fast</li>
            <li><span class="metric-label">Best For:</span> Local patterns, efficiency</li>
        </ul>
    </div>
    <div class="model-card">
        <h4>5. LSTM</h4>
        <ul>
            <li><span class="metric-label">Type:</span> Recurrent</li>
            <li><span class="metric-label">Input:</span> Raw signals</li>
            <li><span class="metric-label">Speed:</span> Moderate</li>
            <li><span class="metric-label">Best For:</span> Sequential patterns</li>
        </ul>
    </div>
    <div class="model-card">
        <h4>6. Hopfield</h4>
        <ul>
            <li><span class="metric-label">Type:</span> Energy-based</li>
            <li><span class="metric-label">Input:</span> Raw signals</li>
            <li><span class="metric-label">Speed:</span> Moderate</li>
            <li><span class="metric-label">Best For:</span> Pattern completion</li>
        </ul>
    </div>
    <div class="model-card">
        <h4>7. VAE</h4>
        <ul>
            <li><span class="metric-label">Type:</span> Variational Autoencoder</li>
            <li><span class="metric-label">Input:</span> Raw signals</li>
            <li><span class="metric-label">Speed:</span> Moderate</li>
            <li><span class="metric-label">Best For:</span> Explainable AI</li>
        </ul>
    </div>
</div>

## Comprehensive Comparison

### Architectural Comparison

<div class="svg-container">
    <svg width="1000" height="600" viewBox="0 0 1000 600" xmlns="http://www.w3.org/2000/svg">
        <!-- Background -->
        <rect width="1000" height="600" fill="#f8f9fa"/>
        
        <!-- Title -->
        <text x="500" y="30" text-anchor="middle" font-size="20" font-weight="bold" fill="#333">Model Comparison: Architecture vs. Performance</text>
        
        <!-- Axes -->
        <line x1="100" y1="500" x2="900" y2="500" stroke="#333" stroke-width="2"/>
        <line x1="100" y1="500" x2="100" y2="50" stroke="#333" stroke-width="2"/>
        
        <!-- Axis labels -->
        <text x="500" y="550" text-anchor="middle" font-size="14" fill="#666">Computational Complexity</text>
        <text x="30" y="275" text-anchor="middle" font-size="14" fill="#666" transform="rotate(-90 30 275)">Classification Accuracy</text>
        
        <!-- Grid lines -->
        <line x1="200" y1="50" x2="200" y2="500" stroke="#ddd" stroke-width="1" stroke-dasharray="5,5"/>
        <line x1="400" y1="50" x2="400" y2="500" stroke="#ddd" stroke-width="1" stroke-dasharray="5,5"/>
        <line x1="600" y1="50" x2="600" y2="500" stroke="#ddd" stroke-width="1" stroke-dasharray="5,5"/>
        <line x1="800" y1="50" x2="800" y2="500" stroke="#ddd" stroke-width="1" stroke-dasharray="5,5"/>
        <line x1="100" y1="400" x2="900" y2="400" stroke="#ddd" stroke-width="1" stroke-dasharray="5,5"/>
        <line x1="100" y1="300" x2="900" y2="300" stroke="#ddd" stroke-width="1" stroke-dasharray="5,5"/>
        <line x1="100" y1="200" x2="900" y2="200" stroke="#ddd" stroke-width="1" stroke-dasharray="5,5"/>
        <line x1="100" y1="100" x2="900" y2="100" stroke="#ddd" stroke-width="1" stroke-dasharray="5,5"/>
        
        <!-- Models as points -->
        <!-- FFNN: Low complexity, Good accuracy -->
        <circle cx="150" cy="350" r="12" fill="#e74c3c"/>
        <text x="150" y="335" text-anchor="middle" font-size="12" fill="#333">FFNN</text>
        
        <!-- CNN: Low-Medium complexity, Good-Excellent accuracy -->
        <circle cx="250" cy="280" r="12" fill="#3498db"/>
        <text x="250" y="265" text-anchor="middle" font-size="12" fill="#333">CNN</text>
        
        <!-- LSTM: Medium complexity, Good-Excellent accuracy -->
        <circle cx="400" cy="250" r="12" fill="#9b59b6"/>
        <text x="400" y="235" text-anchor="middle" font-size="12" fill="#333">LSTM</text>
        
        <!-- Hopfield: Medium complexity, Good-Excellent accuracy -->
        <circle cx="450" cy="260" r="12" fill="#f39c12"/>
        <text x="450" y="245" text-anchor="middle" font-size="12" fill="#333">Hopfield</text>
        
        <!-- VAE: Medium complexity, Good-Excellent accuracy -->
        <circle cx="500" cy="240" r="12" fill="#1abc9c"/>
        <text x="500" y="225" text-anchor="middle" font-size="12" fill="#333">VAE</text>
        
        <!-- Transformer: High complexity, Excellent accuracy -->
        <circle cx="700" cy="150" r="12" fill="#2ecc71"/>
        <text x="700" y="135" text-anchor="middle" font-size="12" fill="#333">Transformer</text>
        
        <!-- 3stageFormer: Highest complexity, Excellent+ accuracy -->
        <circle cx="850" cy="100" r="12" fill="#e67e22"/>
        <text x="850" y="85" text-anchor="middle" font-size="12" fill="#333">3stage</text>
        
        <!-- Legend -->
        <rect x="750" y="50" width="200" height="120" fill="white" stroke="#ddd" stroke-width="1"/>
        <text x="850" y="70" text-anchor="middle" font-size="14" font-weight="bold" fill="#333">Legend</text>
        <circle cx="770" cy="90" r="6" fill="#e74c3c"/>
        <text x="790" y="95" font-size="11" fill="#333">Feedforward NN</text>
        <circle cx="770" cy="110" r="6" fill="#3498db"/>
        <text x="790" y="115" font-size="11" fill="#333">CNN</text>
        <circle cx="770" cy="130" r="6" fill="#9b59b6"/>
        <text x="790" y="135" font-size="11" fill="#333">LSTM</text>
        <circle cx="770" cy="150" r="6" fill="#f39c12"/>
        <text x="790" y="155" font-size="11" fill="#333">Hopfield</text>
        <circle cx="770" cy="170" r="6" fill="#1abc9c"/>
        <text x="790" y="175" font-size="11" fill="#333">VAE</text>
    </svg>
</div>

### Performance Metrics Comparison

| Model | Architecture Type | Input Format | Temporal Modeling | Parameters | Training Speed | Accuracy | Explainability |
|-------|------------------|--------------|-------------------|------------|----------------|----------|----------------|
| **FFNN** | Feature MLP | Statistical features | None | Few (100s-1Ks) | Fastest | Good | Moderate |
| **Transformer** | Single-scale Attention | Raw signals | Global | Many (100Ks) | Moderate | Excellent | High (attention) |
| **3stageFormer** | Multi-scale Attention | Raw (3 scales) | Multi-scale | Most (100Ks+) | Slowest | Excellent+ | High (hierarchical) |
| **CNN** | Convolutional | Raw signals | Local | Moderate (10Ks-100Ks) | Fast | Good-Excellent | Moderate |
| **LSTM** | Recurrent | Raw signals | Sequential | Moderate (10Ks-100Ks) | Moderate | Good-Excellent | High (sequential) |
| **Hopfield** | Energy-based | Raw signals | Associative | Moderate (10Ks-100Ks) | Moderate | Good-Excellent | Moderate |
| **VAE** | Variational Autoencoder | Raw signals | Latent factors | Moderate (10Ks-100Ks) | Moderate | Good-Excellent | Highest (factors) |

### Trade-offs Visualization

<div class="svg-container">
    <svg width="1000" height="500" viewBox="0 0 1000 500" xmlns="http://www.w3.org/2000/svg">
        <!-- Background -->
        <rect width="1000" height="500" fill="#f8f9fa"/>
        
        <!-- Title -->
        <text x="500" y="30" text-anchor="middle" font-size="20" font-weight="bold" fill="#333">Accuracy vs. Efficiency Trade-offs</text>
        
        <!-- Axes -->
        <line x1="100" y1="400" x2="900" y2="400" stroke="#333" stroke-width="2"/>
        <line x1="100" y1="400" x2="100" y2="50" stroke="#333" stroke-width="2"/>
        
        <!-- Axis labels -->
        <text x="500" y="450" text-anchor="middle" font-size="14" fill="#666">Training Speed (Fast → Slow)</text>
        <text x="30" y="225" text-anchor="middle" font-size="14" fill="#666" transform="rotate(-90 30 225)">Classification Accuracy (Good → Excellent)</text>
        
        <!-- Grid lines -->
        <line x1="200" y1="50" x2="200" y2="400" stroke="#ddd" stroke-width="1" stroke-dasharray="3,3"/>
        <line x1="400" y1="50" x2="400" y2="400" stroke="#ddd" stroke-width="1" stroke-dasharray="3,3"/>
        <line x1="600" y1="50" x2="600" y2="400" stroke="#ddd" stroke-width="1" stroke-dasharray="3,3"/>
        <line x1="800" y1="50" x2="800" y2="400" stroke="#ddd" stroke-width="1" stroke-dasharray="3,3"/>
        <line x1="100" y1="300" x2="900" y2="300" stroke="#ddd" stroke-width="1" stroke-dasharray="3,3"/>
        <line x1="100" y1="200" x2="900" y2="200" stroke="#ddd" stroke-width="1" stroke-dasharray="3,3"/>
        <line x1="100" y1="100" x2="900" y2="100" stroke="#ddd" stroke-width="1" stroke-dasharray="3,3"/>
        
        <!-- Models positioned by accuracy and speed -->
        <!-- FFNN: Fastest, Good accuracy -->
        <circle cx="150" cy="320" r="18" fill="#e74c3c" opacity="0.8"/>
        <text x="150" y="305" text-anchor="middle" font-size="13" font-weight="bold" fill="#333">FFNN</text>
        <text x="150" y="360" text-anchor="middle" font-size="10" fill="#666">Fastest</text>
        
        <!-- CNN: Fast, Good-Excellent accuracy -->
        <circle cx="250" cy="250" r="18" fill="#3498db" opacity="0.8"/>
        <text x="250" y="235" text-anchor="middle" font-size="13" font-weight="bold" fill="#333">CNN</text>
        <text x="250" y="290" text-anchor="middle" font-size="10" fill="#666">Best Balance</text>
        
        <!-- LSTM: Moderate, Good-Excellent accuracy -->
        <circle cx="500" cy="220" r="18" fill="#9b59b6" opacity="0.8"/>
        <text x="500" y="205" text-anchor="middle" font-size="13" font-weight="bold" fill="#333">LSTM</text>
        <text x="500" y="260" text-anchor="middle" font-size="10" fill="#666">Sequential</text>
        
        <!-- Hopfield: Moderate, Good-Excellent accuracy -->
        <circle cx="550" cy="230" r="18" fill="#f39c12" opacity="0.8"/>
        <text x="550" y="215" text-anchor="middle" font-size="13" font-weight="bold" fill="#333">Hopfield</text>
        <text x="550" y="270" text-anchor="middle" font-size="10" fill="#666">Energy-based</text>
        
        <!-- VAE: Moderate, Good-Excellent accuracy -->
        <circle cx="600" cy="210" r="18" fill="#1abc9c" opacity="0.8"/>
        <text x="600" y="195" text-anchor="middle" font-size="13" font-weight="bold" fill="#333">VAE</text>
        <text x="600" y="250" text-anchor="middle" font-size="10" fill="#666">Explainable</text>
        
        <!-- Transformer: Moderate-Slow, Excellent accuracy -->
        <circle cx="750" cy="120" r="18" fill="#2ecc71" opacity="0.8"/>
        <text x="750" y="105" text-anchor="middle" font-size="13" font-weight="bold" fill="#333">Transformer</text>
        <text x="750" y="160" text-anchor="middle" font-size="10" fill="#666">High Accuracy</text>
        
        <!-- 3stageFormer: Slowest, Excellent+ accuracy -->
        <circle cx="850" cy="80" r="18" fill="#e67e22" opacity="0.8"/>
        <text x="850" y="65" text-anchor="middle" font-size="13" font-weight="bold" fill="#333">3stage</text>
        <text x="850" y="120" text-anchor="middle" font-size="10" fill="#666">Best Accuracy</text>
        
        <!-- Sweet spot annotation -->
        <ellipse cx="250" cy="250" rx="90" ry="50" fill="none" stroke="#3498db" stroke-width="2" stroke-dasharray="5,5" opacity="0.6"/>
        <text x="250" y="320" text-anchor="middle" font-size="13" fill="#3498db" font-weight="bold">Sweet Spot: CNN</text>
        
        <!-- Efficiency frontier line -->
        <path d="M 150,320 Q 250,250 850,80" fill="none" stroke="#667eea" stroke-width="2" stroke-dasharray="8,4" opacity="0.4"/>
    </svg>
</div>

### Architectural Paradigms Comparison

<div class="svg-container">
    <svg width="1000" height="400" viewBox="0 0 1000 400" xmlns="http://www.w3.org/2000/svg">
        <!-- Background -->
        <rect width="1000" height="400" fill="#f8f9fa"/>
        
        <!-- Title -->
        <text x="500" y="30" text-anchor="middle" font-size="20" font-weight="bold" fill="#333">Temporal Modeling Paradigms</text>
        
        <!-- Paradigm categories -->
        <!-- Feature-based -->
        <rect x="50" y="80" width="120" height="280" fill="#e74c3c" opacity="0.3" rx="5"/>
        <text x="110" y="100" text-anchor="middle" font-size="14" font-weight="bold" fill="#333">Feature-based</text>
        <text x="110" y="120" text-anchor="middle" font-size="12" fill="#666">FFNN</text>
        
        <!-- Attention-based -->
        <rect x="200" y="60" width="200" height="300" fill="#2ecc71" opacity="0.3" rx="5"/>
        <text x="300" y="80" text-anchor="middle" font-size="14" font-weight="bold" fill="#333">Attention-based</text>
        <text x="300" y="100" text-anchor="middle" font-size="12" fill="#666">Transformer</text>
        <text x="300" y="115" text-anchor="middle" font-size="12" fill="#666">3stageFormer</text>
        
        <!-- Convolution-based -->
        <rect x="430" y="100" width="120" height="260" fill="#3498db" opacity="0.3" rx="5"/>
        <text x="490" y="120" text-anchor="middle" font-size="14" font-weight="bold" fill="#333">Convolution</text>
        <text x="490" y="140" text-anchor="middle" font-size="12" fill="#666">CNN</text>
        
        <!-- Recurrent -->
        <rect x="580" y="90" width="120" height="270" fill="#9b59b6" opacity="0.3" rx="5"/>
        <text x="640" y="110" text-anchor="middle" font-size="14" font-weight="bold" fill="#333">Recurrent</text>
        <text x="640" y="130" text-anchor="middle" font-size="12" fill="#666">LSTM</text>
        
        <!-- Energy-based -->
        <rect x="730" y="100" width="120" height="260" fill="#f39c12" opacity="0.3" rx="5"/>
        <text x="790" y="120" text-anchor="middle" font-size="14" font-weight="bold" fill="#333">Energy-based</text>
        <text x="790" y="140" text-anchor="middle" font-size="12" fill="#666">Hopfield</text>
        
        <!-- Generative -->
        <rect x="880" y="70" width="120" height="290" fill="#1abc9c" opacity="0.3" rx="5"/>
        <text x="940" y="90" text-anchor="middle" font-size="14" font-weight="bold" fill="#333">Generative</text>
        <text x="940" y="110" text-anchor="middle" font-size="12" fill="#666">VAE</text>
        
        <!-- Legend -->
        <text x="500" y="380" text-anchor="middle" font-size="12" fill="#666">Height represents modeling capacity / complexity</text>
    </svg>
</div>

## Key Features

<div class="feature-list">
    <div class="feature-item">
        <h4>🎯 Comprehensive Benchmarking</h4>
        <p>Systematic comparison of seven distinct deep learning architectures on standardized metrics including accuracy, precision, recall, F1-score, training time, and inference time.</p>
    </div>
    <div class="feature-item">
        <h4>📊 Multi-Scale Processing</h4>
        <p>Three-Stage Hierarchical Transformer uniquely processes ECG signals at multiple temporal resolutions (1000, 500, 250 timesteps) for comprehensive pattern recognition.</p>
    </div>
    <div class="feature-item">
        <h4>🔍 Explainable AI</h4>
        <p>Variational Autoencoder provides 21 interpretable latent factors (FactorECG approach) enabling clinical interpretability and generative capabilities.</p>
    </div>
    <div class="feature-item">
        <h4>⚡ Efficiency Optimization</h4>
        <p>CNN model offers optimal balance between accuracy and computational efficiency, making it ideal for practical deployment scenarios.</p>
    </div>
    <div class="feature-item">
        <h4>🧠 Energy-Based Learning</h4>
        <p>Hopfield Network demonstrates unique pattern completion and noise robustness through energy-based associative memory mechanisms.</p>
    </div>
    <div class="feature-item">
        <h4>🔄 Sequential Modeling</h4>
        <p>LSTM network provides bidirectional sequential processing with explicit memory gates for rhythm analysis and temporal pattern recognition.</p>
    </div>
</div>

## Key Findings

<div class="feature-list">
    <div class="feature-item">
        <h4>Accuracy Performance</h4>
        <p><strong>3stageFormer</strong> achieves highest accuracy through multi-scale hierarchical processing. <strong>Transformer</strong> provides excellent accuracy with global attention. <strong>CNN, LSTM, VAE, and Hopfield</strong> offer competitive accuracy with different architectural strengths.</p>
    </div>
    <div class="feature-item">
        <h4>Computational Efficiency</h4>
        <p><strong>FFNN</strong> is fastest for training and inference, ideal for real-time applications. <strong>CNN</strong> provides the best accuracy-efficiency balance. <strong>3stageFormer</strong> is slowest but achieves highest accuracy.</p>
    </div>
    <div class="feature-item">
        <h4>Explainability</h4>
        <p><strong>VAE</strong> offers highest explainability through interpretable latent factors. <strong>Transformer and 3stageFormer</strong> provide attention-based interpretability. <strong>LSTM</strong> offers sequential processing interpretability.</p>
    </div>
    <div class="feature-item">
        <h4>Generalization</h4>
        <p>Models processing raw signals (all except FFNN) demonstrate better generalization. <strong>3stageFormer</strong> excels at multi-scale patterns. <strong>Hopfield</strong> shows superior noise robustness.</p>
    </div>
</div>

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Run Complete Benchmark

```bash
python benchmark.py
```

### Individual Model Testing

```bash
# Feedforward Neural Network
python neural_network.py

# Transformer Model
python transformer_ecg.py

# Three-Stage Hierarchical Transformer
python three_stage_former.py

# CNN and LSTM
python cnn_lstm_ecg.py

# Hopfield Network
python hopfield_ecg.py

# Variational Autoencoder
python vae_ecg.py
```

## Citation

If you use this code or findings, please cite:

```
@article{chandra2025ecg,
  title={Comparative Analysis of Neural Network Architectures for ECG Classification: A Comprehensive Study of Seven Deep Learning Approaches},
  author={Chandra, Shyamal Suhana},
  journal={Sapana Micro Software Research},
  year={2025},
  note={Implementation and benchmarking of FFNN, Transformer, 3stageFormer, CNN, LSTM, Hopfield, and VAE architectures}
}
```

### Related Work Citations

This work builds upon the following foundational research:

- **Feedforward NN:** Lloyd, M. D., et al. (2001). "Detection of Ischemia in the Electrocardiogram Using Artificial Neural Networks." *Circulation*, 103(22), 2711-2716.
- **Transformer:** Ikram, Sunnia, et al. (2025). "Transformer-based ECG classification for early detection of cardiac arrhythmias." *Frontiers in Medicine*, 12, 1600855.
- **3stageFormer:** Tang, Xiaoya, et al. (2024). "Hierarchical Transformer for Electrocardiogram Diagnosis." *arXiv preprint arXiv:2411.00755*.
- **Hopfield Network:** "Electrocardiogram (ECG) Signal Modeling and Noise Reduction Using Hopfield Neural Networks." *Engineering, Technology & Applied Science Research (ETASR)*, Vol. 3, No. 1, 2013. [Link](https://etasr.com/index.php/ETASR/article/view/243/156)
- **VAE (FactorECG):** van de Leur, Rutger R., et al. (2022). "Improving explainability of deep neural network-based electrocardiogram interpretation using variational auto-encoders." *European Heart Journal - Digital Health*, 3(3), 2022. DOI: 10.1093/ehjdh/ztac038. [GitHub](https://github.com/UMCUtrecht-ECGxAI/ecgxai)

## References

- Lloyd, M. D., et al. (2001). "Detection of Ischemia in the Electrocardiogram Using Artificial Neural Networks." *Circulation*, 103(22), 2711-2716.
- Ikram, Sunnia, et al. (2025). "Transformer-based ECG classification for early detection of cardiac arrhythmias." *Frontiers in Medicine*, 12, 1600855.
- Tang, Xiaoya, Berquist, Jake, Steinberg, Benjamin A., and Tasdizen, Tolga. (2024). "Hierarchical Transformer for Electrocardiogram Diagnosis." *arXiv preprint arXiv:2411.00755*.
- "Electrocardiogram (ECG) Signal Modeling and Noise Reduction Using Hopfield Neural Networks." *Engineering, Technology & Applied Science Research (ETASR)*, Vol. 3, No. 1, 2013. [Link](https://etasr.com/index.php/ETASR/article/view/243/156)
- van de Leur, Rutger R., et al. (2022). "Improving explainability of deep neural network-based electrocardiogram interpretation using variational auto-encoders." *European Heart Journal - Digital Health*, 3(3), 2022. DOI: 10.1093/ehjdh/ztac038. [GitHub](https://github.com/UMCUtrecht-ECGxAI/ecgxai)
- Liang, Junbang, et al. (2025). "Video Generators are Robot Policies." *arXiv preprint arXiv:2508.00795*. [Project Page](https://videopolicy.cs.columbia.edu/)
