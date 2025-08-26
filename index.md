---
layout: default
title: FNO Summary
---

<!-- MathJax Configuration -->
<script>
  MathJax = {
    tex: {
      inlineMath: [['$', '$'], ['\\(', '\\)']],
      displayMath: [['$$', '$$'], ['\\[', '\\]']],
      processEscapes: true
    }
  };
</script>
<!-- MathJax Script -->
<script type="text/javascript" async
  src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js">
</script>

# Exploring the Fourier Neural Operator: My Summary and Insights

As part of my journey into machine learning techniques for solving partial differential equations (PDEs), I recently explored the Fourier Neural Operator (FNO) introduced in the paper *Fourier Neural Operator for Parametric Partial Differential Equations*. This approach is a fascinating departure from traditional neural network methods and even other PDE solvers like Physics-Informed Neural Networks (PINNs). In this post, I aim to summarize the key ideas behind FNO in an intuitive way, making it accessible for beginners while avoiding overly complex mathematical details. My goal is to provide a clear understanding of how FNO works and why it’s a significant advancement in solving PDEs. You can also view my easy implementation here. [View my implementation](https://github.com/skato-kx/fno-research.github.io)

## 🔗 Paper Info

- **Title**: [Fourier Neural Operator for Parametric Partial Differential Equations](https://arxiv.org/abs/2010.08895)
- **Authors**: Zongyi Li, Nikola Kovachki, Kamyar Azizzadenesheli, Burigede Liu, Kaushik Bhattacharya, Andrew Stuart, Anima Anandkumar
- **Keywords**: FNO, PDE, Neural Operator, Fourier Transform, Deep Learning

## Traditional Neural Networks for PDEs

In traditional neural networks (NNs), both inputs and outputs are discrete vectors. When solving PDEs, this typically means representing the initial condition of a system, say a function $a(x)$, by sampling its values at specific grid points, such as $a(x_1), a(x_2), \ldots, a(x_n)$. Similarly, the output, such as the solution $u(x)$ after some time $t$, is represented as values at these same grid points, e.g., $u(x_1), u(x_2), \ldots, u(x_n)$. 

For example, consider modeling the temperature in a room. You might divide the room into 1000 grid points, assign initial temperatures (e.g., 30°C near an air conditioner and 5°C near a window), and train the network to predict the temperature at these 1000 points after 30 minutes, say converging to 25°C everywhere. The network learns patterns from many such scenarios. However, this approach has limitations: it cannot generalize to different grid sizes or configurations, nor can it accurately predict outcomes for untrained initial conditions (e.g., 100°C near the air conditioner and -50°C near the window). Additionally, updating the state of all 1000 points at each layer of the network is computationally expensive.

## Neural Operators (NO)

To address these limitations, Neural Operators (NOs) were proposed. Unlike traditional NNs, NOs take functions as inputs and produce functions as outputs, learning the mapping (or “operator”) from an initial condition $a(x)$ to a solution $u(x)$ after time $t$. This allows NOs to handle different grid resolutions and unseen initial conditions. However, NOs still rely on discretizing the space into grid points and computing interactions between them, which involves costly integral operations, making them impractical for many real-world applications.

## Fourier Neural Operator (FNO)

The Fourier Neural Operator (FNO) builds on the concept of NOs but introduces a game-changing approach by leveraging the Fourier transform to make computations efficient and practical. Instead of working in the spatial domain, FNO transforms the input function (e.g., the initial condition $a(x)$) into the frequency domain using the Fourier transform, breaking it down into a combination of sine and cosine waves of different frequencies.

To continue with the room temperature example, the initial condition (30°C near the air conditioner and 5°C near the window) can be expressed as a sum of waves: low-frequency waves capturing the overall uniformity of the temperature and high-frequency waves capturing sharp variations between specific points. Over time, high-frequency components (e.g., sharp temperature differences) tend to diminish, leaving a more uniform state (e.g., 25°C everywhere). FNO learns how to adjust the strength of each frequency component—amplifying, suppressing, or preserving them—based on various training examples. This allows FNO to predict the solution for new, unseen initial conditions by applying the learned frequency adjustments.

### Key Advantages of FNO

- **Grid Independence**: By working with functions directly in the Fourier domain, FNO can handle different grid resolutions (e.g., trained on 1000 grid points but applicable to 2000 grid points) without retraining.
- **Generalization**: Since any initial condition can be expressed as a combination of sine and cosine waves, FNO can generalize to unseen initial conditions by leveraging learned transformations for each frequency component.
- **Computational Efficiency**: FNO replaces costly spatial integrals with simple multiplications in the Fourier domain, significantly reducing computational cost.

### FNO Mechanism

The core idea of FNO is to replace kernel integral operations (used in NOs) with multiplications in the Fourier domain, which is far more efficient. The mathematical formulation is as follows:

$$
(K v_t)(x) = \mathcal{F}^{-1} \left( R_\phi \cdot (\mathcal{F} v_t) \right)(x)
$$

Where:
- $\mathcal{F}$ and $\mathcal{F}^{-1}$ represent the Fourier transform and its inverse, respectively.
- $(\mathcal{F} v_t)(k)$ is the Fourier transform of the input function $v_t$, representing its frequency components.
- $R_\phi(k)$ is a learnable parameter (a filter) for each frequency, determining how much each frequency component is adjusted.

### Implementation Details

- **Fast Fourier Transform (FFT)**: By using FFT, the computational complexity is reduced from $O(N^2)$ to $O(N \log N)$, where $N$ is the number of grid points.
- **Truncation of High Frequencies**: Since high-frequency components are often less relevant, FNO truncates the Fourier modes to a finite number $k_{\max}$, further improving efficiency.
- **Learning Objective**: The network learns how much to retain, amplify, or suppress each frequency component, effectively capturing the operator mapping from input to output functions.

This combination of Fourier transforms, truncation, and learnable filters makes FNO both computationally efficient and highly generalizable, marking a significant advancement over previous methods like PINNs and NOs.
