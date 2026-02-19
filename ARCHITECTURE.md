# Architecture: shrew-optim

`shrew-optim` provides optimization algorithms for training neural networks. It implements standard optimizers and learning rate schedulers compatible with `shrew-core` tensors and `shrew-nn` parameters.

## Core Concepts

- **Optimizer Trait**: Defines the interface for applying gradients to parameters (`step`, `zero_grad`).
- **Momentum & Adaptive Learning**: Implementations like Adam and RMSprop statefully track momentum and variance of gradients.
- **Schedulers**: Mechanism to dynamically adjust the learning rate during training (e.g., warm-up, cosine decay).
- **Gradient Clipping**: Utilities to prevent exploding gradients (`clip_grad_norm`).

## File Structure

| File | Description | Lines of Code |
| :--- | :--- | :--- |
| `scheduler.rs` | Implements learning rate schedulers (`StepLR`, `CosineAnnealingLR`, `LinearLR`, `ConstantLR` with warmup capability). | 413 |
| `adam.rs` | Implementation of the Adam optimizer (Adaptive Moment Estimation), the industry standard for most tasks. | 298 |
| `radam.rs` | Rectified Adam implementation, providing more stable training at the start without intense warmup. | 240 |
| `clip.rs` | Utilities for gradient clipping by norm or value, essential for RNN/Transformer stability. | 220 |
| `rmsprop.rs` | RMSprop optimizer, often used for RNNs. | 206 |
| `sgd.rs` | Stochastic Gradient Descent (SGD) with optional momentum and Nesterov acceleration. | 151 |
| `ema.rs` | Exponential Moving Average (EMA) of model parameters, used for smoother validation and inference models. | 142 |
| `optimizer.rs` | Defines the `Optimizer` trait and common shared logic. | 102 |
| `lib.rs` | Crate root. | 37 |
