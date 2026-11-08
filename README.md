# OKHS

Occupation kernel Hilbert space (OKHS) forecasting for fractional dynamical systems. The project includes an RBF OKHS model, a learned kernel with an autoencoder and fractional residual loss, and a classical Kernel DMD baseline.

## Project layout

```text
src/okhs/core/                    OKHS operators, kernels, fractional DMD, neural components
src/okhs/experiments/             Synthetic systems, trajectory generation, training, evaluation
src/okhs/experiments/baselines/   Classical Kernel DMD
results/                          Generated metrics, figures, and TensorBoard logs (ignored by Git)
```

The main benchmark generates seven fractional systems: linear, logistic, quadratic, Lotka–Volterra, pendulum, Van der Pol, and Stuart–Landau. It uses independent training, validation, and test trajectories. The deep model trains an encoder with reconstruction and fractional residual losses, selects its weights using validation forecast MSE, then fits the final OKHS operator with the learned kernel. Test MSE is computed only after the observed context segment.

## Run

Quick functional check:

```powershell
.venv\Scripts\python -m okhs.experiments.benchmark --systems Linear --scale 0.1 --epochs 1 --device cpu --output-dir results/smoke
```

Full comparison, using the original sample sizes and ten training epochs:

```powershell
.venv\Scripts\python -m okhs.experiments.benchmark --noise 0 --output-dir results/benchmark
```

The command uses CUDA if available and falls back to CPU. It fixes the PyTorch seed at 42, fits the three methods on the same training trajectories, and writes `metrics.json`, forecast figures and training curves under `figures/`, and TensorBoard logs under `tensorboard/` in the output directory. Use `--systems Logistic` to run one system. `--scale` reduces the trajectory and time-step counts for a smoke run; scaled results are not directly comparable to a full run. The full Kernel DMD baseline can be computationally expensive because it forms a large snapshot matrix.

On Linux or macOS, replace `.venv\Scripts\python` in the run commands with `.venv/bin/python`.

## Interpretation

Kernel DMD uses a discrete-time spectral model. OKHS fDMD uses fractional occupation kernels and Mittag–Leffler time evolution. Deep OKHS learns the kernel before fitting the final spectral model. The learned residual matrix `W` is used during encoder training, while the final spectral operator is fitted from trajectories after training. Large Gram condition numbers can make forecasts sensitive to numerical choices, so evaluate several seeds and report the distribution of test errors before drawing performance conclusions.
