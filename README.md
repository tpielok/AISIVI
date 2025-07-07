# AISIVI Revisiting Unbiased Implicit Variational Inference

[![Poster PDF](https://img.shields.io/badge/poster-download-blue)](./poster.pdf)
[![ArXiv](https://img.shields.io/badge/arXiv-paper-green)](https://[arxiv.org/abs/XXXX.XXXXX](https://www.arxiv.org/abs/2506.03839))

This repository contains the code, experiments, and poster for our ICML 2025 contribution **"Revisiting Unbiased Implicit Variational Inference"**, where we address fundamental challenges in training semi-implicit variational distributions via efficient and theoretically justified score estimation.

---

## 🧠 Motivation

**Semi-Implicit Variational Inference (SIVI)** enables expressive variational approximations by defining a marginal
\[
q(z) = \mathbb{E}_{\epsilon \sim p(\epsilon)}[q(z|\epsilon)]
\]
However, this marginal is **intractable**, making gradient-based optimization of common divergence objectives (e.g., KL) difficult.

Previous solutions like **UIVI** rely on inner-loop MCMC - expensive and brittle in high dimensions.

---

## ✨ Our Contribution: AISIVI

We propose **AISIVI**, an efficient, path-gradient-compatible alternative for training SIVI objectives using:

- ✅ **Unbiased and consistent score estimation** via **importance sampling**
- ✅ A learned **reverse conditional** proposal \(\tau(\epsilon|z)\)
- ✅ **Low bias & low variance** through proper proposal training