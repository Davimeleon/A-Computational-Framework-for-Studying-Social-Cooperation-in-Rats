# A Computational Framework for Studying Social Cooperation in Rats

**David Backer Peral**  
Wu Tsai Undergraduate Research Fellowship, Summer 2025  
[Yale University / Wu Tsai Institute](https://wti.yale.edu)

---

## 🧠 Overview

Understanding the neural mechanisms that drive **social cooperation** is essential for advancing our knowledge of behavior and for developing treatments for conditions such as **autism spectrum disorder** and **early-life stress**. This project explores social behavior in rats through both **behavioral analysis** and **computational modeling**.

While **reinforcement learning (RL)** has become a powerful framework for understanding individual animal behavior, **multi-agent RL** has not yet been fully leveraged to model social cooperation. Our work seeks to bridge that gap by combining pose-estimated behavior data, video analysis, and multi-agent learning models to uncover underlying strategies and neural correlates.

---

## 🧪 Abstract

Animals such as rats can model the mechanisms of social cooperation, though their behavioral variability makes analysis inherently challenging. To address this, we analyzed the behavior of pairs of rats performing a cooperative task, quantified using pose estimation methods, to identify strategies and behaviors predicting success.

We observed two primary cooperative strategies:

- 🕒 **Waiting for the partner**
- 🧍‍↔️ **Synchronized movement**

And two types of social interaction that influenced cooperation:

- 👀 **Gazing**
- 🐀 **Physical interaction**

These behaviors were analyzed across groups varying in **training history**, **social familiarity**, and **environmental visibility**. Notably, we found that:

- **Gazing increased** in less familiar pairs.
- **Gazing decreased** during successful trials.

This suggests gaze acts more as a mechanism for **social recognition** than **real-time coordination**.

To further probe the cognitive mechanisms, we are training a model using the **Multi-Agent Deep Deterministic Policy Gradient (MADDPG)** algorithm, aligning model outputs with observed rat behavior and neural activity to derive biologically grounded insights into cooperation and social learning.

---

## 📁 Repository Contents

This repository includes:

- 🎞️ `videos/` – Sample behavioral videos from experimental trials  
- 📊 `poster/` – Final research poster presented at the Wu Tsai Summer Symposium  
- 📄 `analysis/` – Code and data for behavior and pose-based metrics  
- 🤖 `modeling/` – Multi-agent RL training and analysis using MADDPG  
- 🧠 `fiber_photometry/` – (Optional) Neural data used to align with behavioral moments  
- 📌 `README.md` – Project overview and abstract

---

## 🧰 Tools & Methods

- **Pose Estimation** – DeepLabCut for frame-by-frame body tracking  
- **Behavioral Quantification** – Custom Python tools for computing gaze, coordination, latency, and synchrony  
- **Reinforcement Learning** – MADDPG model implemented using PyTorch  
- **Neural Analysis** – Fiber photometry recordings from ACC→BLA and ACC→AIC pathways  
- **Statistical Tools** – NumPy, Pandas, Matplotlib, Prism

---

## 📬 Contact

If you have questions or want to collaborate, feel free to reach out:

📧 david.backerperal@yale.edu  
🧠 Wu Tsai Institute: [https://wti.yale.edu](https://wti.yale.edu)

---

## 📜 Acknowledgments

This research was conducted as part of the **Wu Tsai Undergraduate Research Fellowship**. Special thanks to [your mentor/PI's name] and the [lab name] at Yale for their support and guidance.

---

## 📝 Citation

If you use or reference this work, please cite:

> Backer Peral, D. (2025). *A Computational Framework for Studying Social Cooperation in Rats*. Wu Tsai Undergraduate Research Fellowship, Yale University.

