# A Computational Framework for Studying Social Cooperation in Rats

**David Backer Peral**  
Wu Tsai Undergraduate Research Fellowship, Summer 2025  
[Yale University / Wu Tsai Institute](https://wti.yale.edu)

---

## 🧪 Abstract

Understanding the neural mechanisms that drive social cooperation is crucial for advancing our knowledge of social behavior and developing treatments for conditions such as autism spectrum disorder and early-life stress. Animals such as rats can model these mechanisms, though their behavioral variability makes analysis inherently challenging. While reinforcement learning (RL) has emerged as a promising framework for modeling animal behavior, multi-agent RL algorithms have not yet been applied to the study of social cooperation. Such models could offer insight into the neural computations underlying these behaviors.
	
 To address this gap, we analyzed the behavior of pairs of rats performing a cooperative task, quantified using pose estimation methods, to identify strategies and behaviors predicting success. Specifically, we observed two primary strategies: (a) waiting for the partner and (b) synchronized movement. We also identified two social interactions that influenced cooperation: gazing and physical interaction. These behavioral metrics were compared across rat cohorts varying in training history, social familiarity, and environmental visibility. Notably, we found that gazing increases in pairs with lower social familiarity and decreases during successful trials, suggesting that gaze serves primarily as a mechanism for social recognition rather than real-time coordination. 
	
 To better understand the underlying cognitive mechanisms, we are training a model using the Multi-Agent Deep Deterministic Policy Gradient (MADDPG) algorithm, which leverages centralized critics to guide decentralized decision-making. By aligning the model’s learned strategies with observed behavior and neural recordings during key cooperative moments, we aim to uncover interpretable links between computational principles and brain function. This approach will not only test the validity of RL-based models in social contexts but also provide a biologically grounded framework for understanding how cooperation is represented in the brain. Ultimately, our findings may offer novel insights into how disruptions in social learning circuits contribute to psychiatric conditions marked by social deficits.

---

## 📁 Repository Contents

This repository includes:

- 🎞️ `Videos/` – Sample behavioral and model videos showing distinct behaviors and strategies. 
- 📊 `Poster/` – Final research poster presented at the Wu Tsai Summer Symposium  
- 📄 `Behavioral_Analysis_Code/` – Code and data for all experimental and pose-based metrics  
- 🤖 `Modeling_Code/` – Multi-agent RL training and analysis using MADDPG  
- 📌 `README.md` – Project overview and abstract

---

## 🧰 Tools & Methods

- **Pose Estimation** – SLEAP for frame-by-frame body tracking  
- **Behavioral Quantification** – Custom Python tools for computing gaze, coordination, latency, and synchrony  
- **Reinforcement Learning** – MADDPG model implemented using PyTorch  
- **Neural Analysis** – Fiber photometry recordings from ACC→BLA and ACC→AIC pathways  
- **Statistical Tools** – NumPy, Pandas, Matplotlib, Prism

---

## 📬 Contact

If you have questions, feel free to reach out:

📧 david.backerperal@yale.edu  
🧠 Wu Tsai Institute: [https://wti.yale.edu](https://wti.yale.edu)

---

## 📜 Acknowledgments

This research was conducted as part of the **Wu Tsai Undergraduate Research Fellowship**. Special thanks to Amelia Johnson and the Saxena Lab at Yale for their support and guidance.

