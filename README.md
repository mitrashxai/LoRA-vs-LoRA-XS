# LoRA-vs-LoRA-XS
Experiments comparing LoRA and LoRA-XS for efficient fine-tuning of large language models, focusing on accuracy vs. parameter efficiency.
## 🔎 Overview
- **LoRA**: Low-Rank Adaptation that injects small trainable rank-decomposition matrices into transformer layers for parameter-efficient fine-tuning.  
- **LoRA-XS**: A lighter variant designed to reduce parameters and computation even further, while preserving competitive performance.  

This project explores:
- Using 🤗 `datasets` for data loading  
- Applying LoRA and LoRA-XS with `peft` and `loralib`  
- Training and evaluating transformer models  
- Comparing results across accuracy, loss, and efficiency metrics  

---

## ⚙️ Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/<your-username>/LoRA-vs-LoRA-XS.git
cd LoRA-vs-LoRA-XS
pip install -r requirements.txt
