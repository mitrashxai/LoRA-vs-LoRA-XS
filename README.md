# LoRA vs. LoRA-XS: Comparing Parameter-Efficient Fine-Tuning Approaches

This project compares two parameter-efficient fine-tuning (PEFT) methods, standard **LoRA** and the more compact **LoRA-XS**, on a text classification task, to see how much you can shrink the number of trainable parameters before accuracy starts to suffer.

### Project Highlights

- **Task & Model:** Fine-tuned **DistilBERT-base-uncased** on the **GLUE SST-2** sentiment classification task.
- **Two PEFT Strategies Compared:**
  - **LoRA:** Standard low-rank adaptation, injecting trainable low-rank matrices (A and B) into the attention projection layers.
  - **LoRA-XS:** A more extreme variant that freezes the low-rank A and B matrices entirely and trains only a small r×r matrix between them, drastically cutting the number of trainable parameters.
- **Fair, Controlled Comparison:** Both methods were applied to the same attention layers (`q_lin`, `k_lin`, `v_lin`, `out_lin`), with the same rank (r=8), same scaling factor, and the same training setup, isolating the effect of the PEFT strategy itself.

---

### Key Findings

| Method | Accuracy | Trainable Parameters |
|---|---|---|
| **LoRA-XS** | **0.800** | **594K** |
| LoRA | 0.790 | 887K |

- **LoRA-XS matched — and slightly exceeded — standard LoRA's accuracy while training 33% fewer parameters** (594K vs. 887K). This shows that, at least for a lightweight classification task like SST-2, most of LoRA's adaptation capacity isn't strictly necessary: a much smaller trainable core can capture the needed task-specific signal.
- This result is a small-scale, single-run replication of the core idea behind the LoRA-XS approach (Bałazy et al.), that freezing the outer projection directions and learning only a compact core matrix can preserve most of the accuracy of full LoRA at a fraction of the parameter cost.

---

### Project Structure

- `LoRA_VS_LoRA_XS.ipynb`: The full notebook, data loading, both fine-tuning pipelines (LoRA and LoRA-XS), evaluation, and result visualization.
- `Results/`: Contains the final comparison chart (`comparison_chart.png`).

---

### How to Run

1. **Clone the repository:**

```bash
git clone https://github.com/mitrashxai/LoRA-vs-LoRA-XS.git
cd LoRA-vs-LoRA-XS
```

2. **Install the dependencies:**

```bash
pip install transformers accelerate datasets peft torch loralib evaluate
```

3. **Open and run `LoRA_VS_LoRA_XS.ipynb`** (recommended: Google Colab with a GPU runtime). Running all cells in order will reproduce both fine-tuning runs and generate the final comparison chart.

---

### Notes

- Experiments were run on a subset of SST-2 (1,000 training samples, 1 epoch) for fast iteration, this is a lightweight comparison, not a full-scale benchmark.
- Both methods use the same LoRA rank (r=8) and scaling factor (alpha=8) for a fair, apples-to-apples comparison.
