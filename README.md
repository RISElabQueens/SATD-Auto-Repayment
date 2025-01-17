# SATD-Auto-Repayment

This is the replication package for the manuscript, **“Understanding the Effectiveness of LLMs in Automated Self-Admitted Technical Debt Repayment.”** This paper introduces new benchmarks and evaluation metrics for automated SATD repayment and investigates the effectiveness of large language models (LLMs) in automating technical debt repayment.

## Datasets

In **RQ1**, we used our [SATD Tracker](https://github.com/RISElabQueens/SATD-Tracker) tool to extract SATDs from all Python and Java repositories that have at least 500 commits, 10 contributors, 10 stars, and are not forked. To find GitHub repositories meeting these criteria we employed the [SEART](https://seart-ghs.si.usi.ch/) tool, and identified 14,097 Python and 7,982 Java repositories. The SATD Tracker tool successfully processed 13,748 (97.5%) Python and 7,694 (96.4%) Java repositories. We then applied 10 filtering steps to create a clean dataset for SATD repayment. The final dataset contains:
- **58,722** items for Python, and
- **97,347** items for Java.

Please find the datasets [here](https://drive.google.com/drive/folders/1trZ_JtrFTWD5YIlLb5KbM4nVScSlPUA2?usp=drive_link). For both Python and Java, you will find the following files:

- `df0.pkl.gz`: Contains all extracted SATDs by the SATD Tracker tool (1,607,408 samples for Python and 2,672,485 samples for Java).
- `df1.pkl.gz`: Contains the SATD samples after applying the first two filters.
- `df3.pkl.gz`: Contains the SATD samples after applying all filters except for the last one (filtering by Llama3-70B).

Refer to `RQ1-FilterDataset.ipynb` to download the datasets and apply the filtering steps.

**Note:** The Mastropaolo dataset is used in RQ3.  
- `mastropaolo_with_filter_columns.pkl.gz` contains the additional columns required to apply our new filters.

## Evaluation Metrics

In this paper, we introduce three metrics for evaluating the generated code for SATD repayment:

1. **BLEU-diff**:  
   Applies the BLEU score on two diffs. The first diff is between the input code (which contains the SATD comment) and the ground truth. The second diff is between the input code and the generated code.

2. **CrystalBLEU-diff**:  
   Applies the CrystalBLEU score on the same two diffs described above.

3. **Line-Level Exact Match on Diff (LEMOD)**:  
   Calculates line-level exact-match precision, recall, and F1 score between the two diffs.

Please find their implementation in `SATD_tool.py`.

## Language Models

We study the effectiveness of five state-of-the-art LLMs with zero-shot prompt engineering:

- **Llama-3.1-8B-Instruct**  
- **Llama-3.1-70B-Instruct**  
- **Gemma-2-9B-it**  
- **DeepSeek-Coder-V2-Lite-Instruct**  
- **GPT-4o-mini-2024-07-18**  

We also fine-tune two smaller language models (**CodeT5p-220m** and **CodeT5p-770m**) using our large SATD repayment datasets.

