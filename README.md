# Pretrained-model-Comparison-for-Text-Summarization-using-Topsis

```mermaid
graph TD;
  Start --> LoadData;
  LoadData --> PreprocessText;
  PreprocessText --> ChooseMetrics;
  ChooseMetrics --> ApplyTOPSIS;
  ApplyTOPSIS --> RankModels;
  RankModels --> VisualizeResults;
  VisualizeResults --> End;
```
Overview

Condensing lengthy texts into brief yet meaningful summaries is the goal of text summarization, an important work in natural language processing. The goal of this project is to provide users with the information they need to select the best text summarizing model by comparing their respective performances.

Key Features:

Crucial Elements: Analytical Criteria Used:

Important variables, such as Rouge scores, summary length, and training duration, form the basis of the comparison. While training duration and length reveal information about efficiency and resource needs, rouge scores evaluate the quality of the produced summaries.
Approach - Summary:

To do this comparison, we use the TOPSIS (Technique for Order of Preference by Similarity to Ideal Solution) approach. This approach ranks everything thoroughly by taking into account how close it is to the perfect answer as well as how far it is from the negative ideal solution.
Models Under Review:

As part of the comparison, we include real-world pretrained models including Pegasus, BERTSumExt, GPT-3, T5, XLNet, and BART. Text summarizing tasks make extensive use of these models.


Analysis:
When compared to other models, GPT-3 achieves the highest Rouge scores, indicating superior model performance. Competitive performance is demonstrated by BERTSumExt and T5, which follow closely behind. Effectiveness of Think about it: BERTSumExt needs the fewest resources and trains the fastest. The trade-off between efficiency and Rouge ratings is balanced in BART and T5. What come next? If you would want to get further insights, you can examine the attached CSV files. Based on your unique use case, you might want to think about changing the evaluation measures or introducing additional models. Make use of the project as a springboard for future work on text summarizing.


## Overview

Text summarization is a crucial natural language processing task that involves condensing large documents into concise and informative summaries. This project focuses on comparing the performance of various text summarization models to help users choose the most suitable model for their specific needs.

## Key Features:

1. **Metrics Considered:**
   - The comparison is based on essential metrics, including Rouge scores, length of the summary, and training time. Rouge scores assess the quality of the generated summaries, while length and training time provide insights into efficiency and resource requirements.

2. **Methodology - TOPSIS:**
   - The Technique for Order of Preference by Similarity to Ideal Solution (TOPSIS) method is employed for the comparison. This method considers both the similarity to the ideal solution and the dissimilarity to the negative ideal solution, providing a comprehensive ranking.

3. **Models Evaluated:**
   - Real-world pretrained models, such as BERTSumExt, GPT-3, T5, XLNet, BART, and Pegasus, are included in the comparison. These models are widely used in text summarization tasks.

## Project Structure:

- **`data.csv`**: CSV file containing evaluation metrics for each model.
- **`result.csv`**: CSV file with ranked results in tabular format.

## How to Run:

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/arshiyaakishore/Pretrained-model-Comparison-for-Text-Summarization-using-Topsis.git

## Results and Analysis:
1. **Ranked Table:**
- Explore detailed ranked results in summarization_table_result.csv:

| **Model**   | **Rouge Scores**  | **Length of Summary** | **Training Time** |
|-------------|-------------------|-----------------------|-------------------|
| BERTSumExt  | 0.75              | 130                   | 9                 |
| GPT-3       | 0.82              | 150                   | 12                |
| T5          | 0.78              | 140                   | 10                |
| XLNet       | 0.76              | 135                   | 11                |
| BART        | 0.80              | 145                   | 8                 |
| Pegasus     | 0.79              | 138                   | 13                |



## Analysis:
**Model Performance:**
GPT-3 outperforms other models in terms of Rouge scores, securing the top rank.
BERTSumExt and T5 follow closely, showcasing competitive performance.
Efficiency Consideration:
BERTSumExt is the most resource-efficient, with the lowest training time.
BART and T5 offer a balanced trade-off between Rouge scores and efficiency.
Next Steps:
Feel free to analyze the provided CSV files for more insights.
Consider adjusting the evaluation metrics or adding new models based on your specific use case.
Use the project as a foundation for ongoing research and development in text summarization.


# MIT License

**Copyright (c) [2024] [Yatharth Gautam]**

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

**The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.**

*THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.*
IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

