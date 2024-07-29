# NutriNudge Evaluations

## Overview
This project focuses on evaluating the performance and latency of our Retrieval-Augmented Generation (RAG) application: **NutriNudge**. 
We implemented and evaluated several methods to enhance the system's performance.

## Requirements

Evaluation data: All the datasets like Ground Truth and Irrelevant or Noisy queries used in the evaluation notebook are present in the Data folder. 
Evaluation LLM: Phi3 is used for evaluation as we have used Llama3 to develop our application. Make sure to run the following after you have installed Ollama:
```
Ollama pull phi3
```
## Detailed Report & Demonstration
* For a comprehensive analysis of the methodology, improvements, results, and future work, please refer to our [NutriNudge Evaluation Report](https://github.com/chakraborty-arnab/NutriNudge/blob/main/Evaluation/NutriNudge%20Evaluation%20Report.pdf).
* Video Demonstration: [NutriNudge Evaluation]()

## Key Improvements
1. Modified prompt for concise answers
2. Implemented threshold-based retrieval

## Results
Our improvements led to:
- Increased accuracy and relevance scores
- Significant reduction in latency, especially for irrelevant queries


