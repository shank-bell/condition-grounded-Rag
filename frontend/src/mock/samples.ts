// Real responses of the running backend (captured 9 Oct 2026), used ONLY by the dev mock (?mock=1). Not part of the app logic.
import type { QueryResponse } from "../types";

export const SAMPLE_XLMR_KANNADA: QueryResponse = {
 "question": "What accuracy does XLM-R get on XNLI for Kannada?",
 "answer": "The provided sources do not contain a specific accuracy result for the XLM-R model on the XNLI dataset for the Kannada language.\n\nThe sources report the following general results for XLM-R on XNLI:\n- XLM-R obtains 80.9% accuracy on XNLI for cross-lingual transfer [1].\n- XLM-R obtains 83.6% accuracy on XNLI when using multilingual training (translate-train-all) [1].\n- XLM-R (N models, 24 layers, 1024 hidden states) obtains 71.3% accuracy on XNLI, while a single model (24 layers, 1024 hidden states) obtains 70.7% [4].\n\nNote: The evidence does not cover the specific condition of language = Kannada for the model XLM-R on the XNLI dataset.",
 "sources": [
  {
   "chunk_id": "1911.02116:0016",
   "paper_id": "1911.02116",
   "paper_title": "Unsupervised Cross-lingual Representation Learning at Scale",
   "page": 6,
   "section": "results",
   "score": 6.788,
   "preview": "XLM-R sets a new state of the art on XNLI. On cross-lingual transfer, XLM-R obtains 80.9% accuracy, outperforming the XLM-100 and mBERT open-source models by 10.2% and 14.6% average accuracy. On the Swahili and Urdu lowresource languages, XLM-R outperforms XLM-100 by 15.7% and 11..."
  },
  {
   "chunk_id": "1911.02116:0001",
   "paper_id": "1911.02116",
   "paper_title": "Unsupervised Cross-lingual Representation Learning at Scale",
   "page": 1,
   "section": "abstract",
   "score": 5.188,
   "preview": "Abstract This paper shows that pretraining multilingual language models at scale leads to significant performance gains for a wide range of crosslingual transfer tasks. We train a Transformerbased masked language model on one hundred languages, using more than two terabytes of fi..."
  },
  {
   "chunk_id": "1911.02116:0011",
   "paper_id": "1911.02116",
   "paper_title": "Unsupervised Cross-lingual Representation Learning at Scale",
   "page": 4,
   "section": "results",
   "score": 3.548,
   "preview": "We illustrate this trade-off in Figure 2, which shows XNLI performance vs the number of languages the model is pretrained on. Initially, as we go from 7 to 15 languages, the model is able to take advantage of positive transfer which improves performance, especially on low resourc..."
  },
  {
   "chunk_id": "1911.02116:0015",
   "paper_id": "1911.02116",
   "paper_title": "Unsupervised Cross-lingual Representation Learning at Scale",
   "page": 6,
   "section": "results",
   "score": 3.284,
   "preview": "5.2 Cross-lingual Understanding Results Based on these results, we adapt the setting of Lample and Conneau (2019) and use a large Transformer model with 24 layers and 1024 hidden states, with a 250k vocabulary. We use the multilingual MLM loss and train our XLM-R model for 1.5 Mi..."
  },
  {
   "chunk_id": "1911.02116:0024",
   "paper_id": "1911.02116",
   "paper_title": "Unsupervised Cross-lingual Representation Learning at Scale",
   "page": 8,
   "section": "results",
   "score": 3.217,
   "preview": "XNLI: XLM versus BERT. A recurrent criticism against multilingual models is that they obtain worse performance than their monolingual counterparts. In addition to the comparison of XLM-R and RoBERTa, we provide the first comprehensive study to assess this claim on the XNLI benchm..."
  }
 ],
 "analysis": {
  "intent": "result",
  "complexity": "simple",
  "conditions": {
   "task": null,
   "dataset": "XNLI",
   "dataset_version": null,
   "language": "Kannada",
   "model": "XLM-R",
   "model_size": null,
   "setting": null,
   "other_models": [],
   "other_datasets": [],
   "other_languages": []
  }
 },
 "applicability": {
  "coverage": 0.6666666666666666,
  "checks": [
   {
    "condition": "dataset",
    "requested": "XNLI",
    "covered": true,
    "observed": [
     "XNLI",
     "CoNLL-2002",
     "MLQA",
     "GLUE",
     "MNLI-m",
     "MNLI-mm"
    ],
    "chunk_ids": [
     "1911.02116:0016",
     "1911.02116:0001",
     "1911.02116:0011",
     "1911.02116:0015",
     "1911.02116:0024",
     "1911.02116:0003",
     "1911.02116:0013",
     "2010.11934:0028",
     "1911.02116:0017",
     "1901.07291:0014"
    ]
   },
   {
    "condition": "language",
    "requested": "Kannada",
    "covered": false,
    "observed": [
     "English",
     "Spanish",
     "French",
     "German",
     "Bulgarian",
     "Russian"
    ],
    "chunk_ids": []
   },
   {
    "condition": "model",
    "requested": "XLM-R",
    "covered": true,
    "observed": [
     "XLM-R",
     "Devlin et al.(2018)",
     "Lample and Conneau(2019)",
     "BERT-en",
     "RoBERTa",
     "Lample and Conneau (2019)"
    ],
    "chunk_ids": [
     "1911.02116:0016",
     "1911.02116:0001",
     "1911.02116:0011",
     "1911.02116:0015",
     "1911.02116:0003",
     "1911.02116:0013",
     "2010.11934:0028",
     "1911.02116:0017"
    ]
   }
  ],
  "missing": [
   "language=Kannada"
  ],
  "re_retrieved": true,
  "warning": "The evidence covers 2 of 3 conditions in the question. No source reports language = Kannada together with model = XLM-R, dataset = XNLI. For model = XLM-R, dataset = XNLI, the sources record: English, Spanish, French, German, Bulgarian, Russian. Each condition appears in the sources, but only in different combinations; do not assume the findings hold for this one.",
  "reasoning": "language=Kannada: The evidence lists several languages for XNLI and XLM-R (e.g., Hindi, Arabic, Thai), but Kannada is not mentioned in any of the provided snippets for this model/dataset combination or others. search_query: XLM-R performance on XNLI dataset for Kannada language Joint check: no source records model = XLM-R, dataset = XNLI, language = Kannada together.",
  "profile_guided": false,
  "joint_covered": false,
  "escalated": true
 },
 "contradictions": [],
 "claim_checks": [
  {
   "sentence": "- XLM-R obtains 80.9% accuracy on XNLI for cross-lingual transfer [1].",
   "supported": true,
   "chunk_id": "1911.02116:0016",
   "entailment": 0.9937922358512878
  },
  {
   "sentence": "- XLM-R obtains 83.6% accuracy on XNLI when using multilingual training (translate-train-all) [1].",
   "supported": true,
   "chunk_id": "1911.02116:0016",
   "entailment": 0.7344068288803101
  },
  {
   "sentence": "- XLM-R (N models, 24 layers, 1024 hidden states) obtains 71.3% accuracy on XNLI, while a single model (24 layers, 1024 hidden states) obtains 70.7% [4].",
   "supported": true,
   "chunk_id": "1911.02116:0015",
   "entailment": 0.9411815404891968
  }
 ],
 "regenerated": true,
 "retrieval_weak": false,
 "trace": [
  "1 intent=result complexity=simple; conditions={'dataset': 'XNLI', 'language': 'Kannada', 'model': 'XLM-R'}",
  "2 planner agent: The question asks for a specific numerical result from a specific setup.",
  "2 refinement: rewrite",
  "2 agent added refinement (the question was judged vague or multi-part)",
  "3 queries: ['What is the accuracy score of the XLM-R model on the XNLI dataset for the Kannada language?']",
  "4 retrieved 20 -> 5 kept 10",
  "6 coverage 0.67, missing ['language=Kannada'], re-retrieved, NOT recorded together, second opinion by gemma4:12b | agent: language=Kannada: The evidence lists several languages for XNLI and XLM-R (e.g., Hindi, Arabic, Thai), but Kannada is not mentioned in any of the provided snippets for this model/dataset combination or others. search_query: XLM-R performance on XNLI dataset for Kannada language Joint check: no source records model = XLM-R, dataset = XNLI, language = Kannada together.",
  "7 conflicts: none",
  "9 claims checked 3, unsupported 0, regenerated once"
 ],
 "timings_ms": {
  "1_understand": 1570.7,
  "2_plan": 373.6,
  "3_refine": 800.8,
  "4_retrieve": 34.1,
  "5_rerank": 36.6,
  "2b_replan": 0,
  "6_applicability": 3885.1,
  "7_contradiction": 1655.1,
  "8_generate": 2529.8,
  "9_critic": 3644.1,
  "total": 14531.2
 }
};

export const SAMPLE_BERT_SQUAD: QueryResponse = {
 "question": "What F1 does BERT-large get on SQuAD v2.0?",
 "answer": "BERT-large achieves different F1 scores on SQuAD v2.0 depending on the evaluation setting and specific dataset split:\n\n*   **86.9 F1**: Obtained on SQuAD 2.0 (fine-tuned, English, large) [2].\n*   **82.8 F1**: Obtained on SQuAD 2.0 (test set, English, large) for the \"BERT-Large (Best of 3)\" variant [3].\n*   **81.8 F1**: Obtained on SQuAD 2.0 (single model, English, large) [5].\n*   **89.1 F1**: Obtained on SQuAD 2.0 (test set, English, large) [5].\n*   **81.9 F1**: Obtained on SQuAD 2.0 (single, English, large) [6].\n*   **83.1 F1**: Obtained on SQuAD 2.0 (test set, English, large) for the \"BERTLARGE (Single)\" model [6].",
 "sources": [
  {
   "chunk_id": "1909.11942:0004",
   "paper_id": "1909.11942",
   "paper_title": "ALBERT: A LITE BERT FOR SELF-SUPERVISED LEARNING OF LANGUAGE REPRESENTATIONS",
   "page": 2,
   "section": "introduction",
   "score": 4.772,
   "preview": "As a result of these design decisions, we are able to scale up to much larger ALBERT configurations that still have fewer parameters than BERT-large but achieve significantly better performance. We establish new state-of-the-art results on the well-known GLUE, SQuAD, and RACE ben..."
  },
  {
   "chunk_id": "1909.11942:0026",
   "paper_id": "1909.11942",
   "paper_title": "ALBERT: A LITE BERT FOR SELF-SUPERVISED LEARNING OF LANGUAGE REPRESENTATIONS",
   "page": 8,
   "section": "results",
   "score": 4.546,
   "preview": "Table 6: The effect of controlling for training time, BERT-large vs ALBERT-xxlarge configurations. | Models | Steps | Time | SQuAD1.1 | SQuAD2.0 | MNLI | SST-2 | RACE | Avg | | BERT-large | 400k | 34h | 93.5/87.4 | 86.9/84.3 | 87.8 | 94.6 | 77.3 | 87.2 | | ALBERT-xxlarge | 125k |..."
  },
  {
   "chunk_id": "1906.08237:0018",
   "paper_id": "1906.08237",
   "paper_title": "XLNet: Generalized Autoregressive Pretraining for Language Understanding",
   "page": 7,
   "section": "experiments",
   "score": 4.43,
   "preview": "Table 1: Fair comparison with BERT. All models are trained using the same data and hyperparameters as in BERT. We use the best of 3 BERT variants for comparison; i.e., the original BERT, BERT with whole word masking, and BERT without next sentence prediction. | Model | SQuAD1.1 |..."
  },
  {
   "chunk_id": "1810.04805:0020",
   "paper_id": "1810.04805",
   "paper_title": "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
   "page": 7,
   "section": "experiments",
   "score": 4.276,
   "preview": "tuning data, we only lose 0.1-0.4 F1, still outperforming all existing systems by a wide margin.12 4.3 SQuAD v2.0 The SQuAD 2.0 task extends the SQuAD 1.1 problem definition by allowing for the possibility that no short answer exists in the provided paragraph, making the problem ..."
  },
  {
   "chunk_id": "1909.11942:0034",
   "paper_id": "1909.11942",
   "paper_title": "ALBERT: A LITE BERT FOR SELF-SUPERVISED LEARNING OF LANGUAGE REPRESENTATIONS",
   "page": 10,
   "section": "results",
   "score": 3.892,
   "preview": "Table 10: State-of-the-art results on the SQuAD and RACE benchmarks. | Models | SQuAD1.1 dev | SQuAD2.0 dev | SQuAD2.0 test | RACE test(Middle/High) | | Single model (from leaderboa | rd as of Sept. 23, | 2019) | | | | BERT-large | 90.9/84.1 | 81.8/79.0 | 89.1/86.3 | 72.0 (76.6/7..."
  },
  {
   "chunk_id": "1810.04805:0019",
   "paper_id": "1810.04805",
   "paper_title": "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
   "page": 7,
   "section": "experiments",
   "score": 2.512,
   "preview": "Table 3: SQuAD 2.0 results. We exclude entries that use BERT as one of their components. | System | D | ev | Te | st | | | EM | F1 | EM | F1 | | Top Leaderboard Systems | (Dec | 10th, | 2018) | | | Human | 86.3 | 89.0 | 86.9 | 89.5 | | #1 Single - MIR-MRC (F-Net) | - | - | 74.8 |..."
  }
 ],
 "analysis": {
  "intent": "result",
  "complexity": "simple",
  "conditions": {
   "task": null,
   "dataset": "SQuAD",
   "dataset_version": "2.0",
   "language": null,
   "model": "BERT-large",
   "model_size": null,
   "setting": null,
   "other_models": [],
   "other_datasets": [],
   "other_languages": []
  }
 },
 "applicability": {
  "coverage": 1,
  "checks": [
   {
    "condition": "dataset",
    "requested": "SQuAD",
    "covered": true,
    "observed": [
     "SQuAD",
     "RACE",
     "MNLI",
     "SST-2",
     "SQuAD1.1",
     "SQuAD2.0"
    ],
    "chunk_ids": [
     "1909.11942:0004",
     "1909.11942:0026",
     "1906.08237:0018",
     "1909.11942:0034",
     "1907.10529:0016",
     "1907.10529:0017",
     "1907.11692:0027",
     "1810.04805:0019",
     "2004.02984:0015"
    ]
   },
   {
    "condition": "dataset_version",
    "requested": "2.0",
    "covered": true,
    "observed": [
     "1.1",
     "2.0"
    ],
    "chunk_ids": [
     "1909.11942:0004",
     "1909.11942:0026",
     "1909.11942:0034",
     "1907.10529:0016",
     "1907.11692:0027",
     "1810.04805:0019"
    ]
   },
   {
    "condition": "model",
    "requested": "BERT-large",
    "covered": true,
    "observed": [
     "BERT-large",
     "XLNet",
     "RoBERTa",
     "ALBERT-xxlarge",
     "BERT",
     "ALBERT"
    ],
    "chunk_ids": [
     "1909.11942:0004",
     "1909.11942:0026",
     "1810.04805:0020",
     "1909.11942:0034",
     "1907.10529:0016",
     "1907.11692:0027",
     "1810.04805:0019"
    ]
   }
  ],
  "missing": [],
  "re_retrieved": false,
  "warning": null,
  "reasoning": "",
  "profile_guided": false,
  "joint_covered": true,
  "escalated": false
 },
 "contradictions": [
  {
   "verdict": "EXPLAINED",
   "chunk_a": "1909.11942:0034",
   "chunk_b": "1810.04805:0019",
   "paper_a": "1909.11942",
   "paper_b": "1810.04805",
   "metric": "F1",
   "value_a": 90.9,
   "value_b": 81.9,
   "differing": [
    "dataset_version"
   ],
   "reason": "Explained difference (90.9 vs 81.9 F1): dataset version (1.1 vs 2.0).",
   "nli_contradiction": 0.9950518012046814
  },
  {
   "verdict": "EXPLAINED",
   "chunk_a": "1909.11942:0026",
   "chunk_b": "1810.04805:0019",
   "paper_a": "1909.11942",
   "paper_b": "1810.04805",
   "metric": "F1",
   "value_a": 93.5,
   "value_b": 81.9,
   "differing": [
    "dataset_version",
    "setting"
   ],
   "reason": "Explained difference (93.5 vs 81.9 F1): dataset version (1.1 vs 2.0); evaluation setting (fine-tuned vs single).",
   "nli_contradiction": 0.9252520203590393
  }
 ],
 "claim_checks": [
  {
   "sentence": "*   **86.9 F1**: Obtained on SQuAD 2.0 (fine-tuned, English, large) [2].",
   "supported": true,
   "chunk_id": "1909.11942:0026",
   "entailment": 0.9855331778526306
  },
  {
   "sentence": "*   **82.8 F1**: Obtained on SQuAD 2.0 (test set, English, large) for the \"BERT-Large (Best of 3)\" variant [3].",
   "supported": true,
   "chunk_id": "1906.08237:0018",
   "entailment": 0.9897866249084473
  },
  {
   "sentence": "*   **81.8 F1**: Obtained on SQuAD 2.0 (single model, English, large) [5].",
   "supported": true,
   "chunk_id": "1909.11942:0034",
   "entailment": 0.8903571963310242
  },
  {
   "sentence": "*   **89.1 F1**: Obtained on SQuAD 2.0 (test set, English, large) [5].",
   "supported": false,
   "chunk_id": "1909.11942:0034",
   "entailment": 0.011661499738693237
  },
  {
   "sentence": "*   **81.9 F1**: Obtained on SQuAD 2.0 (single, English, large) [6].",
   "supported": true,
   "chunk_id": "1810.04805:0019",
   "entailment": 0.9830198884010315
  },
  {
   "sentence": "*   **83.1 F1**: Obtained on SQuAD 2.0 (test set, English, large) for the \"BERTLARGE (Single)\" model [6].",
   "supported": true,
   "chunk_id": "1810.04805:0019",
   "entailment": 0.05428159981966019
  }
 ],
 "regenerated": true,
 "retrieval_weak": false,
 "trace": [
  "1 intent=result complexity=simple; conditions={'dataset': 'SQuAD', 'dataset_version': '2.0', 'model': 'BERT-large'}",
  "2 planner agent: The question asks for a specific numerical result from a known benchmark.",
  "2 refinement: rewrite",
  "2 agent added refinement (the question was judged vague or multi-part)",
  "3 queries: ['What is the F1 score achieved by the BERT-large model on the SQuAD v2.0 dataset?']",
  "4 retrieved 20 -> 5 kept 10",
  "6 coverage 1.00, missing none, conditions recorded together",
  "7 conflicts: EXPLAINED, EXPLAINED",
  "9 claims checked 6, unsupported 1, regenerated once"
 ],
 "timings_ms": {
  "1_understand": 1215,
  "2_plan": 378.8,
  "3_refine": 825.7,
  "4_retrieve": 31.9,
  "5_rerank": 34.7,
  "2b_replan": 0,
  "6_applicability": 187.5,
  "7_contradiction": 96.4,
  "8_generate": 5158.8,
  "9_critic": 4236.5,
  "total": 12167
 }
};

export const SAMPLE_KANNADA_NLI: QueryResponse = {
 "question": "How well do models perform on Kannada NLI?",
 "answer": "Several models show varying performance on the Kannada NLI task (IndicXNLI):\n\n*   **IndicBERT+Samanantar** achieves 74.7 accuracy on the IndicXNLI test set for Kannada [2, 3].\n*   **IndicBERT** (278M) achieves 73.8 accuracy on the IndicXNLI test set for Kannada [3].\n*   **MuRIL** achieves 74.0 accuracy on the IndicXNLI task for Kannada [3].\n*   **XLMR** achieves 71.5 accuracy on the IndicXNLI task for Kannada [3].\n*   **v1-data** achieves 69.6 accuracy on the IndicXNLI task for Kannada [3].\n*   **mBERT** achieves 58.6 accuracy on the IndicXNLI task for Kannada [3].",
 "sources": [
  {
   "chunk_id": "2005.14165:0051",
   "paper_id": "2005.14165",
   "paper_title": "Language Models are Few-Shot Learners",
   "page": 20,
   "section": "results",
   "score": -2.216,
   "preview": "20 Figure 3.9: Performance of GPT-3 on ANLI Round 3. Results are on the dev-set, which has only 1500 examples and therefore has high variance (we estimate a standard deviation of 1.2%). We find that smaller models hover around random chance, while few-shot GPT-3 175B closes almos..."
  },
  {
   "chunk_id": "2212.05409:0044",
   "paper_id": "2212.05409",
   "paper_title": "Towards Leaving No Indic Language Behind: Building Monolingual Corpora, Benchmark and Models for Indic Languages",
   "page": 16,
   "section": "other",
   "score": -3.766,
   "preview": "Table 10: Scores for IndicBERT+Samanantar model on the IndicXNLI proposed by Aggarwal et al. (2022) (Org.) & current state of verified dataset (HV ) | Lang. | Org. | HV | Lang. | Org. | HV | | as | 71.6 | 72.0 | mr | 73.2 | 73.5 | | bn | 76.3 | 76.5 | or | 74.0 | 73.5 | | gu | 75..."
  },
  {
   "chunk_id": "2212.05409:0054",
   "paper_id": "2212.05409",
   "paper_title": "Towards Leaving No Indic Language Behind: Building Monolingual Corpora, Benchmark and Models for Indic Languages",
   "page": 20,
   "section": "other",
   "score": -3.932,
   "preview": "Table 16: Results on IndicXNLI task. Metric: accuracy. | | as | bn | gu | hi | kn | ml | mr | or | pa | ta | te | ur | Avg. | | mBERT | 46.4 | 59.5 | 56.1 | 63.9 | 58.6 | 55.0 | 54.3 | 34.0 | 58.8 | 57.3 | 56.0 | 56.7 | 54.7 | | XLMR | 63.5 | 70.7 | 70.5 | 75.2 | 71.5 | 71.3 | 69..."
  }
 ],
 "analysis": {
  "intent": "result",
  "complexity": "simple",
  "conditions": {
   "task": null,
   "dataset": null,
   "dataset_version": null,
   "language": "Kannada",
   "model": null,
   "model_size": null,
   "setting": null,
   "other_models": [],
   "other_datasets": [],
   "other_languages": []
  }
 },
 "applicability": {
  "coverage": 1,
  "checks": [
   {
    "condition": "language",
    "requested": "Kannada",
    "covered": true,
    "observed": [
     "English",
     "assamese",
     "marathi",
     "bengali",
     "oriya",
     "gujarati"
    ],
    "chunk_ids": [
     "2212.05409:0044",
     "2212.05409:0054"
    ]
   }
  ],
  "missing": [],
  "re_retrieved": false,
  "warning": null,
  "reasoning": "",
  "profile_guided": false,
  "joint_covered": null,
  "escalated": false
 },
 "contradictions": [],
 "claim_checks": [
  {
   "sentence": "*   **IndicBERT+Samanantar** achieves 74.7 accuracy on the IndicXNLI test set for Kannada [2, 3].",
   "supported": true,
   "chunk_id": "2212.05409:0044",
   "entailment": 0.9926465749740601
  },
  {
   "sentence": "*   **IndicBERT** (278M) achieves 73.8 accuracy on the IndicXNLI test set for Kannada [3].",
   "supported": true,
   "chunk_id": "2212.05409:0054",
   "entailment": 0.46404778957366943
  },
  {
   "sentence": "*   **MuRIL** achieves 74.0 accuracy on the IndicXNLI task for Kannada [3].",
   "supported": true,
   "chunk_id": "2212.05409:0054",
   "entailment": 0.017876800149679184
  },
  {
   "sentence": "*   **XLMR** achieves 71.5 accuracy on the IndicXNLI task for Kannada [3].",
   "supported": true,
   "chunk_id": "2212.05409:0054",
   "entailment": 0.9870792627334595
  },
  {
   "sentence": "*   **v1-data** achieves 69.6 accuracy on the IndicXNLI task for Kannada [3].",
   "supported": true,
   "chunk_id": "2212.05409:0054",
   "entailment": 0.5651244521141052
  },
  {
   "sentence": "*   **mBERT** achieves 58.6 accuracy on the IndicXNLI task for Kannada [3].",
   "supported": true,
   "chunk_id": "2212.05409:0054",
   "entailment": 0.9893993735313416
  }
 ],
 "regenerated": false,
 "retrieval_weak": true,
 "trace": [
  "1 intent=result complexity=simple; conditions={'language': 'Kannada'}",
  "2 planner agent: The question asks for performance metrics on a specific task, requiring a search for reported results.",
  "2 refinement: rewrite",
  "2 agent added refinement (the question was judged vague or multi-part)",
  "3 queries: ['What are the performance metrics and results of various models on the Kannada Natural Language Inference (NLI) dataset?']",
  "4 retrieved 20 -> 5 kept 3 (weak evidence)",
  "6 coverage 1.00, missing none",
  "4 weak-evidence flag withdrawn: every condition named in the question is recorded in the kept passages",
  "7 conflicts: none",
  "9 claims checked 6, unsupported 0"
 ],
 "timings_ms": {
  "1_understand": 1407.6,
  "2_plan": 378.5,
  "3_refine": 2089.4,
  "4_retrieve": 35.5,
  "5_rerank": 37.6,
  "2b_replan": 0,
  "6_applicability": 3.3,
  "7_contradiction": 12.3,
  "8_generate": 3328.2,
  "9_critic": 101,
  "total": 7396.5
 }
};

export const SAMPLE_DISTILBERT_GLUE: QueryResponse = {
 "question": "Compare DistilBERT and BERT-base on GLUE.",
 "answer": "DistilBERT is a distilled version of BERT that is smaller, faster, cheaper, and lighter [6]. On the GLUE benchmark (dev set, English, base), DistilBERT retains 97% of BERT-base performance [1].\n\nThe specific scores for DistilBERT and BERT-base on the GLUE dev set are as follows [1]:\n*   **CoLA**: DistilBERT 51.3 vs. BERT-base 56.3\n*   **MNLI**: DistilBERT 82.2 vs. BERT-base 86.7\n*   **MRPC**: DistilBERT 87.5 vs. BERT-base 88.6\n*   **QNLI**: DistilBERT 89.2 vs. BERT-base 91.8\n*   **QQP**: DistilBERT 88.5 vs. BERT-base 89.6\n*   **RTE**: DistilBERT 59.9 vs. BERT-base 69.3\n\nAdditionally, DistilBERT is significantly smaller than BERT-base, with 66 million parameters compared to 180 million [3]. DistilBERT is also faster, with an inference time of 410 seconds for the STS-B task compared to 668 seconds for BERT-base [3].",
 "sources": [
  {
   "chunk_id": "1910.01108:0006",
   "paper_id": "1910.01108",
   "paper_title": "DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter",
   "page": 3,
   "section": "methods",
   "score": 7.265,
   "preview": "Table 1: DistilBERT retains 97% of BERT performance. Comparison on the dev sets of the GLUE benchmark. ELMo results as reported by the authors. BERT and DistilBERT results are the medians of 5 runs with different seeds. | Model | Score | CoLA | MNLI | MRPC | QNLI | QQP | RTE | SS..."
  },
  {
   "chunk_id": "2004.02984:0020",
   "paper_id": "2004.02984",
   "paper_title": "MobileBERT: a Compact Task-Agnostic BERT for Resource-Limited Devices",
   "page": 6,
   "section": "experiments",
   "score": 7.112,
   "preview": "4.3 Results on GLUE The General Language Understanding Evaluation (GLUE) benchmark (Wang et al., 2018) is a collection of 9 natural language understanding tasks. We compare MobileBERT with BERTBASE and a few state-of-the-art pre-BERT models on the GLUE leaderboard3: OpenAI GPT (R..."
  },
  {
   "chunk_id": "1910.01108:0007",
   "paper_id": "1910.01108",
   "paper_title": "DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter",
   "page": 3,
   "section": "methods",
   "score": 5.884,
   "preview": "Table 2: DistilBERT yields to comparable performance on downstream tasks. Comparison on downstream tasks: IMDb (test accuracy) and SQuAD 1.1 (EM/F1 on dev set). D: with a second step of distillation during fine-tuning. | Model BERT-base DistilBERT DistilBERT (D) | IMDb (acc.) 93...."
  },
  {
   "chunk_id": "1909.10351:0019",
   "paper_id": "1909.10351",
   "paper_title": "TinyBERT: Distilling BERT for Natural Language Understanding",
   "page": 7,
   "section": "experiments",
   "score": 4.879,
   "preview": "In addition, BERT-PKD and DistilBERT initialize their student models with some layers of a pretrained BERT, which makes the student models have to keep the same size settings of Transformer layer (or embedding layer) as their teacher. In our two-stage distillation framework, Tiny..."
  },
  {
   "chunk_id": "1909.10351:0017",
   "paper_id": "1909.10351",
   "paper_title": "TinyBERT: Distilling BERT for Natural Language Understanding",
   "page": 6,
   "section": "experiments",
   "score": 4.558,
   "preview": "5For regression task STS-B, the original train set is better. 4.3 Baselines We compare TinyBERT with BERTTINY, BERTSMALL6 (Turc et al., 2019) and several state-of-the-art KD baselines including BERT- PKD (Sun et al., 2019), PD (Turc et al., 2019), DistilBERT (Sanh et al., 2019) a..."
  },
  {
   "chunk_id": "2004.02984:0023",
   "paper_id": "2004.02984",
   "paper_title": "MobileBERT: a Compact Task-Agnostic BERT for Resource-Limited Devices",
   "page": 7,
   "section": "experiments",
   "score": 2.15,
   "preview": "model with approximately 15M parameters called MobileBERTTINY4, which reduces the number of FFNs in each layer and uses a lighter MHA structure. Besides, to verify the performance of Mobile- BERT on real-world mobile devices, we export the models with TensorFlow Lite5 APIs and me..."
  }
 ],
 "analysis": {
  "intent": "comparison",
  "complexity": "complex",
  "conditions": {
   "task": null,
   "dataset": "GLUE",
   "dataset_version": null,
   "language": null,
   "model": "DistilBERT",
   "model_size": "base",
   "setting": null
  }
 },
 "applicability": {
  "coverage": 1,
  "checks": [
   {
    "condition": "dataset",
    "requested": "GLUE",
    "covered": true,
    "observed": [
     "MRPC",
     "QNLI",
     "SST-2",
     "CoLA",
     "QQP",
     "RTE"
    ],
    "chunk_ids": [
     "2004.02984:0020",
     "2004.02984:0023"
    ]
   },
   {
    "condition": "model",
    "requested": "DistilBERT",
    "covered": true,
    "observed": [
     "DistilBERT",
     "BERT-base",
     "MobileBERTTINY",
     "DistilBERT (D)",
     "ELMo",
     "TinyBERT"
    ],
    "chunk_ids": [
     "1910.01108:0006",
     "1910.01108:0007",
     "1909.10351:0019",
     "1909.10351:0017",
     "1910.01108:0008",
     "1910.01108:0010"
    ]
   },
   {
    "condition": "model_size",
    "requested": "base",
    "covered": true,
    "observed": [
     "base",
     "tiny",
     "109M",
     "small",
     "14.5M",
     "29.2M"
    ],
    "chunk_ids": [
     "1910.01108:0006",
     "2004.02984:0020",
     "1910.01108:0007",
     "1910.01108:0008",
     "2004.02984:0026"
    ]
   }
  ],
  "missing": [],
  "re_retrieved": false,
  "warning": null,
  "reasoning": "",
  "profile_guided": false,
  "escalated": false
 },
 "contradictions": [
  {
   "verdict": "NOT_COMPARABLE",
   "chunk_a": "1909.10351:0017",
   "chunk_b": "2004.02984:0023",
   "paper_a": "1909.10351",
   "paper_b": "2004.02984",
   "metric": null,
   "value_a": null,
   "value_b": null,
   "differing": [],
   "reason": "The sources disagree in wording, but there is no pair of recorded results with the same subject to compare (profile missing).",
   "nli_contradiction": 0.9818206429481506
  }
 ],
 "claim_checks": [
  {
   "sentence": "DistilBERT is a distilled version of BERT that is smaller, faster, cheaper, and lighter [6].",
   "supported": true,
   "chunk_id": "2004.02984:0023",
   "entailment": 0.6216953992843628
  },
  {
   "sentence": "On the GLUE benchmark (dev set, English, base), DistilBERT retains 97% of BERT-base performance [1].",
   "supported": true,
   "chunk_id": "1910.01108:0006",
   "entailment": 0.11390127241611481
  },
  {
   "sentence": "*   **CoLA**: DistilBERT 51.3 vs.",
   "supported": true,
   "chunk_id": "1910.01108:0006",
   "entailment": 0.4060015082359314
  },
  {
   "sentence": "BERT-base 56.3",
   "supported": true,
   "chunk_id": "1910.01108:0006",
   "entailment": 0.007281896658241749
  },
  {
   "sentence": "*   **MNLI**: DistilBERT 82.2 vs.",
   "supported": true,
   "chunk_id": "1910.01108:0006",
   "entailment": 0.47609785199165344
  },
  {
   "sentence": "BERT-base 86.7",
   "supported": true,
   "chunk_id": "1910.01108:0006",
   "entailment": 0.9679464101791382
  },
  {
   "sentence": "*   **MRPC**: DistilBERT 87.5 vs.",
   "supported": true,
   "chunk_id": "1910.01108:0006",
   "entailment": 0.44169318675994873
  },
  {
   "sentence": "BERT-base 88.6",
   "supported": true,
   "chunk_id": "1910.01108:0006",
   "entailment": 0.985420286655426
  },
  {
   "sentence": "*   **QNLI**: DistilBERT 89.2 vs.",
   "supported": true,
   "chunk_id": "1910.01108:0006",
   "entailment": 0.3530116379261017
  },
  {
   "sentence": "BERT-base 91.8",
   "supported": true,
   "chunk_id": "1910.01108:0006",
   "entailment": 0.9760618209838867
  },
  {
   "sentence": "*   **QQP**: DistilBERT 88.5 vs.",
   "supported": true,
   "chunk_id": "1910.01108:0006",
   "entailment": 0.5699235200881958
  },
  {
   "sentence": "BERT-base 89.6",
   "supported": true,
   "chunk_id": "1910.01108:0006",
   "entailment": 0.9058552980422974
  },
  {
   "sentence": "*   **RTE**: DistilBERT 59.9 vs.",
   "supported": true,
   "chunk_id": "1910.01108:0006",
   "entailment": 0.3865625858306885
  },
  {
   "sentence": "BERT-base 69.3",
   "supported": true,
   "chunk_id": "1910.01108:0006",
   "entailment": 0.9649220705032349
  },
  {
   "sentence": "Additionally, DistilBERT is significantly smaller than BERT-base, with 66 million parameters compared to 180 million [3].",
   "supported": true,
   "chunk_id": "1910.01108:0007",
   "entailment": 0.004239198751747608
  },
  {
   "sentence": "DistilBERT is also faster, with an inference time of 410 seconds for the STS-B task compared to 668 seconds for BERT-base [3].",
   "supported": true,
   "chunk_id": "1910.01108:0007",
   "entailment": 0.016105474904179573
  }
 ],
 "regenerated": false,
 "trace": [
  "1 intent=comparison complexity=complex; conditions={'dataset': 'GLUE', 'model': 'DistilBERT', 'model_size': 'base'}",
  "2 planner agent: The request asks for a comparison between two models on a specific benchmark.",
  "2 refinement: rewrite + decompose",
  "3 queries: ['Compare the performance and metrics of the DistilBERT model versus the BERT-base model on the GLUE benchmark.', 'What are the performance metrics of DistilBERT on the GLUE benchmark?', 'What are the performance metrics of BERT-base on the GLUE benchmark?', 'How do the results of DistilBERT and BERT-base compare specifically on the GLUE benchmark?']",
  "4 retrieved 20 -> 5 kept 10",
  "6 coverage 1.00, missing none",
  "7 conflicts: NOT_COMPARABLE",
  "9 claims checked 16, unsupported 0"
 ],
 "timings_ms": {
  "1_understand": 1490,
  "2_plan": 402,
  "3_refine": 1630,
  "4_retrieve": 41,
  "5_rerank": 44,
  "2b_replan": 0,
  "6_applicability": 420,
  "7_contradiction": 61,
  "8_generate": 4870,
  "9_critic": 236,
  "total": 9194
 },
 "retrieval_weak": false
};
