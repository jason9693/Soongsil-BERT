# Finetuning

## Result

### Base Model

|                       | Size  | **NSMC**<br/>(acc) | **Naver NER**<br/>(F1) | **PAWS**<br/>(acc) | **KorNLI**<br/>(acc) | **KorSTS**<br/>(spearman) | **Question Pair**<br/>(acc) | **KorQuaD (Dev)**<br/>(EM/F1) | **Korean-Hate-Speech (Dev)**<br/>(F1) |
| :-------------------- | :---: | :----------------: | :--------------------: | :----------------: | :------------------: | :-----------------------: | :-------------------------: | :---------------------------: | :-----------------------------------: |
| KoBERT                | 351M  |       89.59        |         87.92          |       81.25        |        79.62         |           81.59           |            94.85            |         51.75 / 79.15         |                 66.21                 |
| XLM-Roberta-Base      | 1.03G |       89.03        |         86.65          |       82.80        |        80.23         |           78.45           |            93.80            |         64.70 / 88.94         |                 64.06                 |
| HanBERT               | 614M  |       90.06        |         87.70          |       82.95        |        80.32         |           82.73           |            94.72            |         78.74 / 92.02         |               68.32               |
| KoELECTRA-Base-v3 | 431M  |     90.63      |       88.11        |     84.45      |      82.24       |         85.53         |          95.25          |       84.83 / 93.45       |                 67.61                 |
| Soongsil-BERT | 370M  |     **91.2**      |       -        |     -      |      -       |         76         |          94          |        -      | **69** |

### Small Model

|                        | Size | **NSMC**<br/>(acc) | **Naver NER**<br/>(F1) | **PAWS**<br/>(acc) | **KorNLI**<br/>(acc) | **KorSTS**<br/>(spearman) | **Question Pair**<br/>(acc) | **KorQuaD (Dev)**<br/>(EM/F1) | **Korean-Hate-Speech (Dev)**<br/>(F1) |
| :--------------------- | :--: | :----------------: | :--------------------: | :----------------: | :------------------: | :-----------------------: | :-------------------------: | :---------------------------: | :-----------------------------------: |
| DistilKoBERT           | 108M |       88.60        |         84.65          |       60.50        |        72.00         |           72.59           |            92.48            |         54.40 / 77.97         |                 60.72                 |
| KoELECTRA-Small-v3 | 54M  |     89.36      |       85.40        |     77.45      |      78.60       |         80.79         |          94.85          |       82.11 / 91.13       |               63.07               |
| Soongsil-BERT | 213M  |     **90.7**      |       84        |     69.1      |      76       |         -         |          92          |        -      | **66** |

## Reference
- [Transformers Examples](https://github.com/huggingface/transformers/blob/master/examples/README.md)
- [NSMC](https://github.com/e9t/nsmc)
- [Naver NER Dataset](https://github.com/naver/nlp-challenge)
- [PAWS](https://github.com/google-research-datasets/paws)
- [KorNLI/KorSTS](https://github.com/kakaobrain/KorNLUDatasets)
- [Question Pair](https://github.com/songys/Question_pair)
- [KorQuad](https://korquad.github.io/category/1.0_KOR.html)
- [Korean Hate Speech](https://github.com/kocohub/korean-hate-speech)
- [KoELECTRA](https://github.com/monologg/KoELECTRA)
- [KoBERT](https://github.com/SKTBrain/KoBERT)
- [HanBERT](https://github.com/tbai2019/HanBert-54k-N)
- [HanBert Transformers](https://github.com/monologg/HanBert-Transformers)
