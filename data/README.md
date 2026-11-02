# Dataset notes

## sample_dataset.csv  (included)
26 short items **written by the project authors for demonstration**:
14 NLI items, 6 sarcasm items, 6 open factual questions.
The `human_votes` column contains **synthetic, illustrative** counts over 10
imaginary annotators. It is **NOT real annotation data** and is **NOT ChaosNLI**.
Use it only to check that the pipeline runs.

## Real data: ChaosNLI (recommended)
Your project report does not name a dataset. ChaosNLI (Nie et al., EMNLP 2020) is
the public dataset built for exactly this problem - 100 human annotations per
example, for SNLI, MNLI and abductive NLI - and it is the kind of dataset used by
the NLI disagreement study in your literature table (row 1).

* Repo:     https://github.com/easonnie/ChaosNLI
* Download: https://www.dropbox.com/s/h4j7dqszmpt2679/chaosNLI_v1.0.zip
* License:  CC BY-NC 4.0 (non-commercial)

Unzip so that you have
```
data/chaosNLI_v1.0/chaosNLI_snli.jsonl
data/chaosNLI_v1.0/chaosNLI_mnli_m.jsonl
data/chaosNLI_v1.0/chaosNLI_alphanli.jsonl   (not supported by the converter)
```
then convert (example: 200 random SNLI items):
```
python scripts/convert_chaosnli.py --input data/chaosNLI_v1.0/chaosNLI_snli.jsonl \
       --output data/chaosnli_snli_200.csv --n 200 --seed 0
python main.py --data data/chaosnli_snli_200.csv --backend anthropic --models <model-id>
```
