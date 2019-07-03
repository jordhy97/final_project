# final_project
Aspect and opinion terms extraction for hotel's review from AiryRooms in Bahasa Indonesia

## Corpus description
The corpus is located in the folder data/reviews. The corpus consists of 5000 reviews (78.604 tokens) that are divided into train.txt (4000 reviews) and test.txt (1000 reviews). Here's the label distribution for the corpus.

| Label         | train.txt     | test.txt  |
| ------------- |-------------:| -----:|
| B-ASPECT      | 7005 | 1758 |
| I-ASPECT      | 2292      |   584 |
| B-SENTIMENT   | 9646      |   2384 |
| I-SENTIMENT   | 4265      |   1067 |
| OTHER         | 39897      |   9706 |
| Total        | 63105      |   15499 |

reviews.txt contains raw reviews and reviews_preprocessed.txt contains reviews that have been preprocessed that are used to train word embedding.
