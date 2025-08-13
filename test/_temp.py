from jet.video.utils import deduplicate_segments


segments = [
    {
        "chunk_idx": 0,
        "segment_idx": 11,
        "id": "89f6937c0f39e12447f4fa0e",
        "seek": 4838,
        "start": 48.38,
        "end": 59.92,
        "chapter_title": None,
        "text": " You would have recollection of your childhood, right? Can you just give us a recollection of your childhood? Sino po yung best friend nyo? Saan po kayo nag-aral? Kung meron kayo makamag-an?",
        "eval": {
                "confidence": 0.7847,
                "temperature": 0.0,
                "avg_logprob": -0.2424469035968446,
                "compression_ratio": 1.4765625,
                "no_speech_prob": 0.28175118565559387
        },
        "words": [
            {
                "start": 48.38,
                "end": 49.26,
                "word": " You",
                "probability": 0.116712786257267
            },
            {
                "start": 49.26,
                "end": 49.44,
                "word": " would",
                "probability": 0.8770300149917603
            },
            {
                "start": 49.44,
                "end": 49.62,
                "word": " have",
                "probability": 0.9853416681289673
            },
            {
                "start": 49.62,
                "end": 50.22,
                "word": " recollection",
                "probability": 0.7272551655769348
            },
            {
                "start": 50.22,
                "end": 50.42,
                "word": " of",
                "probability": 0.984682559967041
            },
            {
                "start": 50.42,
                "end": 50.54,
                "word": " your",
                "probability": 0.9763495922088623
            },
            {
                "start": 50.54,
                "end": 51.04,
                "word": " childhood,",
                "probability": 0.994772732257843
            },
            {
                "start": 51.3,
                "end": 51.42,
                "word": " right?",
                "probability": 0.6160953044891357
            },
            {
                "start": 51.96,
                "end": 52.06,
                "word": " Can",
                "probability": 0.9627609252929688
            },
            {
                "start": 52.06,
                "end": 52.16,
                "word": " you",
                "probability": 0.9864711165428162
            },
            {
                "start": 52.16,
                "end": 52.36,
                "word": " just",
                "probability": 0.945361316204071
            },
            {
                "start": 52.36,
                "end": 52.56,
                "word": " give",
                "probability": 0.993833065032959
            },
            {
                "start": 52.56,
                "end": 52.7,
                "word": " us",
                "probability": 0.9961836934089661
            },
            {
                "start": 52.7,
                "end": 52.84,
                "word": " a",
                "probability": 0.7593212723731995
            },
            {
                "start": 52.84,
                "end": 53.36,
                "word": " recollection",
                "probability": 0.9995019733905792
            },
            {
                "start": 53.36,
                "end": 53.52,
                "word": " of",
                "probability": 0.9924731254577637
            },
            {
                "start": 53.52,
                "end": 53.64,
                "word": " your",
                "probability": 0.9773865938186646
            },
            {
                "start": 53.64,
                "end": 54.16,
                "word": " childhood?",
                "probability": 0.999474823474884
            },
            {
                "start": 54.66,
                "end": 54.94,
                "word": " Sino",
                "probability": 0.6575186997652054
            },
            {
                "start": 54.94,
                "end": 55.1,
                "word": " po",
                "probability": 0.8941433429718018
            },
            {
                "start": 55.1,
                "end": 55.24,
                "word": " yung",
                "probability": 0.8681326806545258
            },
            {
                "start": 55.24,
                "end": 55.46,
                "word": " best",
                "probability": 0.9815126657485962
            },
            {
                "start": 55.46,
                "end": 55.8,
                "word": " friend",
                "probability": 0.959471583366394
            },
            {
                "start": 55.8,
                "end": 56.06,
                "word": " nyo?",
                "probability": 0.8276257514953613
            },
            {
                "start": 56.5,
                "end": 56.76,
                "word": " Saan",
                "probability": 0.5429802425205708
            },
            {
                "start": 56.76,
                "end": 56.86,
                "word": " po",
                "probability": 0.9648469686508179
            },
            {
                "start": 56.86,
                "end": 57.12,
                "word": " kayo",
                "probability": 0.8761059045791626
            },
            {
                "start": 57.12,
                "end": 57.24,
                "word": " nag",
                "probability": 0.8951113820075989
            },
            {
                "start": 57.24,
                "end": 58.12,
                "word": "-aral?",
                "probability": 0.7443079749743143
            },
            {
                "start": 58.32,
                "end": 58.56,
                "word": " Kung",
                "probability": 0.8402228951454163
            },
            {
                "start": 58.56,
                "end": 58.92,
                "word": " meron",
                "probability": 0.8095099329948425
            },
            {
                "start": 58.92,
                "end": 59.16,
                "word": " kayo",
                "probability": 0.851032018661499
            },
            {
                "start": 59.16,
                "end": 59.64,
                "word": " makamag",
                "probability": 0.898019274075826
            },
            {
                "start": 59.64,
                "end": 59.92,
                "word": "-an?",
                "probability": 0.6206994205713272
            }
        ]
    },
    {
        "chunk_idx": 1,
        "segment_idx": 12,
        "id": "ec0208a9cc95ed7769d42cde",
        "seek": 5500,
        "start": 55.0,
        "end": 79.36,
        "chapter_title": None,
        "text": " Saan po yung best friend nyo? Saan po kayo nag-aral? Kung meron kayo makamag-anak, recollections of your only 38 years old, you will remember your childhood. I mean I'm double your age but I would remember my childhood in Malabon, your childhood, I know homeschool, I can't believe anyone is totally homeschooled all her life, yes, please tell us about your childhood.",
        "eval": {
            "confidence": 0.7084,
            "temperature": 0.0,
            "avg_logprob": -0.3446975556930693,
            "compression_ratio": 1.6576576576576576,
            "no_speech_prob": 0.8990771174430847
        },
        "words": [
            {
                "start": 55.0,
                "end": 55.0,
                "word": " Saan",
                "probability": 0.4708516523241997
            },
            {
                "start": 55.0,
                "end": 55.06,
                "word": " po",
                "probability": 0.9143111109733582
            },
            {
                "start": 55.06,
                "end": 55.24,
                "word": " yung",
                "probability": 0.7786822319030762
            },
            {
                "start": 55.24,
                "end": 55.52,
                "word": " best",
                "probability": 0.9858465790748596
            },
            {
                "start": 55.52,
                "end": 55.8,
                "word": " friend",
                "probability": 0.9673879742622375
            },
            {
                "start": 55.8,
                "end": 56.06,
                "word": " nyo?",
                "probability": 0.7408478558063507
            },
            {
                "start": 56.48,
                "end": 56.76,
                "word": " Saan",
                "probability": 0.9190934598445892
            },
            {
                "start": 56.76,
                "end": 56.86,
                "word": " po",
                "probability": 0.9720374345779419
            },
            {
                "start": 56.86,
                "end": 57.1,
                "word": " kayo",
                "probability": 0.893155425786972
            },
            {
                "start": 57.1,
                "end": 57.24,
                "word": " nag",
                "probability": 0.8849827647209167
            },
            {
                "start": 57.24,
                "end": 58.06,
                "word": "-aral?",
                "probability": 0.7684739232063293
            },
            {
                "start": 58.32,
                "end": 58.58,
                "word": " Kung",
                "probability": 0.8151159286499023
            },
            {
                "start": 58.58,
                "end": 58.92,
                "word": " meron",
                "probability": 0.858997642993927
            },
            {
                "start": 58.92,
                "end": 59.14,
                "word": " kayo",
                "probability": 0.81108558177948
            },
            {
                "start": 59.14,
                "end": 59.64,
                "word": " makamag",
                "probability": 0.8810748855272929
            },
            {
                "start": 59.64,
                "end": 60.34,
                "word": "-anak,",
                "probability": 0.3984034111102422
            },
            {
                "start": 60.4,
                "end": 61.24,
                "word": " recollections",
                "probability": 0.7093559602896372
            },
            {
                "start": 61.24,
                "end": 61.86,
                "word": " of",
                "probability": 0.6690431833267212
            },
            {
                "start": 61.86,
                "end": 62.22,
                "word": " your",
                "probability": 0.5193524360656738
            },
            {
                "start": 62.22,
                "end": 62.64,
                "word": " only",
                "probability": 0.7970179915428162
            },
            {
                "start": 62.64,
                "end": 63.1,
                "word": " 38",
                "probability": 0.9013602137565613
            },
            {
                "start": 63.1,
                "end": 63.36,
                "word": " years",
                "probability": 0.8201810717582703
            },
            {
                "start": 63.36,
                "end": 63.82,
                "word": " old,",
                "probability": 0.9924426078796387
            },
            {
                "start": 64.22,
                "end": 64.32,
                "word": " you",
                "probability": 0.9672555327415466
            },
            {
                "start": 64.32,
                "end": 64.5,
                "word": " will",
                "probability": 0.9863885045051575
            },
            {
                "start": 64.5,
                "end": 65.03999999999999,
                "word": " remember",
                "probability": 0.9976162910461426
            },
            {
                "start": 65.03999999999999,
                "end": 65.98,
                "word": " your",
                "probability": 0.9497065544128418
            },
            {
                "start": 65.98,
                "end": 66.38,
                "word": " childhood.",
                "probability": 0.9975200295448303
            },
            {
                "start": 66.64,
                "end": 66.72,
                "word": " I",
                "probability": 0.9763808250427246
            },
            {
                "start": 66.72,
                "end": 66.82,
                "word": " mean",
                "probability": 0.3887631595134735
            },
            {
                "start": 66.82,
                "end": 67.06,
                "word": " I'm",
                "probability": 0.7511758208274841
            },
            {
                "start": 67.06,
                "end": 67.28,
                "word": " double",
                "probability": 0.9628445506095886
            },
            {
                "start": 67.28,
                "end": 67.53999999999999,
                "word": " your",
                "probability": 0.9464563131332397
            },
            {
                "start": 67.53999999999999,
                "end": 67.82,
                "word": " age",
                "probability": 0.9992730021476746
            },
            {
                "start": 67.82,
                "end": 68.02,
                "word": " but",
                "probability": 0.7278383374214172
            },
            {
                "start": 68.02,
                "end": 68.22,
                "word": " I",
                "probability": 0.9733515977859497
            },
            {
                "start": 68.22,
                "end": 68.34,
                "word": " would",
                "probability": 0.6658109426498413
            },
            {
                "start": 68.34,
                "end": 68.62,
                "word": " remember",
                "probability": 0.9962382316589355
            },
            {
                "start": 68.62,
                "end": 68.88,
                "word": " my",
                "probability": 0.9929161071777344
            },
            {
                "start": 68.88,
                "end": 69.38,
                "word": " childhood",
                "probability": 0.9988975524902344
            },
            {
                "start": 69.38,
                "end": 69.68,
                "word": " in",
                "probability": 0.9642589092254639
            },
            {
                "start": 69.68,
                "end": 70.18,
                "word": " Malabon,",
                "probability": 0.9536539316177368
            },
            {
                "start": 70.72,
                "end": 70.82,
                "word": " your",
                "probability": 0.9579415321350098
            },
            {
                "start": 70.82,
                "end": 71.32,
                "word": " childhood,",
                "probability": 0.9995927214622498
            },
            {
                "start": 72.12,
                "end": 72.26,
                "word": " I",
                "probability": 0.9828392863273621
            },
            {
                "start": 72.26,
                "end": 72.42,
                "word": " know",
                "probability": 0.9951527118682861
            },
            {
                "start": 72.42,
                "end": 73.2,
                "word": " homeschool,",
                "probability": 0.914544016122818
            },
            {
                "start": 73.34,
                "end": 73.4,
                "word": " I",
                "probability": 0.99216228723526
            },
            {
                "start": 73.4,
                "end": 73.66,
                "word": " can't",
                "probability": 0.9921835958957672
            },
            {
                "start": 73.66,
                "end": 73.84,
                "word": " believe",
                "probability": 0.9918796420097351
            },
            {
                "start": 73.84,
                "end": 74.34,
                "word": " anyone",
                "probability": 0.9816911816596985
            },
            {
                "start": 74.34,
                "end": 74.56,
                "word": " is",
                "probability": 0.9583480358123779
            },
            {
                "start": 74.56,
                "end": 75.0,
                "word": " totally",
                "probability": 0.9930840730667114
            },
            {
                "start": 75.0,
                "end": 75.72,
                "word": " homeschooled",
                "probability": 0.9859300454457601
            },
            {
                "start": 75.72,
                "end": 75.82,
                "word": " all",
                "probability": 0.9572428464889526
            },
            {
                "start": 75.82,
                "end": 75.98,
                "word": " her",
                "probability": 0.9669672250747681
            },
            {
                "start": 75.98,
                "end": 76.48,
                "word": " life,",
                "probability": 0.9987199306488037
            },
            {
                "start": 76.78,
                "end": 77.02,
                "word": " yes,",
                "probability": 0.933233380317688
            },
            {
                "start": 77.6,
                "end": 78.0,
                "word": " please",
                "probability": 0.9875659942626953
            },
            {
                "start": 78.0,
                "end": 78.38,
                "word": " tell",
                "probability": 0.9933688044548035
            },
            {
                "start": 78.38,
                "end": 78.53999999999999,
                "word": " us",
                "probability": 0.9990224838256836
            },
            {
                "start": 78.53999999999999,
                "end": 78.72,
                "word": " about",
                "probability": 0.99429851770401
            },
            {
                "start": 78.72,
                "end": 78.9,
                "word": " your",
                "probability": 0.9933191537857056
            },
            {
                "start": 78.9,
                "end": 79.36,
                "word": " childhood.",
                "probability": 0.999601423740387
            }
        ]
    },
]

prev_segments = []

chunk_start_ms = 0.0
tolerance = 0.5
overlap_duration = 5.0
result_current, result_prev = deduplicate_segments(
    segments, prev_segments, overlap_duration, chunk_start_ms, tolerance)


expected_prev = [
    {
        "chunk_idx": 0,
        "segment_idx": 11,
        "id": "89f6937c0f39e12447f4fa0e",
        "seek": 4838,
        "start": 48.38,
        "end": 59.92,
        "chapter_title": None,
        "text": " You would have recollection of your childhood, right? Can you just give us a recollection of your childhood?",
        "eval": {
            "confidence": 0.7847,
            "temperature": 0.0,
            "avg_logprob": -0.2424469035968446,
            "compression_ratio": 1.4765625,
            "no_speech_prob": 0.28175118565559387
        },
        "words": [
            {
                "start": 48.38,
                "end": 49.26,
                "word": " You",
                "probability": 0.116712786257267
            },
            {
                "start": 49.26,
                "end": 49.44,
                "word": " would",
                "probability": 0.8770300149917603
            },
            {
                "start": 49.44,
                "end": 49.62,
                "word": " have",
                "probability": 0.9853416681289673
            },
            {
                "start": 49.62,
                "end": 50.22,
                "word": " recollection",
                "probability": 0.7272551655769348
            },
            {
                "start": 50.22,
                "end": 50.42,
                "word": " of",
                "probability": 0.984682559967041
            },
            {
                "start": 50.42,
                "end": 50.54,
                "word": " your",
                "probability": 0.9763495922088623
            },
            {
                "start": 50.54,
                "end": 51.04,
                "word": " childhood,",
                "probability": 0.994772732257843
            },
            {
                "start": 51.3,
                "end": 51.42,
                "word": " right?",
                "probability": 0.6160953044891357
            },
            {
                "start": 51.96,
                "end": 52.06,
                "word": " Can",
                "probability": 0.9627609252929688
            },
            {
                "start": 52.06,
                "end": 52.16,
                "word": " you",
                "probability": 0.9864711165428162
            },
            {
                "start": 52.16,
                "end": 52.36,
                "word": " just",
                "probability": 0.945361316204071
            },
            {
                "start": 52.36,
                "end": 52.56,
                "word": " give",
                "probability": 0.993833065032959
            },
            {
                "start": 52.56,
                "end": 52.7,
                "word": " us",
                "probability": 0.9961836934089661
            },
            {
                "start": 52.7,
                "end": 52.84,
                "word": " a",
                "probability": 0.7593212723731995
            },
            {
                "start": 52.84,
                "end": 53.36,
                "word": " recollection",
                "probability": 0.9995019733905792
            },
            {
                "start": 53.36,
                "end": 53.52,
                "word": " of",
                "probability": 0.9924731254577637
            },
            {
                "start": 53.52,
                "end": 53.64,
                "word": " your",
                "probability": 0.9773865938186646
            },
            {
                "start": 53.64,
                "end": 54.16,
                "word": " childhood?",
                "probability": 0.999474823474884
            },
        ]
    },
]

expected_current = segments

assert result_current == expected_current
assert result_prev == expected_prev
