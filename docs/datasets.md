# Public datasets

The testbed evaluates coresets on seven public datasets from the UCI Machine
Learning Repository, plus the small pendigits set. Nothing is redistributed.
Each descriptor in `PublicDataset.REGISTRY` pins a download URL, its SHA-256
and size, the license and citation, and the preprocessing. The download is
converted once into a binary point file in the local cache, and that file is
what the testbed loads.

```
coreset-testbed datasets list                        # descriptors
coreset-testbed datasets fetch higgs                 # download, check SHA-256
coreset-testbed datasets convert higgs               # raw + zscore binaries, verified
coreset-testbed datasets convert nytimes --variant raw,tfidf
coreset-testbed datasets stats higgs --variant zscore --k 10,25,50 --seed 1
```

In a config, `{type: public, params: {name: higgs, variant: zscore}}` loads
the converted file (`variant` defaults to `raw`). Loading converts on first use
if the download is cached. It downloads only with `--allow-download`.

## Datasets

All eight are licensed **CC BY 4.0**. The license was read off each dataset's
UCI page on 2026-11-08.

| dataset | n | d | nnz (raw) | download | binary (raw) | citation | preprocessing |
|---|---:|---:|---:|---:|---:|---|---|
| covertype | 581,012 | 54 | 6,901,775 | 75.8 MB | 251.0 MB | Blackard (1998), [10.24432/C50K5N](https://doi.org/10.24432/C50K5N) | drop `Cover_Type` (label). Keep 10 quantitative and 44 one-hot 0/1 columns |
| census1990 | 2,458,285 | 68 | 71,928,503 | 361.3 MB | 1.34 GB | Meek, Thiesson, Heckerman (2001), [10.24432/C5VP42](https://doi.org/10.24432/C5VP42) | drop `caseid`. The 68 attributes are the source's integer codes |
| kddcup99 | 4,898,431 | 38 | 46,443,269 | 18.1 MB | 1.49 GB | Stolfo et al. (1999), [10.24432/C51C7N](https://doi.org/10.24432/C51C7N) | full `kddcup.data`. Drop columns 1-3 (symbolic `protocol_type`, `service`, `flag`) and 41 (label). Keep 38 numeric columns, 0/1 flags included |
| yearmsd | 515,345 | 90 | 46,381,046 | 211.0 MB | 371.0 MB | Bertin-Mahieux (2011), [10.24432/C50K61](https://doi.org/10.24432/C50K61) | drop column 0 (release year, the regression target) |
| susy | 5,000,000 | 18 | 88,938,127 | 922.4 MB | 720.0 MB | Whiteson (2014), [10.24432/C54606](https://doi.org/10.24432/C54606) | drop column 0 (label) |
| higgs | 11,000,000 | 28 | 283,685,620 | 2.82 GB | 2.46 GB | Whiteson (2014), [10.24432/C5V312](https://doi.org/10.24432/C5V312) | drop column 0 (label) |
| nytimes | 300,000 | 102,660 | 69,679,427 | 234.2 MB | 838.6 MB | Newman (2008), [10.24432/C5ZG6P](https://doi.org/10.24432/C5ZG6P) | word counts, one row per document, word ids made 0-based |
| pendigits | 10,992 | 16 | 153,308 | 0.5 MB | 1.4 MB | Alpaydin, Alimoglu (1996), [10.24432/C5MG6K](https://doi.org/10.24432/C5MG6K) | drop `Class` (label) |

nnz counts the non-zero values. Dense binaries store zeros too, so their size is
`64 + 8 n d` bytes. The variant files have the same size as the raw ones. After
z-scoring, only constant features stay zero: covertype has 31,374,648 non-zeros,
census1990 167,163,380, kddcup99 181,241,947, yearmsd 46,381,050, susy
90,000,000, higgs 308,000,000 and pendigits 175,872. The nytimes tf-idf file
keeps all 69,679,427 entries, because no word occurs in every document.

### Sources and provenance decisions

- **KDDCup99.** UCI's `kdd+cup+1999+data.zip` (dataset 130) holds only
  dangling links into `kdd0/databases/kddcup99/`. `kdd.ics.uci.edu`, the UCI
  KDD Archive, lists the file but answers 403. The descriptor therefore
  downloads the Internet Archive capture of the original file,
  `https://web.archive.org/web/20070709044442id_/http://kdd.ics.uci.edu/databases/kddcup99/kddcup.data.gz`.
  Its SHA-256 (`3b6c942a…74292`) equals the digest scikit-learn pins for its
  own mirror of the file (`fetch_kddcup99`, figshare file 5976045). The file
  has 4,898,431 rows, as the archive states.
- **KDDCup99 symbolic columns.** One-hot encoding would add 3 + 70 + 11 = 84
  sparse 0/1 columns (the counts of distinct protocol, service and flag values
  in the full file). Under squared Euclidean distance these would dominate the
  z-scored numeric columns, and the encoding is a modelling choice the study
  does not need, so the three columns are dropped. `num_outbound_cmds` is 0 in
  every row. It is kept (std 0) so that column indices match the source.
- **NYTimes.** The dataset's zip bundles all five corpora, about 2 GB. The
  descriptor downloads only `docword.nytimes.txt.gz` from the same UCI
  directory (`ml/machine-learning-databases/bag-of-words/`). The vocabulary
  (`vocab.nytimes.txt`, SHA-256 `e92598bc…858ef`) is not needed for clustering.
- The other URLs are UCI's `static/public/<id>/` files. YearPredictionMSD,
  SUSY and HIGGS are zips, and the descriptor's `member` names the entry that is
  read: `YearPredictionMSD.txt`, `SUSY.csv.gz` and `HIGGS.csv.gz`.

## Preprocessing policy

- Values are parsed as float64 by `Double.parseDouble`, which rounds correctly.
  The binary therefore holds exactly the doubles the text denotes. Every point
  has weight 1.
- Only the columns listed in the descriptor are dropped: ids, labels, targets
  and, for KDDCup99, symbolic attributes. No imputation is done, since none of
  the files has missing values. No deduplication is done either.
- **Order is preserved.** Conversion never shuffles. Experiments that need a
  random order or random splits must shuffle with an explicit seed.
- Variants are separate files, derived deterministically from the raw binary:
  - `zscore` (dense): `(x - mean) / std` per feature, using the population std.
    A constant feature becomes 0. The mean takes one pass, the squared
    deviations a second (no cancellation), and the write a third. All passes are
    sequential, so the result is bit-reproducible.
  - `tfidf` (sparse): `count * ln(N / df)` per entry, where `df` is the number
    of documents containing the word. Every non-empty row is then scaled to unit
    L2 norm. Words that occur in every document get weight 0 and are dropped.
- Converting the same download twice gives byte-identical binaries (checked by
  reconverting covertype, kddcup99 and nytimes into a fresh cache). The
  SHA-256 values below pin them.

## Binary format

`PointFile` (reader) and `PointFileWriter` implement two layouts with one
64-byte little-endian header:

| offset | bytes | field |
|---:|---:|---|
| 0 | 4 | magic: `CBIN` (dense) or `CCSR` (sparse) |
| 4 | 2 | version, 1 |
| 6 | 2 | flags: 1 = zscore, 2 = tfidf |
| 8 | 8 | n |
| 16 | 8 | d |
| 24 | 8 | nnz (dense: non-zero values, sparse: stored entries) |
| 32 | 8 | CRC32C of the payload |
| 40 | 24 | reserved, zero |

The payload starts at byte 64. In `.cbin` it is `n * d` float64, row-major. In
`.csr` it is `n + 1` int64 row offsets, then `nnz` int32 column indices
(strictly increasing within a row), zero padding to a multiple of 8, then `nnz`
float64 values (finite, non-zero). Every section is 8-byte aligned, so the
files can be memory-mapped. For example, numpy can read a dense file with
`np.memmap(f, '<f8', offset=64, shape=(n, d))`.

`PointFile.open(file, chunkRows)` streams one `PointBlock` per chunk:
65,536 rows by default, capped at 8 Mi entries. It reuses one I/O buffer, and
each block owns exactly sized arrays (zeros of dense rows are dropped, as in
every block). There are no per-point objects. `readAll` concatenates the chunks
into a `PointSet` without copying, and `forEachRow` streams raw rows for
transforms. `verify` recomputes the CRC32C, and the header is checked against
the file length on every open. The writer builds `<file>.part` and renames it
on `finish()`.

Loading the raw binaries into a `PointSet` (warm page cache, Apple M2 Max,
JDK 17):

| file | load | throughput |
|---|---:|---:|
| covertype.cbin | 0.10 s | 2.5 GB/s |
| census1990.cbin | 0.74 s | 1.8 GB/s |
| kddcup99.cbin | 0.53 s | 2.8 GB/s |
| yearmsd.cbin | 0.26 s | 1.5 GB/s |
| susy.cbin | 0.49 s | 1.5 GB/s |
| higgs.cbin | 1.92 s | 1.3 GB/s |
| nytimes.csr | 0.38 s | 2.2 GB/s |

Parsing the text took 6 s (covertype) to 373 s (HIGGS) on the same machine.

## Checksums

The binaries were produced by `datasets convert` on 2026-11-08. If a
conversion gives a different digest, the format or the parsing has changed.

| dataset | download SHA-256 | binary | binary SHA-256 | CRC32C |
|---|---|---|---|---|
| covertype | `a07902ee1c9d3231c6655f23e6f75a6797d0ba26a2359f533c2c0e65d05c9bd4` | `covertype.cbin` | `004e620a894e498888cf6714e8253a6036a6e6ce1486a81249e2efd7f8bb778d` | `fc6a0e23` |
|  |  | `covertype.zscore.cbin` | `df00ee35997430973fffb9282cdcc9f54289bad638382e0b3b42ba873d789335` | `c822ca8d` |
| census1990 | `38e99f50855ddc03167b715b36519ef40b77295ae4e85dfef0c6429ac79ed799` | `census1990.cbin` | `e33ffb7dd8d1f777a508c10a17ef02243244f2b9e394cff8ac94a85b54c4b3bc` | `bceee3ce` |
|  |  | `census1990.zscore.cbin` | `1489eb31a665143789177ce7b843050dd9d9d4b0ea149044ac23c9d4bdd3b858` | `ac26ef55` |
| kddcup99 | `3b6c942aa0356c0ca35b7b595a26c89d343652c9db428893e7494f837b274292` | `kddcup99.cbin` | `8460f1bbb573b13c8b63bd8cd9fd21ace371479f30b14a66d9fc9a6a075a3265` | `a75182f5` |
|  |  | `kddcup99.zscore.cbin` | `95f07653b84d98bfa0d9489830b151fdb128107847575a61cfb7903786e19a3c` | `f3877ef1` |
| yearmsd | `06f801af323bb7798e800583acce4ea1ed2697ac12c23f4424aea0a7a3d09e11` | `yearmsd.cbin` | `429fd373f0a76a6611f5b8ba816a95ce40cb710f1737e8bbd1021d8d9e1671f6` | `62e69f60` |
|  |  | `yearmsd.zscore.cbin` | `d9ee90c640e33a641e2a01091137d71ed27d4cf82cbe643a89e4e9bca247305e` | `eb5a9238` |
| susy | `dd20f2a756677d369415ef0596c89266e2c5a175958056c22a36cb5321754169` | `susy.cbin` | `366b03d2b993f5a61eed76942149bdb28834078c2a5932f94882fc6931e056e0` | `1ad96657` |
|  |  | `susy.zscore.cbin` | `1c9cf054f0ae976534862a15deaf50a22c175be984c56b5dac40592e9d5b1f3b` | `d76e06b2` |
| higgs | `eca8d41530af41bef24854c4eb0e3e315d41b2c837aaff67d1fde908a29dee0f` | `higgs.cbin` | `168346d6646662ac84d2efbf871c5944326f7f8ef23b4c49b314338f60b3e2a9` | `7e363ee2` |
|  |  | `higgs.zscore.cbin` | `6c00c5c8d8db9035796d8c51972ea6ba9d26d4c5187eb3e855825489a8d4ae23` | `9fc85c32` |
| nytimes | `6fc63144a5834ddcc255313625a235a85440bf5b702d06a6c676f749262e43b7` | `nytimes.csr` | `33cacdcf5c186d233d64489d11d323afff8fa3a45b1660475262d921ce628d82` | `120a02e9` |
|  |  | `nytimes.tfidf.csr` | `5e0f7e1e749b8e39431aa724d83091682bb91d20e9e88ca014ddcf2ac5e35212` | `6c2b8985` |
| pendigits | `3a3043879657816e18af4bc0822cb89aff03b30988511590e2c05454ce092656` | `pendigits.cbin` | `687dbccd8c044f4c24db27fab26e741f1f3f1f4b7c7079913e1824ad96e7be1a` | `ceaf6a1f` |
|  |  | `pendigits.zscore.cbin` | `769babe4db24bfda8cf5c948b1d0800ab1ee335768bbcc6561808e8949a2857e` | `384937c2` |

### Validation against the sources

Each binary was compared with its source by an independent reader (Python and
numpy, not the Java loader). The check covered the first 1,000 rows and every
997th row after them: 1,010 rows (pendigits) to 12,032 rows (HIGGS) per
dataset, and 1,299 documents for nytimes. Each sampled row was parsed from the
download, columns were dropped, and the result was compared bit for bit with
the memory-mapped binary. All rows matched. The row and entry counts matched
the source, and the nnz in each header matched a full count. Recomputing the
z-score from the raw binary with numpy reproduced the sampled rows of every
`.zscore.cbin` exactly (maximum absolute difference 0).

## Reference statistics

These numbers come from `datasets stats <name> --variant <v> --seed 1`. The
weight sum equals n, since every weight is 1. The total variance is the mean
squared distance to the centroid, i.e. the 1-means cost divided by n. The
"std" columns are the smallest, median and largest per-feature population
standard deviation. The cost is that of one greedy k-means++ seeding
(`KMeansPlusPlus`, 2 + ln k trials per step, `SplittableRandom(1)`) with every
point assigned to its nearest seed and no Lloyd steps. It is a sanity scale,
not an optimum. Balance is the largest cluster size divided by the smallest in
that assignment.

| dataset | variant | weight sum | total variance | std min / median / max | k-means++ cost k = 10 / 25 / 50 | balance k = 10 / 25 / 50 |
|---|---|---:|---:|---|---|---|
| covertype | raw | 581,012 | 4.327e+06 | 0.002272 / 0.1484 / 1559 | 5.162e+11 / 2.396e+11 / 1.554e+11 | 12.09 / 7.602 / 15.46 |
| covertype | zscore | 581,012 | 54 | 1 / 1 / 1 | 3.042e+07 / 1.95e+07 / 7.541e+06 | 15.65 / 1144 / 1.523e+04 |
| census1990 | raw | 2,458,285 | 4521 | 0.03882 / 0.8453 / 65.15 | 3.833e+08 / 2.559e+08 / 1.961e+08 | 36.52 / 11.75 / 60.23 |
| census1990 | zscore | 2,458,285 | 68 | 1 / 1 / 1 | 1.326e+08 / 9.596e+07 / 7.689e+07 | 19.15 / 106.4 / 44.9 |
| kddcup99 | raw | 4,898,431 | 1.302e+12 | 0 / 0.2325 / 9.414e+05 | 7.504e+15 / 1.144e+14 / 1.153e+13 | 4.898e+06 / 4.873e+06 / 2.902e+06 |
| kddcup99 | zscore | 4,898,431 | 37 | 0 / 1 / 1 | 8.094e+07 / 2.388e+07 / 8.405e+06 | 2.856e+06 / 2.814e+06 / 2.807e+06 |
| yearmsd | raw | 515,345 | 9.531e+06 | 4.371 / 119.8 / 1749 | 3.151e+12 / 2.462e+12 / 2.15e+12 | 11.03 / 560 / 1288 |
| yearmsd | zscore | 515,345 | 90 | 1 / 1 / 1 | 4.962e+07 / 4.5e+07 / 4.216e+07 | 375.9 / 134.2 / 1029 |
| susy | raw | 5,000,000 | 11.18 | 0.197 / 0.7733 / 1.003 | 4.153e+07 / 3.274e+07 / 2.577e+07 | 12.61 / 11.95 / 33.08 |
| susy | zscore | 5,000,000 | 18 | 1 / 1 / 1 | 6.983e+07 / 5.162e+07 / 4.229e+07 | 7.585 / 18.96 / 20.09 |
| higgs | raw | 11,000,000 | 19.65 | 0.1646 / 1.006 / 1.4 | 2.36e+08 / 1.934e+08 / 1.679e+08 | 3.013 / 12.07 / 12.71 |
| higgs | zscore | 11,000,000 | 28 | 1 / 1 / 1 | 3.44e+08 / 2.892e+08 / 2.567e+08 | 5.641 / 8.33 / 19.28 |
| nytimes | raw | 300,000 | 930.7 | 0 / 0.02816 / 2.213 | 3.629e+08 / 3.102e+08 / 3.063e+08 | 4281 / 1.68e+04 / 3.756e+04 |
| nytimes | tfidf | 300,000 | 0.988 | 0 / 0.001757 / 0.04485 | 5.759e+05 / 5.603e+05 / 5.484e+05 | 16.11 / 4.219 / 9.718 |
| pendigits | raw | 10,992 | 1.487e+04 | 16.22 / 30.24 / 41.76 | 8.178e+07 / 4.481e+07 / 3.08e+07 | 3.185 / 7.084 / 20.74 |
| pendigits | zscore | 10,992 | 16 | 1 / 1 / 1 | 9.163e+04 / 5.33e+04 / 3.757e+04 | 3.556 / 5.179 / 18.12 |

How to read these numbers:

- **KDDCup99 is extremely unbalanced.** At every k, the seeding isolates
  single outlier connections (smallest cluster = 1 point), and one cluster
  holds most of the data. The labels fit this picture: 2,807,886 smurf,
  1,072,017 neptune and 972,781 normal rows. The raw scale is
  dominated by `src_bytes` (std 9.4e5) and `dst_bytes` (6.5e5). A uniform
  sample of any practical size misses the singleton clusters, which makes this
  dataset the stress case for sampling-based coresets.
- In the z-scored covertype the smallest-cluster sizes match the rarest one-hot
  soil types. `Soil_Type15` occurs in 3 rows, and the smallest cluster at k = 50
  has 3 points. `Soil_Type7` occurs in 105 rows, and the smallest cluster at
  k = 25 has 105. Unit variance puts such rows far
  from everything else. yearmsd has similar small clusters, which suggests
  outlying songs in timbre space, but this was not checked.
- On raw nytimes counts, the balance reaches 3.8e4. Since the raw squared
  distances grow with document length, length most likely drives this; it was
  not checked. tf-idf with unit-norm rows gives balanced clusters (4-16). Use
  `tfidf` unless raw counts are the point of an experiment.
- SUSY and HIGGS are already near unit scale and balanced (3-33). They are the
  large, well-conditioned dense cases.

Each stats run took 3 s (pendigits) to 397 s (HIGGS raw). Almost all of that
time is the three seedings.
