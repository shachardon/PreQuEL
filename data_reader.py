import csv
import pandas as pd


def read_annotated_file(path, index="index", comet=True, bertScore=True, features=False, ref=False, hter=False,
                        replace_hter=False, replace_comet=False):
    indices = []
    originals = []
    translations = []
    z_means = []
    refs = []
    comets = []
    bertScores = []
    hters = []
    lans = []
    unigrams = []
    bigrams = []
    grams3 = []
    grams4 = []
    grams5 = []
    with open(path, mode="r", encoding="utf-8-sig") as csvfile:
        reader = csv.DictReader(csvfile, delimiter="\t", quoting=csv.QUOTE_NONE)
        for row in reader:
            indices.append(row[index])
            originals.append(row["original"])
            translations.append(row["translation"])
            z_means.append(float(row["z_mean"]))
            if ref:
                refs.append(row["ref"])
            if comet:
                comets.append(float(row["comet"]))
            if bertScore:
                bertScores.append(float(row["bertScore"]))
            if hter:
                hters.append(float(row["hter"]))
            if features:
                lans.append(float(row["lan"]))
                unigrams.append(float(row["unigram"]))
                bigrams.append(float(row["bigram"]))
                grams3.append(float(row["3gram"]))
                grams4.append(float(row["4gram"]))
                grams5.append(float(row["5gram"]))
            else:
                lans.append(0)
                unigrams.append(0)
                bigrams.append(0)
                grams3.append(0)
                grams4.append(0)
                grams5.append(0)

                # print(row["5gram"])

    df = pd.DataFrame(
        {'index': indices,
         'original': originals,
         'translation': translations,
         'z_mean': z_means
         })

    if ref:
        df["original"] = refs  # override the source sentence.

    if comet:
        df['comet'] = comets

    if bertScore:
        df['bertScore'] = bertScores

    if hter:
        df['hter'] = hters
        df['hter'] = -df['hter']

    # if features:
    df['unigram'] = unigrams
    df['bigram'] = bigrams
    df['3gram'] = grams3
    df['4gram'] = grams4
    df['5gram'] = grams5
    df['lan'] = lans

    # print(df)

    if replace_hter:
        tmp_hter = df['hter'].copy()
        df['hter'] = df['z_mean'].copy()
        df['z_mean'] = tmp_hter

    if replace_comet:
        tmp_comet = df['comet'].copy()
        df['comet'] = df['z_mean'].copy()
        df['z_mean'] = tmp_comet

    return df


def read_test_file(path, index="index"):
    indices = []
    originals = []
    translations = []
    with open(path, mode="r", encoding="utf-8-sig") as csvfile:
        reader = csv.DictReader(csvfile, delimiter="\t", quoting=csv.QUOTE_NONE)
        for row in reader:
            indices.append(row[index])
            originals.append(row["original"])
            translations.append(row["translation"])

    return pd.DataFrame(
        {'index': indices,
         'original': originals,
         'translation': translations,
         })
