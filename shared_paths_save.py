# data

TRAIN_WMT = "/cs/snapless/oabend/shachar.don/data/data_for_marian/WMT-News.de-en.chrf.ngram.lan.comet.bertScore.train"
DEV_WMT = "/cs/snapless/oabend/shachar.don/data/data_for_marian/WMT-News.de-en.chrf.ngram.lan.comet.bertScore.dev"

TRAIN_TATOEBA = "/cs/snapless/oabend/shachar.don/data/data_for_marian/Tatoeba.de-en.chrf.ngram.lan.comet.bertScore.train"
DEV_TATOEBA = "/cs/snapless/oabend/shachar.don/data/data_for_marian/Tatoeba.de-en.chrf.ngram.lan.comet.bertScore.dev"

TRAIN_NEWS20 = "/cs/snapless/oabend/shachar.don/data/data_for_marian/newstest2020.ende.chrf.ngram.lan.comet.bertScore.train"
DEV_NEWS20 = "/cs/snapless/oabend/shachar.don/data/data_for_marian/newstest2020.ende.chrf.ngram.lan.comet.bertScore.dev"

TRAIN_BIBLE = "/cs/snapless/oabend/shachar.don/data/data_for_marian/bible-uedin.de-en.chrf.ngram.lan.comet.bertScore.train"
DEV_BIBLE = "/cs/snapless/oabend/shachar.don/data/data_for_marian/bible-uedin.de-en.chrf.ngram.lan.comet.bertScore.dev"

TRAIN_GLOBAL_VOICES = "/cs/snapless/oabend/shachar.don/data/data_for_marian/GlobalVoices.de-en.chrf.ngram.lan.comet.bertScore.train"
DEV_GLOBAL_VOICES = "/cs/snapless/oabend/shachar.don/data/data_for_marian/GlobalVoices.de-en.chrf.ngram.lan.comet.bertScore.dev"

TRAIN_DA = "/cs/snapless/oabend/shachar.don/data/data_for_marian/with_comet/DA.ende.df.train.ngram.lan.hter.fix.fix_hter"
DEV_DA = "/cs/snapless/oabend/shachar.don/data/data_for_marian/with_comet/DA.ende.df.dev.ngram.lan.hter.fix.fix_hter"

# more
CACHE_DIR = "/cs/labs/oabend/shachar.don/pre-translationQE/my_model/cache"


# facebook
FACEBOOK_TRAIN_WMT = "/cs/snapless/oabend/shachar.don/data/data_for_marian/facebook_output/WMT-News.de-en.chrf.ngram.lan.comet.bertScore.train"
FACEBOOK_DEV_WMT = "/cs/snapless/oabend/shachar.don/data/data_for_marian/facebook_output/WMT-News.de-en.chrf.ngram.lan.comet.bertScore.dev"
FACEBOOK_TRAIN_TATOEBA = "/cs/snapless/oabend/shachar.don/data/data_for_marian/facebook_output/Tatoeba.de-en.chrf.ngram.lan.comet.bertScore.train"
FACEBOOK_DEV_TATOEBA = "/cs/snapless/oabend/shachar.don/data/data_for_marian/facebook_output/Tatoeba.de-en.chrf.ngram.lan.comet.bertScore.dev"
FACEBOOK_TRAIN_NEWS20 = "/cs/snapless/oabend/shachar.don/data/data_for_marian/facebook_output/newstest2020.ende.chrf.ngram.lan.comet.bertScore.train"
FACEBOOK_DEV_NEWS20 = "/cs/snapless/oabend/shachar.don/data/data_for_marian/facebook_output/newstest2020.ende.chrf.ngram.lan.comet.bertScore.dev"
FACEBOOK_TRAIN_BIBLE = "/cs/snapless/oabend/shachar.don/data/data_for_marian/facebook_output/bible-uedin.de-en.chrf.ngram.lan.comet.bertScore.train"
FACEBOOK_DEV_BIBLE = "/cs/snapless/oabend/shachar.don/data/data_for_marian/facebook_output/bible-uedin.de-en.chrf.ngram.lan.comet.bertScore.dev"
FACEBOOK_TRAIN_GLOBAL_VOICES = "/cs/snapless/oabend/shachar.don/data/data_for_marian/facebook_output/GlobalVoices.de-en.chrf.ngram.lan.comet.bertScore.train"
FACEBOOK_DEV_GLOBAL_VOICES = "/cs/snapless/oabend/shachar.don/data/data_for_marian/facebook_output/GlobalVoices.de-en.chrf.ngram.lan.comet.bertScore.dev"

# zh
zh_dir = "/cs/snapless/oabend/shachar.don/data/data_for_marian/zh_data/"
zh_wmt = "WMT-News.en-zh.en.ngram.lan.comet.bertScore"
zh_news20 = "newstestB2020-enzh.en.ngram.lan.comet.bertScore"
zh_bible = "bible-uedin.en-zh.en.ngram.lan.comet.bertScore"
ZH_TRAIN_WMT = zh_dir + zh_wmt + ".aligned.train"
ZH_DEV_WMT = zh_dir + zh_wmt + ".aligned.dev"
ZH_TRAIN_NEWS20 = zh_dir + zh_news20 + ".aligned.train"
ZH_DEV_NEWS20 = zh_dir + zh_news20 + ".aligned.dev"
ZH_TRAIN_BIBLE = zh_dir + zh_bible + ".aligned.train"
ZH_DEV_BIBLE = zh_dir + zh_bible + ".aligned.dev"

# zh DA
zh_da = "DA.enzh.ngram.lan.hter"
ZH_DA_TRAIN = zh_dir + zh_da + ".train"
ZH_DA_DEV = zh_dir + zh_da + ".dev"

# en
reverse_dir = "/cs/snapless/oabend/shachar.don/data/data_for_marian/reverse/"
reverse_wmt = "WMT-News.de-en"
reverse_news20 = "newstest2020.ende"
reverse_bible = "bible-uedin.de-en"
reverse_tatoeba = "Tatoeba.de-en"
reverse_global_voices = "GlobalVoices.de-en"
reverse_suffix = ".chrf.comet.bertScore"
RE_TRAIN_WMT = reverse_dir + reverse_wmt + reverse_suffix + ".train"
RE_DEV_WMT = reverse_dir + reverse_wmt + reverse_suffix + ".dev"
RE_TEST_WMT = reverse_dir + reverse_wmt + reverse_suffix + ".test"
RE_TRAIN_NEWS20 = reverse_dir + reverse_news20 + reverse_suffix + ".train"
RE_DEV_NEWS20 = reverse_dir + reverse_news20 + reverse_suffix + ".dev"
RE_TEST_NEWS20 = reverse_dir + reverse_news20 + reverse_suffix + ".test"
RE_TRAIN_BIBLE = reverse_dir + reverse_bible + reverse_suffix + ".train"
RE_DEV_BIBLE = reverse_dir + reverse_bible + reverse_suffix + ".dev"
RE_TEST_BIBLE = reverse_dir + reverse_bible + reverse_suffix + ".test"
RE_TRAIN_TATOEBA = reverse_dir + reverse_tatoeba + reverse_suffix + ".train"
RE_DEV_TATOEBA = reverse_dir + reverse_tatoeba + reverse_suffix + ".dev"
RE_TEST_TATOEBA = reverse_dir + reverse_tatoeba + reverse_suffix + ".test"
RE_TRAIN_GLOBAL = reverse_dir + reverse_global_voices + reverse_suffix + ".train"
RE_DEV_GLOBAL = reverse_dir + reverse_global_voices + reverse_suffix + ".dev"
RE_TEST_GLOBAL = reverse_dir + reverse_global_voices + reverse_suffix + ".test"

# et-en
ET_ET_TRAIN_DA = "/cs/snapless/oabend/shachar.don/data/et-en/train.eten.df.short.tsv"
ET_ET_DEV_DA = "/cs/snapless/oabend/shachar.don/data/et-en/dev.eten.df.short.tsv"

# WMT19
WMT19_TRAIN = "/cs/snapless/oabend/shachar.don/data/task1_en-de_traindev/train/wmt19_train.csv"
WMT19_DEV = "/cs/snapless/oabend/shachar.don/data/task1_en-de_traindev/dev/wmt19_dev.csv"