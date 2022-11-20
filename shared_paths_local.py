
data_folder = "data/"

# ========================== en-de ==========================
en_de_folder = data_folder + "en-de/"
TRAIN_WMT = en_de_folder + "WMT-News.de-en.chrf.ngram.lan.comet.bertScore.train"
DEV_WMT = en_de_folder + "WMT-News.de-en.chrf.ngram.lan.comet.bertScore.dev"
TRAIN_TATOEBA = en_de_folder + "Tatoeba.de-en.chrf.ngram.lan.comet.bertScore.train"
DEV_TATOEBA = en_de_folder + "Tatoeba.de-en.chrf.ngram.lan.comet.bertScore.dev"
TRAIN_NEWS20 = en_de_folder + "newstest2020.ende.chrf.ngram.lan.comet.bertScore.train"
DEV_NEWS20 = en_de_folder + "newstest2020.ende.chrf.ngram.lan.comet.bertScore.dev"
TRAIN_BIBLE = en_de_folder + "bible-uedin.de-en.chrf.ngram.lan.comet.bertScore.train"
DEV_BIBLE = en_de_folder + "bible-uedin.de-en.chrf.ngram.lan.comet.bertScore.dev"
TRAIN_GLOBAL_VOICES = en_de_folder + "GlobalVoices.de-en.chrf.ngram.lan.comet.bertScore.train"
DEV_GLOBAL_VOICES = en_de_folder + "GlobalVoices.de-en.chrf.ngram.lan.comet.bertScore.dev"
TRAIN_DA = en_de_folder + "DA.ende.df.train.ngram.lan.hter.fix.fix_hter"
DEV_DA = en_de_folder + "DA.ende.df.dev.ngram.lan.hter.fix.fix_hter"


# facebook
facebook_folder = en_de_folder + "facebook/"
FACEBOOK_TRAIN_WMT = facebook_folder + "WMT-News.de-en.chrf.ngram.lan.comet.bertScore.train"
FACEBOOK_DEV_WMT = facebook_folder + "WMT-News.de-en.chrf.ngram.lan.comet.bertScore.dev"
FACEBOOK_TRAIN_TATOEBA = facebook_folder + "Tatoeba.de-en.chrf.ngram.lan.comet.bertScore.train"
FACEBOOK_DEV_TATOEBA = facebook_folder + "Tatoeba.de-en.chrf.ngram.lan.comet.bertScore.dev"
FACEBOOK_TRAIN_NEWS20 = facebook_folder + "newstest2020.ende.chrf.ngram.lan.comet.bertScore.train"
FACEBOOK_DEV_NEWS20 = facebook_folder + "newstest2020.ende.chrf.ngram.lan.comet.bertScore.dev"
FACEBOOK_TRAIN_BIBLE = facebook_folder + "bible-uedin.de-en.chrf.ngram.lan.comet.bertScore.train"
FACEBOOK_DEV_BIBLE = facebook_folder + "bible-uedin.de-en.chrf.ngram.lan.comet.bertScore.dev"
FACEBOOK_TRAIN_GLOBAL_VOICES = facebook_folder + "GlobalVoices.de-en.chrf.ngram.lan.comet.bertScore.train"
FACEBOOK_DEV_GLOBAL_VOICES = facebook_folder + "GlobalVoices.de-en.chrf.ngram.lan.comet.bertScore.dev"

# WMT19
WMT19_TRAIN = en_de_folder + "wmt19_train.csv"
WMT19_DEV = en_de_folder + "wmt19_dev.csv"

# ========================== en-zh ==========================
en_zh_folder = data_folder + "en-zh/"
zh_wmt = "WMT-News.en-zh.en.ngram.lan.comet.bertScore"
zh_news20 = "newstestB2020-enzh.en.ngram.lan.comet.bertScore"
zh_bible = "bible-uedin.en-zh.en.ngram.lan.comet.bertScore"
ZH_TRAIN_WMT = en_zh_folder + zh_wmt + ".aligned.train"
ZH_DEV_WMT = en_zh_folder + zh_wmt + ".aligned.dev"
ZH_TRAIN_NEWS20 = en_zh_folder + zh_news20 + ".aligned.train"
ZH_DEV_NEWS20 = en_zh_folder + zh_news20 + ".aligned.dev"
ZH_TRAIN_BIBLE = en_zh_folder + zh_bible + ".aligned.train"
ZH_DEV_BIBLE = en_zh_folder + zh_bible + ".aligned.dev"
zh_da = "DA.enzh.ngram.lan.hter"
ZH_DA_TRAIN = en_zh_folder + zh_da + ".train"
ZH_DA_DEV = en_zh_folder + zh_da + ".dev"

# ========================== de-en ==========================
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

# ========================== et-en ==========================
et_en_folder = data_folder + "et-en/"
ET_ET_TRAIN_DA = et_en_folder + "train.eten.df.short.tsv"
ET_ET_DEV_DA = et_en_folder + "dev.eten.df.short.tsv"

# more
CACHE_DIR = "cache/"