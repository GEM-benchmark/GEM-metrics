

'''
This is the main list of splits that are supported. The naming schema follows
{DATASET_NAME}_{SPLIT_NAME}.
For validation sets, both `val` and `validation` are allowed, but the
corresponding filename should use `validation` - this was introduced to maintain
backwards compability.

Should the dataset not be found at the corresponding URL, the results will be
empty.

In this file, we first construct the _SUPPORTED DATASETS with all the
GEM-supported datasets. If you want to use a non-supported dataset, simply
add it to the dictionary _after_ the automatic URL addition code below.
'''

from gem_metrics import data


_SUPPORTED_TASKS = {
    "sportsett_basketball": {
        "language":
            "en",
        "task":
            "data2text",
        "challenge_sets": [
            "sportsett_basketball_test", "sportsett_basketball_validation"
        ]
    },
    "turku_hockey_data2text": {
        "language":
            "fi",
        "task":
            "data2text",
        "challenge_sets": [
            "turku_hockey_data2text_test", "turku_hockey_data2text_validation"
        ]
    },
    "squad_v2": {
        "language": "en",
        "task": "question_generation",
        "challenge_sets": ["squad_v2_test", "squad_v2_validation"]
    },
    "surface_realisation_st_2020": {
        "language":
            "en",
        "task":
            "data2text",
        "challenge_sets": [
            "surface_realisation_st_2020_test",
            "surface_realisation_st_2020_validation"
        ]
    },
    "RotoWire_English_German": {
        "language":
            "de",
        "task":
            "data2text",
        "challenge_sets": [
            "RotoWire_English_German_test", "RotoWire_English_German_validation"
        ]
    },
    "Taskmaster": {
        "language": "en",
        "task": "dialog",
        "challenge_sets": ["Taskmaster_test", "Taskmaster_validation"]
    },
    "dstc10_track2_task2": {
        "language":
            "en",
        "task":
            "data2text",
        "challenge_sets": [
            "dstc10_track2_task2_test", "dstc10_track2_task2_validation"
        ]
    },
    "ART": {
        "language": "en",
        "task": "data2text",
        "challenge_sets": ["ART_test", "ART_validation"]
    },
    "conversational_weather": {
        "language":
            "en",
        "task":
            "data2text",
        "challenge_sets": [
            "conversational_weather_test", "conversational_weather_validation"
        ]
    },
    "BiSECT_de": {
        "language": "de",
        "task": "simplification",
        "challenge_sets": ["BiSECT_de_test", "BiSECT_de_validation"]
    },
    "BiSECT_en": {
        "language":
            "en",
        "task":
            "simplification",
        "challenge_sets": [
            "BiSECT_en_challenge_bisect", "BiSECT_en_challenge_hsplit",
            "BiSECT_en_test", "BiSECT_en_validation"
        ]
    },
    "BiSECT_es": {
        "language": "es",
        "task": "simplification",
        "challenge_sets": ["BiSECT_es_test", "BiSECT_es_validation"]
    },
    "BiSECT_fr": {
        "language": "fr",
        "task": "simplification",
        "challenge_sets": ["BiSECT_fr_test", "BiSECT_fr_validation"]
    },
    "SIMPITIKI": {
        "language":
            "en",
        "task":
            "simplification",
        "challenge_sets": [
            "SIMPITIKI_challenge_itwiki_test",
            "SIMPITIKI_challenge_itwiki_train",
            "SIMPITIKI_challenge_itwiki_val",
            "SIMPITIKI_challenge_seen_transformations_test",
            "SIMPITIKI_challenge_seen_transformations_train",
            "SIMPITIKI_challenge_seen_transformations_val",
            "SIMPITIKI_challenge_tn_test",
            "SIMPITIKI_challenge_unseen_transformations_test", "SIMPITIKI_test",
            "SIMPITIKI_validation"
        ]
    },
    "cochrane_simplification": {
        "language":
            "en",
        "task":
            "simplification",
        "challenge_sets": [
            "cochrane_simplification_test", "cochrane_simplification_validation"
        ]
    },
    "indonlg_indosum": {
        "language": "en",
        "task": "summarization",
        "challenge_sets": [
            "indonlg_indosum_test", "indonlg_indosum_validation"
        ]
    },
    "mlb_data_to_text": {
        "language":
            "en",
        "task":
            "data2text",
        "challenge_sets": [
            "mlb_data_to_text_test", "mlb_data_to_text_validation"
        ]
    },
    "opusparcus_de_80": {
        "language":
            "en",
        "task":
            "paraphrasing",
        "challenge_sets": [
            "opusparcus_de_80_test", "opusparcus_de_80_validation"
        ]
    },
    "opusparcus_en_80": {
        "language":
            "en",
        "task":
            "paraphrasing",
        "challenge_sets": [
            "opusparcus_en_80_test", "opusparcus_en_80_validation"
        ]
    },
    "opusparcus_fi_80": {
        "language":
            "en",
        "task":
            "paraphrasing",
        "challenge_sets": [
            "opusparcus_fi_80_test", "opusparcus_fi_80_validation"
        ]
    },
    "opusparcus_fr_80": {
        "language":
            "en",
        "task":
            "paraphrasing",
        "challenge_sets": [
            "opusparcus_fr_80_test", "opusparcus_fr_80_validation"
        ]
    },
    "opusparcus_ru_80": {
        "language":
            "en",
        "task":
            "paraphrasing",
        "challenge_sets": [
            "opusparcus_ru_80_test", "opusparcus_ru_80_validation"
        ]
    },
    "opusparcus_sv_80": {
        "language":
            "en",
        "task":
            "paraphrasing",
        "challenge_sets": [
            "opusparcus_sv_80_test", "opusparcus_sv_80_validation"
        ]
    },
    "turku_paraphrase_corpus": {
        "language":
            "en",
        "task":
            "paraphrasing",
        "challenge_sets": [
            "turku_paraphrase_corpus_test", "turku_paraphrase_corpus_validation"
        ]
    },
    "viggo": {
        "language":
            "en",
        "task":
            "data2text",
        "challenge_sets": [
            "viggo_challenge_train_10_percent",
            "viggo_challenge_train_1_percent",
            "viggo_challenge_train_20_percent",
            "viggo_challenge_train_2_percent",
            "viggo_challenge_train_5_percent", "viggo_test", "viggo_validation"
        ]
    },
    "common_gen": {
        "language":
            "en",
        "task":
            "data2text",
        "challenge_sets": [
            "common_gen_val", "common_gen_test",
            "common_gen_challenge_test_scramble",
            "common_gen_challenge_train_sample",
            "common_gen_challenge_validation_sample"
        ]
    },
    "cs_restaurants": {
        "language":
            "cs",
        "task":
            "data2text",
        "challenge_sets": [
            "cs_restaurants_val", "cs_restaurants_test",
            "cs_restaurants_challenge_test_scramble",
            "cs_restaurants_challenge_train_sample",
            "cs_restaurants_challenge_validation_sample"
        ]
    },
    "e2e_nlg": {
        "language":
            "en",
        "task":
            "data2text",
        "challenge_sets": [
            "e2e_nlg_val", "e2e_nlg_test", "e2e_nlg_challenge_test_scramble",
            "e2e_nlg_challenge_train_sample",
            "e2e_nlg_challenge_validation_sample"
        ]
    },
    "mlsum_de": {
        "language":
            "de",
        "task":
            "summarization",
        "challenge_sets": [
            "mlsum_de_val", "mlsum_de_test", "mlsum_de_test_5000",
            "mlsum_de_challenge_test_covid", "mlsum_de_challenge_train_sample",
            "mlsum_de_challenge_validation_sample"
        ]
    },
    "mlsum_es": {
        "language":
            "es",
        "task":
            "summarization",
        "challenge_sets": [
            "mlsum_es_val", "mlsum_es_test", "mlsum_es_test_5000",
            "mlsum_es_challenge_test_covid", "mlsum_es_challenge_train_sample",
            "mlsum_es_challenge_validation_sample"
        ]
    },
    "schema_guided_dialog": {
        "language":
            "en",
        "task":
            "dialog",
        "challenge_sets": [
            "schema_guided_dialog_val", "schema_guided_dialog_test",
            "schema_guided_dialog_challenge_test_backtranslation",
            "schema_guided_dialog_challenge_test_bfp02",
            "schema_guided_dialog_challenge_test_bfp05",
            "schema_guided_dialog_challenge_test_nopunc",
            "schema_guided_dialog_challenge_test_scramble",
            "schema_guided_dialog_challenge_train_sample",
            "schema_guided_dialog_challenge_validation_sample"
        ]
    },
    "totto": {
        "language":
            "en",
        "task":
            "data2text",
        "challenge_sets": [
            "totto_val", "totto_test", "totto_challenge_test_scramble",
            "totto_challenge_train_sample", "totto_challenge_validation_sample"
        ]
    },
    "web_nlg_en": {
        "language":
            "en",
        "task":
            "data2text",
        "challenge_sets": [
            "web_nlg_en_val", "web_nlg_en_test",
            "web_nlg_en_challenge_test_scramble",
            "web_nlg_en_challenge_test_numbers",
            "web_nlg_en_challenge_train_sample",
            "web_nlg_en_challenge_validation_sample"
        ]
    },
    "web_nlg_ru": {
        "language":
            "ru",
        "task":
            "data2text",
        "challenge_sets": [
            "web_nlg_ru_val", "web_nlg_ru_test",
            "web_nlg_ru_challenge_test_scramble",
            "web_nlg_ru_challenge_train_sample",
            "web_nlg_ru_challenge_validation_sample"
        ]
    },
    "wiki_auto_asset_turk": {
        "language":
            "en",
        "task":
            "text_simplification",
        "challenge_sets": [
            "wiki_auto_asset_turk_val", "wiki_auto_asset_turk_test_asset",
            "wiki_auto_asset_turk_test_turk",
            "wiki_auto_asset_turk_challenge_test_asset_backtranslation",
            "wiki_auto_asset_turk_challenge_test_asset_bfp02",
            "wiki_auto_asset_turk_challenge_test_asset_bfp05",
            "wiki_auto_asset_turk_challenge_test_asset_nopunc",
            "wiki_auto_asset_turk_challenge_test_turk_backtranslation",
            "wiki_auto_asset_turk_challenge_test_turk_bfp02",
            "wiki_auto_asset_turk_challenge_test_turk_bfp05",
            "wiki_auto_asset_turk_challenge_test_turk_nopunc",
            "wiki_auto_asset_turk_challenge_train_sample",
            "wiki_auto_asset_turk_challenge_validation_sample"
        ]
    },
    "wiki_lingua_english_en": {
        "language":
            "en",
        "task":
            "summarization",
        "challenge_sets": [
            "wiki_lingua_english_en_val", "wiki_lingua_english_en_test",
            "wiki_lingua_english_en_test_5000"
        ]
    },
    "wiki_lingua_russian_ru": {
        "language":
            "en",
        "task":
            "summarization",
        "challenge_sets": [
            "wiki_lingua_russian_ru_val", "wiki_lingua_russian_ru_test",
            "wiki_lingua_russian_ru_test_5000"
        ]
    },
    "wiki_lingua_spanish_es": {
        "language":
            "en",
        "task":
            "summarization",
        "challenge_sets": [
            "wiki_lingua_spanish_es_val", "wiki_lingua_spanish_es_test",
            "wiki_lingua_spanish_es_test_5000"
        ]
    },
    "wiki_lingua_turkish_tr": {
        "language":
            "en",
        "task":
            "summarization",
        "challenge_sets": [
            "wiki_lingua_turkish_tr_val", "wiki_lingua_turkish_tr_test",
            "wiki_lingua_turkish_tr_test_5000"
        ]
    },
    "wiki_lingua_vietnamese_vi": {
        "language":
            "en",
        "task":
            "summarization",
        "challenge_sets": [
            "wiki_lingua_vietnamese_vi_val", "wiki_lingua_vietnamese_vi_test",
            "wiki_lingua_vietnamese_vi_test_5000"
        ]
    },
    "xsum": {
        "language":
            "en",
        "task":
            "summarization",
        "challenge_sets": [
            "xsum_val", "xsum_test", "xsum_challenge_test_backtranslation",
            "xsum_challenge_test_bfp_02", "xsum_challenge_test_bfp_05",
            "xsum_challenge_test_nopunc", "xsum_challenge_test_covid",
            "xsum_challenge_train_sample", "xsum_challenge_validation_sample"
        ]
    },
    "OrangeSum_abstract": {
        "language":
            "fr",
        "task":
            "summarization",
        "challenge_sets": [
            "OrangeSum_abstract_test", "OrangeSum_abstract_validation"
        ]
    },
    "OrangeSum_title": {
        "language": "fr",
        "task": "summarization",
        "challenge_sets": [
            "OrangeSum_title_test", "OrangeSum_title_validation"
        ]
    },
    "wiki_lingua_ar": {
        "language": "ar",
        "task": "summarization",
        "challenge_sets": ["wiki_lingua_ar_test", "wiki_lingua_ar_validation"]
    },
    "wiki_lingua_cs": {
        "language": "cs",
        "task": "summarization",
        "challenge_sets": ["wiki_lingua_cs_test", "wiki_lingua_cs_validation"]
    },
    "wiki_lingua_de": {
        "language": "de",
        "task": "summarization",
        "challenge_sets": ["wiki_lingua_de_test", "wiki_lingua_de_validation"]
    },
    "wiki_lingua_en": {
        "language": "en",
        "task": "summarization",
        "challenge_sets": ["wiki_lingua_en_test", "wiki_lingua_en_validation"]
    },
    "wiki_lingua_es": {
        "language": "es",
        "task": "summarization",
        "challenge_sets": ["wiki_lingua_es_test", "wiki_lingua_es_validation"]
    },
    "wiki_lingua_fr": {
        "language": "fr",
        "task": "summarization",
        "challenge_sets": ["wiki_lingua_fr_test", "wiki_lingua_fr_validation"]
    },
    "wiki_lingua_hi": {
        "language": "hi",
        "task": "summarization",
        "challenge_sets": ["wiki_lingua_hi_test", "wiki_lingua_hi_validation"]
    },
    "wiki_lingua_id": {
        "language": "id",
        "task": "summarization",
        "challenge_sets": ["wiki_lingua_id_test", "wiki_lingua_id_validation"]
    },
    "wiki_lingua_it": {
        "language": "it",
        "task": "summarization",
        "challenge_sets": ["wiki_lingua_it_test", "wiki_lingua_it_validation"]
    },
    "wiki_lingua_ja": {
        "language": "ja",
        "task": "summarization",
        "challenge_sets": ["wiki_lingua_ja_test", "wiki_lingua_ja_validation"]
    },
    "wiki_lingua_ko": {
        "language": "ko",
        "task": "summarization",
        "challenge_sets": ["wiki_lingua_ko_test", "wiki_lingua_ko_validation"]
    },
    "wiki_lingua_nl": {
        "language": "nl",
        "task": "summarization",
        "challenge_sets": ["wiki_lingua_nl_test", "wiki_lingua_nl_validation"]
    },
    "wiki_lingua_pt": {
        "language": "pt",
        "task": "summarization",
        "challenge_sets": ["wiki_lingua_pt_test", "wiki_lingua_pt_validation"]
    },
    "wiki_lingua_ru": {
        "language": "ru",
        "task": "summarization",
        "challenge_sets": ["wiki_lingua_ru_test", "wiki_lingua_ru_validation"]
    },
    "wiki_lingua_th": {
        "language": "th",
        "task": "summarization",
        "challenge_sets": ["wiki_lingua_th_test", "wiki_lingua_th_validation"]
    },
    "wiki_lingua_tr": {
        "language": "tr",
        "task": "summarization",
        "challenge_sets": ["wiki_lingua_tr_test", "wiki_lingua_tr_validation"]
    },
    "wiki_lingua_vi": {
        "language": "vi",
        "task": "summarization",
        "challenge_sets": ["wiki_lingua_vi_test", "wiki_lingua_vi_validation"]
    },
    "wiki_lingua_zh": {
        "language": "zh",
        "task": "summarization",
        "challenge_sets": ["wiki_lingua_zh_test", "wiki_lingua_zh_validation"]
    },
    "xlsum_amharic": {
        "language": "amharic",
        "task": "summarization",
        "challenge_sets": ["xlsum_amharic_test", "xlsum_amharic_validation"]
    },
    "xlsum_arabic": {
        "language": "arabic",
        "task": "summarization",
        "challenge_sets": ["xlsum_arabic_test", "xlsum_arabic_validation"]
    },
    "xlsum_azerbaijani": {
        "language":
            "azerbaijani",
        "task":
            "summarization",
        "challenge_sets": [
            "xlsum_azerbaijani_test", "xlsum_azerbaijani_validation"
        ]
    },
    "xlsum_bengali": {
        "language": "bengali",
        "task": "summarization",
        "challenge_sets": ["xlsum_bengali_test", "xlsum_bengali_validation"]
    },
    "xlsum_burmese": {
        "language": "burmese",
        "task": "summarization",
        "challenge_sets": ["xlsum_burmese_test", "xlsum_burmese_validation"]
    },
    "xlsum_chinese_simplified": {
        "language":
            "chinese_simplified",
        "task":
            "summarization",
        "challenge_sets": [
            "xlsum_chinese_simplified_test",
            "xlsum_chinese_simplified_validation"
        ]
    },
    "xlsum_chinese_traditional": {
        "language":
            "chinese_traditional",
        "task":
            "summarization",
        "challenge_sets": [
            "xlsum_chinese_traditional_test",
            "xlsum_chinese_traditional_validation"
        ]
    },
    "xlsum_english": {
        "language": "en",
        "task": "summarization",
        "challenge_sets": ["xlsum_english_test", "xlsum_english_validation"]
    },
    "xlsum_french": {
        "language": "fr",
        "task": "summarization",
        "challenge_sets": ["xlsum_french_test", "xlsum_french_validation"]
    },
    "xlsum_gujarati": {
        "language": "gu",
        "task": "summarization",
        "challenge_sets": ["xlsum_gujarati_test", "xlsum_gujarati_validation"]
    },
    "xlsum_hausa": {
        "language": "ha",
        "task": "summarization",
        "challenge_sets": ["xlsum_hausa_test", "xlsum_hausa_validation"]
    },
    "xlsum_hindi": {
        "language": "hi",
        "task": "summarization",
        "challenge_sets": ["xlsum_hindi_test", "xlsum_hindi_validation"]
    },
    "xlsum_igbo": {
        "language": "igbo",
        "task": "summarization",
        "challenge_sets": ["xlsum_igbo_test", "xlsum_igbo_validation"]
    },
    "xlsum_indonesian": {
        "language":
            "indonesian",
        "task":
            "summarization",
        "challenge_sets": [
            "xlsum_indonesian_test", "xlsum_indonesian_validation"
        ]
    },
    "xlsum_japanese": {
        "language": "ja",
        "task": "summarization",
        "challenge_sets": ["xlsum_japanese_test", "xlsum_japanese_validation"]
    },
    "xlsum_kirundi": {
        "language": "kirundi",
        "task": "summarization",
        "challenge_sets": ["xlsum_kirundi_test", "xlsum_kirundi_validation"]
    },
    "xlsum_korean": {
        "language": "ko",
        "task": "summarization",
        "challenge_sets": ["xlsum_korean_test", "xlsum_korean_validation"]
    },
    "xlsum_kyrgyz": {
        "language": "kyrgyz",
        "task": "summarization",
        "challenge_sets": ["xlsum_kyrgyz_test", "xlsum_kyrgyz_validation"]
    },
    "xlsum_marathi": {
        "language": "marathi",
        "task": "summarization",
        "challenge_sets": ["xlsum_marathi_test", "xlsum_marathi_validation"]
    },
    "xlsum_nepali": {
        "language": "nepali",
        "task": "summarization",
        "challenge_sets": ["xlsum_nepali_test", "xlsum_nepali_validation"]
    },
    "xlsum_oromo": {
        "language": "oromo",
        "task": "summarization",
        "challenge_sets": ["xlsum_oromo_test", "xlsum_oromo_validation"]
    },
    "xlsum_pashto": {
        "language": "pashto",
        "task": "summarization",
        "challenge_sets": ["xlsum_pashto_test", "xlsum_pashto_validation"]
    },
    "xlsum_persian": {
        "language": "persian",
        "task": "summarization",
        "challenge_sets": ["xlsum_persian_test", "xlsum_persian_validation"]
    },
    "xlsum_pidgin": {
        "language": "pidgin",
        "task": "summarization",
        "challenge_sets": ["xlsum_pidgin_test", "xlsum_pidgin_validation"]
    },
    "xlsum_portuguese": {
        "language":
            "pt",
        "task":
            "summarization",
        "challenge_sets": [
            "xlsum_portuguese_test", "xlsum_portuguese_validation"
        ]
    },
    "xlsum_punjabi": {
        "language": "punjabi",
        "task": "summarization",
        "challenge_sets": ["xlsum_punjabi_test", "xlsum_punjabi_validation"]
    },
    "xlsum_russian": {
        "language": "ru",
        "task": "summarization",
        "challenge_sets": ["xlsum_russian_test", "xlsum_russian_validation"]
    },
    "xlsum_scottish_gaelic": {
        "language":
            "scottish_gaelic",
        "task":
            "summarization",
        "challenge_sets": [
            "xlsum_scottish_gaelic_test", "xlsum_scottish_gaelic_validation"
        ]
    },
    "xlsum_serbian_cyrillic": {
        "language":
            "serbian_cyrillic",
        "task":
            "summarization",
        "challenge_sets": [
            "xlsum_serbian_cyrillic_test", "xlsum_serbian_cyrillic_validation"
        ]
    },
    "xlsum_serbian_latin": {
        "language":
            "serbian_latin",
        "task":
            "summarization",
        "challenge_sets": [
            "xlsum_serbian_latin_test", "xlsum_serbian_latin_validation"
        ]
    },
    "xlsum_sinhala": {
        "language": "sinhala",
        "task": "summarization",
        "challenge_sets": ["xlsum_sinhala_test", "xlsum_sinhala_validation"]
    },
    "xlsum_somali": {
        "language": "somali",
        "task": "summarization",
        "challenge_sets": ["xlsum_somali_test", "xlsum_somali_validation"]
    },
    "xlsum_spanish": {
        "language": "es",
        "task": "summarization",
        "challenge_sets": ["xlsum_spanish_test", "xlsum_spanish_validation"]
    },
    "xlsum_swahili": {
        "language": "sw",
        "task": "summarization",
        "challenge_sets": ["xlsum_swahili_test", "xlsum_swahili_validation"]
    },
    "xlsum_tamil": {
        "language": "tamil",
        "task": "summarization",
        "challenge_sets": ["xlsum_tamil_test", "xlsum_tamil_validation"]
    },
    "xlsum_telugu": {
        "language": "telugu",
        "task": "summarization",
        "challenge_sets": ["xlsum_telugu_test", "xlsum_telugu_validation"]
    },
    "xlsum_thai": {
        "language": "th",
        "task": "summarization",
        "challenge_sets": ["xlsum_thai_test", "xlsum_thai_validation"]
    },
    "xlsum_tigrinya": {
        "language": "tigrinya",
        "task": "summarization",
        "challenge_sets": ["xlsum_tigrinya_test", "xlsum_tigrinya_validation"]
    },
    "xlsum_turkish": {
        "language": "turkish",
        "task": "summarization",
        "challenge_sets": ["xlsum_turkish_test", "xlsum_turkish_validation"]
    },
    "xlsum_ukrainian": {
        "language": "ukrainian",
        "task": "summarization",
        "challenge_sets": [
            "xlsum_ukrainian_test", "xlsum_ukrainian_validation"
        ]
    },
    "xlsum_urdu": {
        "language": "urdu",
        "task": "summarization",
        "challenge_sets": ["xlsum_urdu_test", "xlsum_urdu_validation"]
    },
    "xlsum_uzbek": {
        "language": "uzbek",
        "task": "summarization",
        "challenge_sets": ["xlsum_uzbek_test", "xlsum_uzbek_validation"]
    },
    "xlsum_vietnamese": {
        "language":
            "vi",
        "task":
            "summarization",
        "challenge_sets": [
            "xlsum_vietnamese_test", "xlsum_vietnamese_validation"
        ]
    },
    "xlsum_welsh": {
        "language": "welsh",
        "task": "summarization",
        "challenge_sets": ["xlsum_welsh_test", "xlsum_welsh_validation"]
    },
    "xlsum_yoruba": {
        "language": "yo",
        "task": "summarization",
        "challenge_sets": ["xlsum_yoruba_test", "xlsum_yoruba_validation"]
    },
    "wiki_cat_sum_animal": {
        "language":
            "en",
        "task":
            "summarization",
        "challenge_sets": [
            "wiki_cat_sum_animal_challenge_test_topic_abstractivity_3",
            "wiki_cat_sum_animal_challenge_test_topic_abstractivity_4",
            "wiki_cat_sum_animal_challenge_test_topic_abstractivity_5",
            "wiki_cat_sum_animal_challenge_test_topic_abstractivity_6",
            "wiki_cat_sum_animal_challenge_test_topic_diversity_3",
            "wiki_cat_sum_animal_challenge_test_topic_diversity_4",
            "wiki_cat_sum_animal_challenge_test_topic_diversity_5",
            "wiki_cat_sum_animal_challenge_test_topic_diversity_6",
            "wiki_cat_sum_animal_test",
            "wiki_cat_sum_animal_validation",
        ]
    },
    "wiki_cat_sum_company": {
        "language":
            "en",
        "task":
            "summarization",
        "challenge_sets": [
            "wiki_cat_sum_company_challenge_test_abstractivity_3",
            "wiki_cat_sum_company_challenge_test_abstractivity_4",
            "wiki_cat_sum_company_challenge_test_abstractivity_5",
            "wiki_cat_sum_company_challenge_test_abstractivity_2",
            "wiki_cat_sum_company_challenge_test_topic_diversity_3",
            "wiki_cat_sum_company_challenge_test_topic_diversity_4",
            "wiki_cat_sum_company_challenge_test_topic_diversity_5",
            "wiki_cat_sum_company_challenge_test_topic_diversity_2",
            "wiki_cat_sum_company_test",
            "wiki_cat_sum_company_validation",
        ]
    },
    "wiki_cat_sum_film": {
        "language":
            "en",
        "task":
            "summarization",
        "challenge_sets": [
            "wiki_cat_sum_film_challenge_test_abstractivity_3",
            "wiki_cat_sum_film_challenge_test_abstractivity_4",
            "wiki_cat_sum_film_challenge_test_abstractivity_5",
            "wiki_cat_sum_film_challenge_test_abstractivity_2",
            "wiki_cat_sum_film_challenge_test_diversity_3",
            "wiki_cat_sum_film_challenge_test_diversity_4",
            "wiki_cat_sum_film_challenge_test_diversity_5",
            "wiki_cat_sum_film_challenge_test_diversity_2",
            "wiki_cat_sum_film_test",
            "wiki_cat_sum_film_validation",
        ]
    },
}

_SUPPORTED_DATASETS = {}
for _, task_values in _SUPPORTED_TASKS.items():
  for challenge_set_name in task_values["challenge_sets"]:
    _SUPPORTED_DATASETS[challenge_set_name] = {
        "language": task_values["language"],
        "task": task_values["task"]
    }

# Also add "*_validation" compatibility.
# DO NOT MODIFY THIS PART.
_VAL_COMPATIBILITY_DICT = {}
for key, value in _SUPPORTED_DATASETS.items():
    if key.endswith("_val"):
        _VAL_COMPATIBILITY_DICT[key.replace("_val", "_validation")] = value
_SUPPORTED_DATASETS.update(_VAL_COMPATIBILITY_DICT)

# Now automatically add download links.
for dataset_name, settings in _SUPPORTED_DATASETS.items():
    # For both val and validation named datasets, the download should link to validation.
    if dataset_name.endswith("_val"):
        sanitized_dataset_name = dataset_name.replace("_val", "_validation")
    else:
        sanitized_dataset_name = dataset_name
    settings['url'] = f"https://github.com/GEM-benchmark/GEM-metrics/releases/download/data/{sanitized_dataset_name}.json"

# If you want to add a custom dataset / url, you can add it here.
# Just ensure that your entry has `language`, `task`, and `url` set.

# HERE

# Access functions used by the main scripts.
def get_all_datasets():
    return list(_SUPPORTED_DATASETS.keys())

def get_language_for_dataset(dataset_name):
    data_config = _SUPPORTED_DATASETS.get(dataset_name, {'language': 'en'})
    return data_config['language']

def get_task_type_for_dataset(dataset_name):
    data_config = _SUPPORTED_DATASETS.get(dataset_name, {'task':'text2text'})
    return data_config["task"]

def get_url_for_dataset(dataset_name):
    data_config = _SUPPORTED_DATASETS.get(dataset_name, {'url': ""})
    return data_config["url"]


'''
EVALUATION SUITE SETTINGS

We support two types of challenge sets:
1) Transformations
2) Subpopulations

Transformed datasets will be evaluated in relation to the parent_datapoints from
which they are derived. This feature is added here - simply add the name of the
challenge set and the parent set.

Subpopulations are partitions of test set of particular interest. If you have a
file for subpopulations, add them to the list below.
'''


# URLs to download standard references from
_TRANSFORMATION_PARENT_DATASETS = {
    "cs_restaurants_challenge_test_scramble": "cs_restaurants_test",
    "web_nlg_ru_challenge_test_scramble": "web_nlg_ru_test",
    "schema_guided_dialog_challenge_test_backtranslation": "schema_guided_dialog_test",
    "schema_guided_dialog_challenge_test_bfp02": "schema_guided_dialog_test",
    "schema_guided_dialog_challenge_test_bfp05": "schema_guided_dialog_test",
    "schema_guided_dialog_challenge_test_nopunc": "schema_guided_dialog_test",
    "schema_guided_dialog_challenge_test_scramble": "schema_guided_dialog_test",
    "xsum_challenge_test_backtranslation": "xsum_test",
    "xsum_challenge_test_bfp_02": "xsum_test",
    "xsum_challenge_test_bfp_05": "xsum_test",
    "xsum_challenge_test_nopunc": "xsum_test",
    "e2e_nlg_challenge_test_scramble": "e2e_nlg_test",
    "web_nlg_en_challenge_test_scramble": "web_nlg_en_test",
    "web_nlg_en_challenge_test_numbers": "web_nlg_en_test",
    "wiki_auto_asset_turk_challenge_test_asset_backtranslation": "wiki_auto_asset_turk_test_asset",
    "wiki_auto_asset_turk_challenge_test_asset_bfp02": "wiki_auto_asset_turk_test_asset",
    "wiki_auto_asset_turk_challenge_test_asset_bfp05": "wiki_auto_asset_turk_test_asset",
    "wiki_auto_asset_turk_challenge_test_asset_nopunc": "wiki_auto_asset_turk_test_asset",
    "wiki_auto_asset_turk_challenge_test_turk_backtranslation": "wiki_auto_asset_turk_test_turk",
    "wiki_auto_asset_turk_challenge_test_turk_bfp02": "wiki_auto_asset_turk_test_turk",
    "wiki_auto_asset_turk_challenge_test_turk_bfp05": "wiki_auto_asset_turk_test_turk",
    "wiki_auto_asset_turk_challenge_test_turk_nopunc": "wiki_auto_asset_turk_test_turk",
}

def get_all_transformation_sets():
    return list(_TRANSFORMATION_PARENT_DATASETS.keys())

def get_parent_dataset_for_transformation(dataset_name):
    return _TRANSFORMATION_PARENT_DATASETS.get(dataset_name, None)

_SUBPOPULATION_BASE = [
    "cs_restaurants_test",
    "e2e_nlg_test",
    "schema_guided_dialog_test",
    "totto_test",
    "xsum_test",
    "web_nlg_en_test",
    "web_nlg_ru_test",
    "wiki_auto_asset_turk_test_asset",
    "wiki_auto_asset_turk_test_turk",
]

_SUBPOPULATION_DATASETS = {
    dataset_name: f"https://github.com/GEM-benchmark/GEM-metrics/releases/download/data/{dataset_name}_contrast_sets.json"
    for dataset_name in _SUBPOPULATION_BASE
}

def get_all_subpopulation_sets():
    return list(_SUBPOPULATION_DATASETS.keys())

def get_url_for_subpopulation(dataset_name):
    return _SUBPOPULATION_DATASETS.get(dataset_name, None)


