

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


_SUPPORTED_DATASETS = {
    "common_gen_val": {"language": "en", "task": "data2text"},
    "common_gen_test": {"language": "en", "task": "data2text"},
    "common_gen_challenge_test_scramble": {"language": "en", "task": "data2text"},
    "common_gen_challenge_train_sample": {"language": "en", "task": "data2text"},
    "common_gen_challenge_validation_sample": {"language": "en", "task": "data2text"},
    "cs_restaurants_val": {"language": "cs", "task": "data2text"},
    "cs_restaurants_test": {"language": "cs", "task": "data2text"},
    "cs_restaurants_challenge_test_scramble": {"language": "cs", "task": "data2text"},
    "cs_restaurants_challenge_train_sample": {"language": "cs", "task": "data2text"},
    "cs_restaurants_challenge_validation_sample": {"language": "cs", "task": "data2text"},
    # "dart_val": {"language": "en", "task": "data2text"},
    # "dart_test": {"language": "en", "task": "data2text"},
    "e2e_nlg_val": {"language": "en", "task": "data2text"},
    "e2e_nlg_test": {"language": "en", "task": "data2text"},
    "e2e_nlg_challenge_test_scramble": {"language": "en", "task": "data2text"},
    "e2e_nlg_challenge_train_sample": {"language": "en", "task": "data2text"},
    "e2e_nlg_challenge_validation_sample": {"language": "en", "task": "data2text"},
    "mlsum_de_val": {"language": "de", "task": "summarization"},
    "mlsum_de_test": {"language": "de", "task": "summarization"},
    "mlsum_de_challenge_test_covid": {"language": "de", "task": "summarization"},
    "mlsum_de_challenge_train_sample": {"language": "de", "task": "summarization"},
    "mlsum_de_challenge_validation_sample": {"language": "de", "task": "summarization"},
    "mlsum_es_val": {"language": "es", "task": "summarization"},
    "mlsum_es_test": {"language": "es", "task": "summarization"},
    "mlsum_es_challenge_test_covid": {"language": "es", "task": "summarization"},
    "mlsum_es_challenge_train_sample": {"language": "es", "task": "summarization"},
    "mlsum_es_challenge_validation_sample": {"language": "es", "task": "summarization"},
    "schema_guided_dialog_val": {"language": "en", "task": "data2text"},
    "schema_guided_dialog_test": {"language": "en", "task": "data2text"},
    "schema_guided_dialog_challenge_test_backtranslation": {"language": "en", "task": "data2text"},
    "schema_guided_dialog_challenge_test_bfp02": {"language": "en", "task": "data2text"},
    "schema_guided_dialog_challenge_test_bfp05": {"language": "en", "task": "data2text"},
    "schema_guided_dialog_challenge_test_nopunc": {"language": "en", "task": "data2text"},
    "schema_guided_dialog_challenge_test_scramble": {"language": "en", "task": "data2text"},
    "schema_guided_dialog_challenge_train_sample": {"language": "en", "task": "data2text"},
    "schema_guided_dialog_challenge_validation_sample": {"language": "en", "task": "data2text"},
    "totto_val": {"language": "en", "task": "data2text"},
    "totto_test": {"language": "en", "task": "data2text"},
    "totto_challenge_test_scramble": {"language": "en", "task": "data2text"},
    "totto_challenge_train_sample": {"language": "en", "task": "data2text"},
    "totto_challenge_validation_sample": {"language": "en", "task": "data2text"},
    "web_nlg_en_val": {"language": "en", "task": "data2text"},
    "web_nlg_en_test": {"language": "en", "task": "data2text"},
    "web_nlg_en_challenge_test_scramble": {"language": "en", "task": "data2text"},
    "web_nlg_en_challenge_test_numbers": {"language": "en", "task": "data2text"},
    "web_nlg_en_challenge_train_sample": {"language": "en", "task": "data2text"},
    "web_nlg_en_challenge_validation_sample": {"language": "en", "task": "data2text"},
    "web_nlg_ru_val": {"language": "ru", "task": "data2text"},
    "web_nlg_ru_test": {"language": "ru", "task": "data2text"},
    "web_nlg_ru_challenge_test_scramble": {"language": "ru", "task": "data2text"},
    "web_nlg_ru_challenge_train_sample": {"language": "ru", "task": "data2text"},
    "web_nlg_ru_challenge_validation_sample": {"language": "ru", "task": "data2text"},
    # "wiki_auto_asset_turk_val": {"language": "en", "task": "text_simplification"},
    "wiki_auto_asset_turk_test_asset": {"language": "en", "task": "text_simplification"},
    "wiki_auto_asset_turk_test_turk": {"language": "en", "task": "text_simplification"},
    "wiki_auto_asset_turk_challenge_test_asset_backtranslation": {"language": "en", "task": "text_simplification"},
    "wiki_auto_asset_turk_challenge_test_asset_bfp02": {"language": "en", "task": "text_simplification"},
    "wiki_auto_asset_turk_challenge_test_asset_bfp05": {"language": "en", "task": "text_simplification"},
    "wiki_auto_asset_turk_challenge_test_asset_nopunc": {"language": "en", "task": "text_simplification"},
    "wiki_auto_asset_turk_challenge_test_turk_backtranslation": {"language": "en", "task": "text_simplification"},
    "wiki_auto_asset_turk_challenge_test_turk_bfp02": {"language": "en", "task": "text_simplification"},
    "wiki_auto_asset_turk_challenge_test_turk_bfp05": {"language": "en", "task": "text_simplification"},
    "wiki_auto_asset_turk_challenge_test_turk_nopunc": {"language": "en", "task": "text_simplification"},
    "wiki_auto_asset_turk_challenge_train_sample": {"language": "en", "task": "text_simplification"},
    "wiki_auto_asset_turk_challenge_validation_sample": {"language": "en", "task": "text_simplification"},
    "wiki_lingua_spanish_es_val": {"language": "en", "task": "summarization"},
    "wiki_lingua_spanish_es_test": {"language": "en", "task": "summarization"},
    "wiki_lingua_russian_ru_val": {"language": "en", "task": "summarization"},
    "wiki_lingua_russian_ru_test": {"language": "en", "task": "summarization"},
    "wiki_lingua_turkish_tr_val": {"language": "en", "task": "summarization"},
    "wiki_lingua_turkish_tr_test": {"language": "en", "task": "summarization"},
    "wiki_lingua_vietnamese_vi_val": {"language": "en", "task": "summarization"},
    "wiki_lingua_vietnamese_vi_test": {"language": "en", "task": "summarization"},
    "xsum_val": {"language": "en", "task": "summarization"},
    "xsum_test": {"language": "en", "task": "summarization"},
    "xsum_challenge_test_backtranslation": {"language": "en", "task": "summarization"},
    "xsum_challenge_test_bfp_02": {"language": "en", "task": "summarization"},
    "xsum_challenge_test_bfp_05": {"language": "en", "task": "summarization"},
    "xsum_challenge_test_nopunc": {"language": "en", "task": "summarization"},
    "xsum_challenge_test_covid": {"language": "en", "task": "summarization"},
    "xsum_challenge_train_sample": {"language": "en", "task": "summarization"},
    "xsum_challenge_validation_sample": {"language": "en", "task": "summarization"},
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
    # The Hugging Face Hub has a limit of 2,000 files per folder, so we store each reference in a separate folder.
    settings['url'] = f"https://huggingface.co/datasets/GEM/references/resolve/main/{sanitized_dataset_name}/{sanitized_dataset_name}.json"

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
 

