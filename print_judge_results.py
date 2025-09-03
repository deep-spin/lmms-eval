import json

LANGS=["de","eng","es","it","fr"]
MODELS = ["Aya","Pixtral"]
path_dict  = {"Aya" : "/mnt/scratch-artemis/manos/data/tower-vision-eval-outputs/tower-vision/aya-vision-bench-gen-final-results/aya-vision-bench-gen/v6/CohereForAI__aya-vision-8b/20250610_093522_CohereForAI__aya-vision-8b_vs_utter-project__EuroVLM-9B-Preview_with_litellm_proxy__neulab__claude-3-7-sonnet-20250219/",
"Pixtral": "/mnt/scratch-artemis/manos/data/tower-vision-eval-outputs/tower-vision/aya-vision-bench-gen-final-results/aya-vision-bench-gen/v6/CohereForAI__aya-vision-8b/20250610_093522_CohereForAI__aya-vision-8b_vs_utter-project__EuroVLM-9B-Preview_with_litellm_proxy__neulab__claude-3-7-sonnet-20250219/"}

for model in MODELS:
    print(f"***{model} vs EuroVLM(baseline)***")

    for lang in LANGS:
        print(f"Language: {lang}")
        with open(path_dict[model] + f"{lang}/judge_results.json", "r") as f:
            results = json.load(f)
        print(results)