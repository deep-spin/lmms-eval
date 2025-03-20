
import base64
from PIL import Image
from io import BytesIO
import numpy as np
import copy

from loguru import logger

from lmms_eval.tasks.commute.comet_utils.comet import RefCOMET


# def base64_to_bytes(base64_string):
#     # Remove the header if it exists (e.g., "data:image/jpeg;base64,")
#     if "base64," in base64_string:
#         base64_string = base64_string.split("base64,")[1]
#     # Decode base64 string to bytes
#     img_bytes = base64.b64decode(base64_string)
#     return img_bytes

def commute_process_docs(docs):

    def add_output_to_docs(example):
        return {
            'output': example['correct_translation']
        }
    # Process images in place
    # docs = docs.select(range(20)) # filter out some samples!
    
    samples = docs.to_list()
    samples_duplicated = copy.deepcopy(samples)
    new_samples = []
    for sample in samples:
        sample["output"] = sample["correct_translation"]
        new_samples.append(sample)
    for sample in samples_duplicated:
        sample["output"] = sample["incorrect_translation"]
        new_samples.append(sample)
    docs = docs.from_list(new_samples)
    return docs

def commute_doc_to_visual(doc):
    image = Image.open(BytesIO(doc['image']['bytes']))
    image = image.convert('RGB')
    return [image]



def commute_doc_to_text(doc,lmms_eval_specific_kwargs=None ):
    source_txt = doc["source"]
    pre_prompt = lmms_eval_specific_kwargs["pre_prompt"]
    pre_prompt = pre_prompt.format(source=source_txt)
    input = pre_prompt+doc["output"]
    return f"{input}"


def commute_process_results(doc, results):
    loglikelihood = results[0][0]
    target = doc["correct_translation"]
    source = doc["source"]
    return {"results": {"image_name": doc["image_name"], "source": source, "loglikelihood": loglikelihood, } }

def commute_aggregate_results(results):
    # comet = RefCOMET(model="Unbabel/XCOMET-XL")
    # sources = [res["source"] for res in results]
    # hypotheses = [res["prediction"] for res in results]
    # references = [res["ground_truth"] for res in results]
    # incorrect_translations = [res["incorrect_translation"] for res in results]
    # comet.make_samples(sources, hypotheses, references)
    # segments_scores_correct = comet.evaluate(hypotheses, references, sources, gpus=1, batch_size=16).result["segments_scores"]
    # segments_scores_incorrect = comet.evaluate(hypotheses, incorrect_translations, sources, gpus=1, batch_size=16).result["segments_scores"]
    # pairwise_accuracy = compute_pairwise_accuracy(segments_scores_correct, segments_scores_incorrect)
    
    # Split results into two halves
    half_idx = len(results) // 2
    results_correct = results[:half_idx]
    results_incorrect = results[half_idx:]

    # Extract loglikelihoods from each half
    loglikelihood_correct = [res["loglikelihood"] for res in results_correct]
    loglikelihood_incorrect = [res["loglikelihood"] for res in results_incorrect]

    accuracy = [1 for l_correct, l_incorrect in zip(loglikelihood_correct, loglikelihood_incorrect) if l_correct<l_incorrect ]
    contrastive_accuracy = sum(accuracy)/len(loglikelihood_correct)
    results = {"contrastive_accuracy": contrastive_accuracy}
    return results
