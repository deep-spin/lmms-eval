
import base64
from PIL import Image
from io import BytesIO
import numpy as np

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
    # Process images in place
    # docs = docs.select(range(20)) # filter out some samples!
    return docs

def commute_doc_to_visual(doc):
    image = doc["image"].convert('RGB')
    return [image]


def commute_doc_to_text(doc,lmms_eval_specific_kwargs=None ):
    source_txt = doc["source"]
    pre_prompt = lmms_eval_specific_kwargs["pre_prompt"]
    pre_prompt = pre_prompt.format(source=source_txt)
    return f"{pre_prompt}"


def commute_process_results(doc, results):
    pred = results[0]
    target = doc["correct_translation"]
    incorrect_translation = doc["incorrect_translation"]
    source = doc["source"]
    return {"results": {"image_name": doc["image_name"], "source": source, "prediction": pred, "ground_truth": target, "incorrect_translation": incorrect_translation} }

def compute_pairwise_accuracy(segments_scores_correct, segments_scores_incorrect):
    pairwise_scores = []
    for score_correct, score_incorrect in zip(segments_scores_correct, segments_scores_incorrect):
        if score_correct > score_incorrect:
            pairwise_scores.append(1)
        else:
            pairwise_scores.append(0)
    pairwise_accuracy = sum(pairwise_scores) / len(pairwise_scores)
    return pairwise_accuracy

def commute_aggregate_results(results):
    comet = RefCOMET(model="Unbabel/XCOMET-XL")
    sources = [res["source"] for res in results]
    hypotheses = [res["prediction"] for res in results]
    references = [res["ground_truth"] for res in results]
    incorrect_translations = [res["incorrect_translation"] for res in results]
    comet.make_samples(sources, hypotheses, references)
    segments_scores_correct = comet.evaluate(hypotheses, references, sources, gpus=1, batch_size=16).result["segments_scores"]
    segments_scores_incorrect = comet.evaluate(hypotheses, incorrect_translations, sources, gpus=1, batch_size=16).result["segments_scores"]
    pairwise_accuracy = compute_pairwise_accuracy(segments_scores_correct, segments_scores_incorrect)
    results = {"pairwise_accuracy": pairwise_accuracy, 
               "avg_correct_score": np.mean(segments_scores_correct), 
               "avg_incorrect_score": np.mean(segments_scores_incorrect),
            #    "segments_scores_correct": segments_scores_correct,
            #    "segments_scores_incorrect": segments_scores_incorrect
               }
    return results
