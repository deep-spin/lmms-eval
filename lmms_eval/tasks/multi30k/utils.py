import numpy as np
from loguru import logger as eval_logger
from lmms_eval.tasks.multi30k.comet_utils.comet import RefCOMET

# def multi30k_process_docs(docs):
#     return docs

def multi30k_doc_to_text(doc, lmms_eval_specific_kwargs):
    source_txt = doc["en"]
    pre_prompt = lmms_eval_specific_kwargs["pre_prompt"]
    pre_prompt = pre_prompt.format(source=source_txt)
    return f"{pre_prompt}"


def multi30k_doc_to_visual(doc):
    return [doc["image"].convert("RGB")]


def multi30k_process_results_fr(doc, results):
    pred_answer = results[0]
    source = doc["en"]
    target = doc["fr"]
    return {"results":{"prediction":pred_answer,"ground_truth": target, "source":source}}

def multi30k_process_results_de(doc, results):
    pred_answer = results[0]
    source = doc["en"]
    target = doc["de"]
    return {"results":{"prediction":pred_answer,"ground_truth": target, "source":source}}

def multi30k_process_results_cs(doc, results):
    pred_answer = results[0]
    source = doc["en"]
    target = doc["cs"]
    return {"results":{"prediction":pred_answer,"ground_truth": target, "source":source}}

def multi30k_aggregate_results(results):
    comet = RefCOMET(model="Unbabel/XCOMET-XL")
    sources = [res["source"] for res in results]
    hypotheses = [res["prediction"] for res in results]
    references = [res["ground_truth"] for res in results]
    comet.make_samples(sources, hypotheses, references)
    segments_scores_correct = comet.evaluate(hypotheses, references, sources, gpus=1, batch_size=16).result["segments_scores"]
    results = {
               f"avg_XCOMET-XL_score": np.mean(segments_scores_correct),
               }
    return results

