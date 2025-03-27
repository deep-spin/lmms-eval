import numpy as np
from loguru import logger as eval_logger
from lmms_eval.tasks.multi30k.comet_utils.comet import RefCOMET

def process_docs(docs):
    # docs = docs.select(range(10))
    # docs = docs.filter(lambda x: x["is_bad_source"] != "true")
    return docs

def process_docs_reverse(docs):
    # docs = docs.select(range(10))
    # docs = docs.filter(lambda x: x["is_bad_source"] != "true")
    # Swap source and target
    docs = docs.map(lambda x: {
        **x,  # Keep all other fields
        'source': x['target'],
        'target': x['source']
    })
    return docs

def doc_to_text(doc, lmms_eval_specific_kwargs):
    source_txt = doc["source"]
    pre_prompt = lmms_eval_specific_kwargs["pre_prompt"]
    pre_prompt = pre_prompt.format(source=source_txt)
    return f"{pre_prompt}"

def doc_to_visual(doc):
    return []

def process_results(doc, results):
    pred_answer = results[0]
    source = doc["source"]
    target = doc["target"]
    return {"results":{"prediction":pred_answer,"ground_truth": target, "source":source}}

def aggregate_results(results):
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

