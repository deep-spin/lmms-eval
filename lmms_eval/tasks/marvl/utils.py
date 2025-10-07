from lmms_eval.utils import eval_logger

def marvl_doc_to_visual(doc):
    return [doc["image"].convert("RGB")]

def marvl_doc_to_text(doc, model_specific_prompt_kwargs):
    conversations = doc['conversations']
    query = conversations[0]['value'].replace('<image>\n', '').strip() + '. Answer with the option directly.'
    return query
    
def marvl_doc_to_target(doc):
    conversations = doc['conversations']
    answer = str(conversations[1])
    if 'true' in answer: return 'true'
    elif 'false' in answer: return 'false'
    else: raise Exception(f"get target failed for id {doc['id']} - conversations: {conversations}")



def check_output(doc,output):
    accepted_answers = {
    "(b)": "true",
    "(a)": "false",
    "false": "false",
    "true": "true",
    "false.": "false",
    "true.": "true",
    "b (b)": "true",
    "a (a)": "false",
    "b": "true",
    "a": "false",
    "b (true)": "true",
    "a (false)": "false",
    "b.": "true",
    "a.": "false",
    "a false": "false",
    "b true": "true",
    "a:": "false",
    "b:": "true",
    "(b) true": "true",
    "(a) false": "false",
    "yes":"true",
    "no":"false",
    "yes.":"true",
    "no.":"false",
    }
    out = output.strip().lower()
    if out in accepted_answers.keys():
        return accepted_answers[out]
    else:
        eval_logger.warning("Model's output not in accepted answers. Doc: {doc}. Falling back to error...")
        import ipdb;ipdb.set_trace()
        return None


def marvl_process_result(doc, results):
    target = marvl_doc_to_target(doc)
    pred = results[0]
    if target.strip().lower() not in ["true" , "false"]:
        eval_logger.warning("Target '{target}' needs post-processing. Task: Marvl")
    out = check_output(doc,pred)
    if target.strip().lower()==out:
        return {"exact_match": 1.0}
    else:
        return {"exact_match": 0.0}
