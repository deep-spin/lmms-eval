COMPARATIVE_GEN_SYSTEM_PROMPT = """\
Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user prompt displayed below. You will be given assistant A's answer and assistant B's answer. Your job is to evaluate which assistant's answer is better, by taking into account the given image.

Begin your evaluation by generating your own answer to the prompt. You must provide your answers before judging any answers. To provide your answer, take into account the given image.

When evaluating the assistants' answers, compare both assistants' answers with your answer. You must identify and correct any mistakes or inaccurate information.

Then consider if the assistant's answers are helpful, relevant, and concise. Helpful means the answer correctly responds to the prompt or follows the instructions. Note when user prompt has any ambiguity or more than one interpretation, it is more helpful and appropriate to ask for clarifications or more information from the user than providing an answer based on assumptions. Relevant means all parts of the response closely connect or are appropriate to what is being asked. Concise means the response is clear and not verbose or excessive.

Then consider the creativity and novelty of the assistant's answers when needed. Finally, identify any missing important information in the assistants' answers that would be beneficial to include when responding to the user prompt.

After providing your explanation, you must output only one of the following choices as your final verdict with a label:

1. Assistant A is significantly better: [[A>>B]]
2. Assistant A is slightly better: [[A>B]]
3. Tie, relatively the same: [[A=B]]
4. Assistant B is slightly better: [[B>A]]
5. Assistant B is significantly better: [[B>>A]]

Example output: "My final verdict is tie: [[A=B]]".\
"""

COMPARATIVE_GEN_USER_PROMPT = """Question:{question}\n\n<|The Start of Assistant A's Answer|>\n{answer_1}\n<|The End of Assistant A's Answer|>\n\n<|The Start of Assistant B's Answer|>\n{answer_2}\n<|The End of Assistant B's Answer|>"""

COMPARATIVE_SYS_PROMPT_NO_GEN = """Please act as an impartial judge and evaluate the quality of the responses (Response (A) and Response (B)) based on the provided instruction and the image content."""

COMPARATIVE_USER_PROMPT_NO_GEN = """\
Which of the following responses better addresses the given instruction in {language}?

The response should be primarily in {language}.
The evaluation should prioritize accuracy and correctness.
If both responses are incorrect or contain inaccurate information, treat them as a Tie.
After assessing accuracy and correctness, consider other factors like helpfulness, relevance, depth, creativity, and level of detail.
Do not let the length or order of the responses influence your judgment.
Ensure your evaluation is objective and free from position bias.

Begin your evaluation by comparing the two responses and providing a brief explanation of your decision.

After your comparison, select one of the following choices as your final decision:

1. Response (A) is significantly better: [[A≫B]]

2. Response (A) is slightly better: [[A>B]]

3. Tie, Response (A) and Response (B) are relatively the same: [[A=B]]

4. Response (B) is slightly better: [[B>A]]

5. Response (B) is significantly better: [[B≫A]]

Instruction: {question}
Response (A): {completion_a}
Response (B): {completion_b}

Your response must strictly follow this format:
Explanation: <concise comparison and explanation in English>
Final Decision: < [[B≫A]], [[B>A]], [[A≫B]], [[A>B]], [[A=B]] >
"""

DIRECT_ASSESSMENT_SYSTEM_PROMPT = """\
You are an impartial judge and will be given an image, a question, and an answer of an AI assistant. Your job is to evaluate the quality of the answer.

Begin your evaluation by generating your own answer to the prompt. You must provide your answers before judging any answers.

When evaluating the assistant's answer, compare the answer with your answer. You must identify and correct any mistakes or inaccurate information.

Then consider if the assistant's answer is helpful, relevant, and concise. Helpful means the answer correctly responds to the prompt or follows the instructions. Note when user prompt has any ambiguity or more than one interpretation, it is more helpful and appropriate to ask for clarifications or more information from the user than providing an answer based on assumptions. Relevant means all parts of the response closely connect or are appropriate to what is being asked. Concise means the response is clear and not verbose or excessive.

After providing your explanation, you must output only a score between 0 and 100, where 0 is the worst and 100 is the best.

Example output: "My final score is 85."
"""

DIRECT_ASSESSMENT_USER_PROMPT = """Question:{question}\n\n<|The Start of Assistant Answer|>\n{answer}\n<|The End of Assistant Answer|>\n\n"""