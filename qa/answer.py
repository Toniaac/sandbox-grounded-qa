# Copyright (c) 2022 Cohere Inc. and its affiliates.
#
# Licensed under the MIT License (the "License");
# you may not use this file except in compliance with the License.
#
# You may obtain a copy of the License in the LICENSE file at the top
# level of this repository.

import numpy as np
import pandas as pd
from qa.model import get_sample_answer
from qa.search import embedding_search, get_results_paragraphs_from_paper
from qa.util import pretty_print


def trim_stop_sequences(s, stop_sequences):
    """Remove stop sequences found at the end of returned generated text."""

    for stop_sequence in stop_sequences:
        if s.endswith(stop_sequence):
            return s[:-len(stop_sequence)]
    return s


def answer(question, context, co, model, chat_history=""):
    """Answer a question given some context."""
    #print(model)
    if 'command' in model:
        prompt = (
            f'read the paragraphs below and answer the question in detail, with bullet points if necessary. If the question cannot be answered based on the context alone, write "sorry i had trouble answering this question, based on the information i found\n'
            f"\n"
            f"Context:\n"
            f"{ context }\n"
            f"\n"
            f"Question: { question }\n"
            "Answer:")
        if chat_history:
            prompt = (
                f'read the context and chat history below and answer the question in detail, with bullet points if necessary. If the question cannot be answered based on the context alone, write "sorry i had trouble answering this question, based on the information i found\n'
                f"\n"
                f"Context:\n"
                f"{ context }\n"
                f"\n"
                f"Chat History:\n"
                f"{ chat_history }\n"
                f"\n"
                f"Question: { question }\n"
                "Answer:")
        stop_sequences = []

    else:
        prompt = ("This is an example of question answering based on a text passage:\n "
                  f"Context:-{context}\nQuestion:\n-{question}\nAnswer:\n-")
        if chat_history:
            prompt = ("This is an example of factual question answering chat bot. It "
                      "takes the text context and answers related questions:\n "
                      f"Context:-{context}\nChat Log\n{chat_history}\nbot:")
        stop_sequences = ["\n"]

    num_generations = 4
    prompt = "".join(co.tokenize(text=prompt).token_strings[-1700:])
    prediction = co.generate(model=model,
                             prompt=prompt,
                             max_tokens=300,
                             temperature=0.5,
                             stop_sequences=stop_sequences,
                             num_generations=num_generations,
                             return_likelihoods="GENERATION")
    generations = [[
        trim_stop_sequences(prediction.generations[i].text.strip(), stop_sequences),
        prediction.generations[i].likelihood
    ] for i in range(num_generations)]
    generations = list(filter(lambda x: not x[0].isspace(), generations))
    response = generations[np.argmax([g[1] for g in generations])][0]
    print("Actual Answer: \n", response.strip())
    return response.strip()


def answer_with_search(question,
                       co,
                       serp_api_token,
                       chat_history="",
                       model='command-xlarge-nightly',
                       embedding_model="multilingual-22-12",
                       url=None,
                       n_paragraphs=1,
                       verbosity=0):
    """Generates completion based on search results."""

    paragraphs, paragraph_sources = get_results_paragraphs_multi_process(question, serp_api_token, url=url)
    if not paragraphs:
        return ("", "", "")
    sample_answer = get_sample_answer(question, co)

    results = embedding_search(paragraphs, paragraph_sources, sample_answer, co, model=embedding_model)

    if verbosity > 1:
        pprint_results = "\n".join([r[0] for r in results])
        pretty_print("OKGREEN", f"all search result context: {pprint_results}")

    results = results[-n_paragraphs:]
    context = "\n".join([r[0] for r in results])

    if verbosity:
        pretty_print("OKCYAN", "relevant result context: " + context)

    response = answer(question, context, co, chat_history=chat_history, model=model)

    return (response, [r[1] for r in results], [r[0] for r in results])


def answer_with_paper(question, 
                        paper_pii,
                        co,
                        chat_history="",
                        model='xlarge',
                        embedding_model="large",
                        n_paragraphs=1,
                        verbosity=0):
    """Generates completion based on search results."""

    paragraphs = get_results_paragraphs_from_paper(paper_pii)
    if not paragraphs:
        return ("", "", "")
    sample_answer = get_sample_answer(question, paper_pii, co)
    #print('Sample answer: ', sample_answer)
    results = embedding_search(paragraphs, sample_answer, co, model=embedding_model)
    
    if verbosity > 1:
        pprint_results = "\n".join([r[0] for r in results])
        pretty_print("OKGREEN", f"all search result context: {pprint_results}")

    results = results[-n_paragraphs:]
    #print('Results: ', results)
    #df = pd.read_csv(paper_pii+'.csv')
    #context = "The title of the paper is: " + df['titles'][0] + "\n" + "The abstract of the paper is: " + df['paragraphs'][0] + "\n"
    context = "\n".join([r[0] for r in results])
    #print("Context: ", context)
    if verbosity:
        pretty_print("OKCYAN", "relevant result context: " + context)

    response = answer(question, context, co, model=model)

    return (response, paper_pii, [r[0] for r in results])
