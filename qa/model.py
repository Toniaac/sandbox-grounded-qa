# Copyright (c) 2022 Cohere Inc. and its affiliates.
#
# Licensed under the MIT License (the "License");
# you may not use this file except in compliance with the License.
#
# You may obtain a copy of the License in the LICENSE file at the top
# level of this repository.

import os

import cohere
import numpy as np
from cohere.classify import Example

from qa.util import pretty_print
import pandas as pd
_DATA_DIRNAME = os.path.join(os.path.dirname(__file__), "prompt_data")


def get_contextual_search_query(history, co, model="xlarge", verbosity=0):
    """Adds message history context to user query."""

    prompt_path = os.path.join(_DATA_DIRNAME, "get_contextual_search_query.prompt")
    #prompt = "Rephrase the question so that it fits the context of the conversation. If the question is descriptive enough, just keep it as is:\n"
    #prompt += f"{history}\n-"
    #prompt += "Rephrased question: "
    #print("contextual question prompt: ", prompt)
    
    with open(prompt_path) as f:
        prompt = f.read() + f"{history}\n-"
    prediction = co.generate(
        model=model,
        prompt=prompt,
        max_tokens=100,
        temperature=0.75,
        k=0,
        p=0.75,
        frequency_penalty=0,
        presence_penalty=0,
        stop_sequences=["\n"],
        return_likelihoods="GENERATION",
        num_generations=4,
    )
    likelihood = [g.likelihood for g in prediction.generations]
    result = prediction.generations[np.argmax(likelihood)].text

    print("Rephrased question: ", result.strip())

    if verbosity:
        pretty_print("OKGREEN", "contextual question prompt: " + prompt)
        pretty_print("OKCYAN", "contextual question: " + result)
    return result.strip()


def get_sample_answer(question, paper_pii, co, model="command-xlarge-nightly"):
    """Return a sample answer to a question based on the model's training data."""

    #prompt_path = os.path.join(_DATA_DIRNAME, "get_sample_answer.prompt")
    df = pd.read_csv(paper_pii+'.csv')
    context = "Title: " + df['titles'][0] + "\nAbstract: " + df['paragraphs'][0]
    prompt = "Answer the question based on the context. If you cannot find an answer based on the context, keep all the key words and try to answer it without context:\n" + context + "\nQuestion: " + question + "\nAnswer:"
    #with open(prompt_path) as f:
    #    prompt = f.read() + f"{question}\nAnswer:"
    response = co.generate(model=model,
                           prompt=prompt,
                           max_tokens=200,
                           temperature=0.5,
                           k=0,
                           p=0.7,
                           stop_sequences=["--"])
    print("sample_answer: ", response.generations[0].text)
    return response.generations[0].text
