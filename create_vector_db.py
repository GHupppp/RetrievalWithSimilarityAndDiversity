import pandas as pd
import random
from datasets import load_dataset

def read_SciQ():
    ds = load_dataset('allenai/sciq')
    questions = ds['train']['question']
    answers = ds['train']['correct_answer']
    supports = ds['train']['support']
    all_data_list = [f'{q}. {a}. {s}' for (q, a, s) in zip(questions, answers, supports)]
    print('all_data_list------', all_data_list[0:2])
    return all_data_list[0:3000]

def read_STS():
    dataset = load_dataset('mteb/stsbenchmark-sts')
    sentence1_list = dataset['train']['sentence1']
    sentence2_list = dataset['train']['sentence2']
    all_data_list = sentence1_list + sentence2_list
    print('all_data_list------', all_data_list[0:2])
    return all_data_list

def read_BoolQ():
    train_data = pd.read_json('BoolQtrain.jsonl', lines=True)
    dev_data = pd.read_json('BoolQdev.jsonl', lines=True)
    all_data = pd.concat([train_data, dev_data])
    all_data_list = []
    for (index, row) in all_data.iterrows():
        all_data_list.append(row['question'] + '. ' + row['title'] + '. ' + row['passage'] + '.')
    print('all_data_list------', all_data_list[0:2])
    return all_data_list

def read_ARC():
    arc_train = pd.read_json('ARCtrain.jsonl', lines=True)
    arc_test = pd.read_json('ARCtest.jsonl', lines=True)
    arc_dev = pd.read_json('ARCdev.jsonl', lines=True)
    all_data = pd.concat([arc_train, arc_test, arc_dev])
    all_data_list = []
    for (index, row) in all_data.iterrows():
        all_data_list.append(row['question'] + '. ' + '. '.join(row['answers']))
    print('all_data_list------', all_data_list[0:2])
    return all_data_list

def read_OpenBookQA():
    OpenBookQA_train = pd.read_json('OpenBookQAtrain.jsonl', lines=True)
    OpenBookQA_test = pd.read_json('OpenBookQAtest.jsonl', lines=True)
    OpenBookQA_dev = pd.read_json('OpenBookQAdev.jsonl', lines=True)
    all_data = pd.concat([OpenBookQA_train, OpenBookQA_test, OpenBookQA_dev])
    all_data_list = []
    for (index, row) in all_data.iterrows():
        stem = row['question']['stem']
        choices = row['question']['choices']
        choices_to_str = [choi['text'] for choi in choices]
        all_data_list.append(stem + '. ' + '. '.join(choices_to_str))
    print('all_data_list------', all_data_list[0:2])
    return all_data_list

def read_Puzzle():
    all_data = pd.read_json('puzzle.jsonl', lines=True)
    all_data_list = []
    for (index, row) in all_data.iterrows():
        all_data_list.append(row['messages'][0]['content'])
    print('all_data_list------', all_data_list[0:2])
    return all_data_list
from langchain_community.vectorstores import FAISS

def embedding_corpus(all_data_list, embedding_model, sample_ratio=0.2):
    print('total item number:', len(all_data_list))
    sample_size = int(sample_ratio * len(all_data_list))
    start_loc = 0
    query_list = all_data_list[start_loc:start_loc + sample_size]
    del all_data_list[start_loc:start_loc + sample_size]
    print('query item number:', len(query_list), 'db item number:', len(all_data_list))
    print('query_list------', query_list[0], '\n')
    vectordb = FAISS.from_texts(texts=all_data_list, embedding=embedding_model)
    print('vectordb------', vectordb)
    return (vectordb, query_list)

def OLD_embedding_corpus(all_data_list, embedding_model, sample_size=100):
    query_list = random.sample(all_data_list, sample_size)
    for ql in query_list:
        all_data_list.remove(ql)
    query_list = [qm[:qm.find('.') + 1] for qm in query_list]
    print('query_list------', query_list[0], '\n')
    vectordb = FAISS.from_texts(texts=all_data_list, embedding=embedding_model)
    print('vectordb------', vectordb)
    return (vectordb, query_list)