import json
import pandas as pd
from vec_ret_sim_div import VecRetSimDiv
from vec_ret_sim_div import cos_similarity_list
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def SimDivRetriever(embedding, vectordb, query, batch, k):
    simi_search = vectordb.similarity_search(query, batch)
    texts_list = [docu.page_content for docu in simi_search]
    embed_texts = embedding.embed_documents(texts_list)
    embed_query = embedding.embed_query(query)
    (res, res_index) = VecRetSimDiv(embed_texts, embed_query, k)
    cos_sim_list = cos_similarity_list(embed_query, res)
    vsd = []
    for i in res_index:
        vsd.append(texts_list[i])
    return (vsd, cos_sim_list)

def test_main1(file_name, embedding, vectordb, query_list, batch, k):
    query_sim_results_list = []
    query_text_results_list = []
    for query in query_list:
        (vsd, cosSim) = SimDivRetriever(embedding, vectordb, query, batch, k)
        query_results = [query] + vsd
        query_text_results_list.append(query_results)
        vsd_text_sum = '\n'.join(vsd)
        embed_vsd_text_sum = embedding.embed_documents([vsd_text_sum])
        embed_query = embedding.embed_query(query)
        cos_query_vsd_text_sum = cosine_similarity([embed_query], embed_vsd_text_sum)
        cosSim.append(cos_query_vsd_text_sum[0][0])
        query_sim_results_list.append(cosSim)
    col_name = ['query'] + ['query_result' + str(i) for i in range(1, k + 1)]
    df = pd.DataFrame(columns=col_name, data=query_text_results_list)
    df.to_csv(file_name + 'text.csv', encoding='utf-8', index=False)
    col_name = ['Vector' + str(i) for i in range(1, k + 1)] + ['Avg_cos_sim', 'Pairwise_cos_sim', 'Sum_vector', 'Vsd_text_sum']
    df = pd.DataFrame(columns=col_name, data=query_sim_results_list)
    df.to_csv(file_name + '.csv', encoding='utf-8', index=False)