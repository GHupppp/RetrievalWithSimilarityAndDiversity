from vec_ret_sim_div import cos_similarity_list
from sklearn.metrics.pairwise import cosine_similarity
import json
import pandas as pd
import numpy as np

def MMR(embedding, vectordb, query, k, lambda_mult=0.5):
    margi_search = vectordb.max_marginal_relevance_search(query, k, fetch_k=3 * k, lambda_mult=lambda_mult)
    texts_list = [docu.page_content for docu in margi_search]
    embed_texts = embedding.embed_documents(texts_list)
    embed_query = embedding.embed_query(query)
    cos_sim_list = cos_similarity_list(embed_query, embed_texts)
    return (texts_list, cos_sim_list)

def test_main2(file_name, embedding, vectordb, query_list, k, lambda_mult=0.5):
    query_sim_results_list = []
    query_text_results_list = []
    for query in query_list:
        (mmr, cosSim) = MMR(embedding, vectordb, query, k, lambda_mult)
        query_results = [query] + mmr
        query_text_results_list.append(query_results)
        mmr_text_sum = '\n'.join(mmr)
        embed_mmr_text_sum = embedding.embed_documents([mmr_text_sum])
        embed_query = embedding.embed_query(query)
        cos_query_mmr_text_sum = cosine_similarity([embed_query], embed_mmr_text_sum)
        cosSim.append(cos_query_mmr_text_sum[0][0])
        query_sim_results_list.append(cosSim)
    col_name = ['query'] + ['query_result' + str(i) for i in range(1, k + 1)]
    df = pd.DataFrame(columns=col_name, data=query_text_results_list)
    df.to_csv(file_name + 'text.csv', encoding='utf-8', index=False)
    col_name = ['Vector' + str(i) for i in range(1, k + 1)] + ['Avg_cos_sim', 'Pairwise_cos_sim', 'Sum_vector', 'Mmr_text_sum']
    df = pd.DataFrame(columns=col_name, data=query_sim_results_list)
    df.to_csv(file_name + '.csv', encoding='utf-8', index=False)