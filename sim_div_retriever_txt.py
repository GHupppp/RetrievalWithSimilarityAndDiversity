from vec_ret_sim_div import VecRetSimDivTxt
from vec_ret_sim_div import cos_similarity_list
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd

def SimDivRetrieverTxt(embedding, vectordb, query, batch=20, k=4):
    simi_search = vectordb.similarity_search(query, batch)
    texts_list = [docu.page_content for docu in simi_search]
    res = VecRetSimDivTxt(texts_list, query, embedding, k)
    embed_res = embedding.embed_documents(res)
    embed_query = embedding.embed_query(query)
    cos_sim_list = cos_similarity_list(embed_query, embed_res)
    return (res, cos_sim_list)

def test_main1_txt(file_name, embedding, vectordb, query_list, batch=20, k=4):
    list_of_list = []
    for query in query_list:
        (vsd, cosSim) = SimDivRetrieverTxt(embedding, vectordb, query, batch, k)
        cosSim_csv = [cs[0][0] for cs in cosSim]
        vsd_text_sum = '\n'.join(vsd)
        embed_vsd_text_sum = embedding.embed_documents([vsd_text_sum])
        embed_query = embedding.embed_query(query)
        cos_query_vsd_text_sum = cosine_similarity(np.array(embed_query).reshape(1, len(embed_query)), np.array(embed_vsd_text_sum[0]).reshape(1, len(embed_query)))
        cosSim_csv.append(cos_query_vsd_text_sum[0][0])
        list_of_list.append(cosSim_csv)
    col_name = ['Vector1', 'Vector2', 'Vector3', 'Vector4', 'Sum_vector', 'Vsd_text_sum']
    df = pd.DataFrame(columns=col_name, data=list_of_list)
    df.to_csv(file_name + '.csv', encoding='utf-8', index=False)