import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def cos_similarity_list(query_vector, vector_list):
    k = len(vector_list)
    cos_sim_list = list(cosine_similarity([query_vector], vector_list)[0])
    avg_cos_sim = np.mean(cos_sim_list)
    cos_sim_list.append(avg_cos_sim)
    cos_sims = cosine_similarity(vector_list, vector_list)
    pairwise_cos_sim = (np.sum(cos_sims) - k) / (k * (k - 1))
    cos_sim_list.append(pairwise_cos_sim)
    sum_v = np.sum(np.array(vector_list), axis=0)
    cos_query_sum = cosine_similarity([query_vector], [sum_v])
    cos_sim_list.append(cos_query_sum[0][0])
    return cos_sim_list

def VecRetSimDiv(vectors, query, k):
    query_dim = len(query)
    results = []
    results_index = []
    max_index = 0
    for i in range(k):
        results.append(vectors[max_index])
        results_index.append(max_index)
        sum_results = np.sum(results, axis=0)
        max_simi = -1
        for j in range(len(vectors)):
            if j not in results_index:
                v = np.add(sum_results, vectors[j])
                cos_simi = cosine_similarity([v], [query])
                if cos_simi > max_simi:
                    max_simi = cos_simi
                    max_index = j
            else:
                pass
    return (results, results_index)

def VecRetSimDivTxt(txt_list, query, embedding_model, k=4):
    embed_query = embedding_model.embed_query(query)
    query_dim = len(embed_query)
    results = []
    max_index = 0
    for i in range(k):
        results.append(txt_list[max_index])
        del txt_list[max_index]
        sum_results = '\n'.join(results)
        max_simi = 0
        for j in range(len(txt_list)):
            tmp_sum = sum_results + '\n' + txt_list[j]
            embed_tmp_sum = embedding_model.embed_documents([tmp_sum])
            cos_simi = cosine_similarity([embed_tmp_sum], [embed_query])
            if cos_simi > max_simi:
                max_simi = cos_simi
                max_index = j
    return results
'\nL = [[5, 2, 6], [1, 2, 3], [3, 2, 6], [1, 5, 6], [3, 8, 1], [2, 26, 6]]\nq = [1, 1, 1]\nprint([cosine_similarity(np.array(q).reshape(1, 3), np.array(v).reshape(1, 3)) for v in L])\n\nres, res_index = VecRetSimDiv(L, q, k=3)\nsum_v = sum_vectors(res)\nprint(res, sum_v)\nprint(res_index)\ncosine_similarity(np.array(sum_v).reshape(1, 3), np.array(q).reshape(1, 3))\n'