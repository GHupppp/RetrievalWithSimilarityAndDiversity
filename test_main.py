from create_vector_db import embedding_corpus
from create_vector_db import read_OpenBookQA
from create_vector_db import read_ARC
from create_vector_db import read_BoolQ
from create_vector_db import read_Puzzle
from create_vector_db import read_STS
from create_vector_db import read_SciQ
from sim_div_retriever_txt import test_main1_txt
from sim_div_retriever import test_main1
from mmr_retriever import test_main2
from langchain_community.embeddings import HuggingFaceEmbeddings
from datetime import datetime

def main():
    HuggingFace_embedding = HuggingFaceEmbeddings()
    embedding_model = HuggingFace_embedding
    all_data_list = read_SciQ()
    print(len(all_data_list))
    (vectordb, query_list) = embedding_corpus(all_data_list, embedding_model, 0.2)
    print('Vector database is OK')
    k = 15
    test_main1('SDR', embedding_model, vectordb, query_list, batch=3 * k, k=k)
    print('SDR.csv and SDRtext.csv is OK')
    test_main2('MMR20', embedding_model, vectordb, query_list, k, lambda_mult=0.2)
    print('MMR20.csv and MMR20text.csv is OK')
    test_main2('MMR30', embedding_model, vectordb, query_list, k, lambda_mult=0.3)
    print('MMR30.csv and MMR30text.csv is OK')
    test_main2('MMR40', embedding_model, vectordb, query_list, k, lambda_mult=0.4)
    print('MMR40.csv and MMR40text.csv is OK')
    test_main2('MMR50', embedding_model, vectordb, query_list, k, lambda_mult=0.5)
    print('MMR50.csv and MMR50text.csv is OK')
    test_main2('MMR60', embedding_model, vectordb, query_list, k, lambda_mult=0.6)
    print('MMR60.csv and MMR60text.csv is OK')
    test_main2('MMR70', embedding_model, vectordb, query_list, k, lambda_mult=0.7)
    print('MMR70.csv and MMR70text.csv is OK')
    test_main2('MMR80', embedding_model, vectordb, query_list, k, lambda_mult=0.8)
    print('MMR80.csv and MMR80text.csv is OK')
    test_main2('MMR90', embedding_model, vectordb, query_list, k, lambda_mult=0.9)
    print('MMR90.csv and MMR90text.csv is OK')
print(datetime.now())
main()
print(datetime.now())