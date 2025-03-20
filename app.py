import streamlit as st
import pandas as pd
import csv
from embedder import  Embedding_Model
from gemini_utils import translate
import os 

from torch import matmul as matmul
from torch import argsort as argsort
from torch import load as load
from torch import Tensor as torch_Tensor


def fetch_ranking(query_embedding, image_embeddings,k=3):
    """
    주어진 query_embedding에 대해, 이미지 임베딩들과의 dot product 계산하고 (normalized 되어있음)
    유사도 기준 내림차순으로 정렬된 인덱스 리스트를 반환합니다.
    -> top - K 반환
    """

    dot_sim = matmul(image_embeddings, query_embedding.unsqueeze(1)).squeeze(1)
    ranking = argsort(dot_sim, descending=True)
    return ranking[:k]

@st.cache_data
def prepare():
    embedder = Embedding_Model()
    '''
    Just for Demo
    '''
    with open('./items.csv', 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        item_list = list(reader)
    image_embeddings= load('image_embeddings.pt')
    return embedder, item_list,image_embeddings

embedder, item_list,image_embeddings = prepare()



# Streamlit UI 구성
st.title("🛍️ 제품 추천 웹앱")

# 사용자 입력 받기
query = st.text_input("검색할 제품 키워드를 입력하세요: ", "")

K = 3
# 검색 버튼
if st.button("검색"):
    if query:
        eng_query = translate(query)
        # 제품 정보 가져오기
        print(eng_query)
        query_embedding = embedder.embed_text(eng_query)
        rankings_K = fetch_ranking(query_embedding, image_embeddings, K)  # 상위 K개의 인덱스 리스트 반환
        
        if isinstance(rankings_K, torch_Tensor):
            st.write("### 검색 결과")
            # ranking 결과에 있는 각 상품을 출력
            for rank, idx in enumerate(rankings_K, start=1):
                product = item_list[idx]
                st.markdown(f"#### 결과 {rank}")
                # 이미지와 텍스트 영역 비율 설정
                col1, col2 = st.columns([1, 3])
                with col1:
                    st.image(product['Image_path'], width=150)  # 이미지 출력
                with col2:
                    st.write(f"**[{product['Name']}]({product['Link']})**")
                    st.write(f"💰 가격: {product['Price']}")
                    st.markdown(f"[🔗 구매 링크]({product['Link']})")
                st.markdown("---")  # 각 결과 사이 구분선
        else:
            st.warning("검색 결과가 없습니다.")
    else:
        st.warning("키워드를 입력해주세요!")