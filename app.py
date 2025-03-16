import streamlit as st
import pandas as pd
import csv
from embedder import  Embedding_Model
from gemini_utils import translate
import os 

from torch.nn.functional import cosine_similarity
from torch import argmax as argmax
from torch import load as load

def fetch_products(query_embedding):
    cos_sim = cosine_similarity(
        image_embeddings,
        query_embedding.unsqueeze(0).expand_as(image_embeddings),
        dim=1
    )
    best_idx = argmax(cos_sim).item()

    return best_idx


@st.cache_data
def prepare():
    embedder = Embedding_Model()
    '''
    Just for Demo
    '''
    with open('./items_with_detail_updated.csv', 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        item_list = list(reader)
    image_embeddings= load('image_embeddings.pt')
    return embedder, item_list,image_embeddings

embedder, item_list,image_embeddings = prepare()



# Streamlit UI 구성
st.title("🛍️ 제품 추천 웹앱")

# 사용자 입력 받기
query = st.text_input("검색할 제품 키워드를 입력하세요: ", "")

# 검색 버튼
if st.button("검색"):
    if query:
        eng_query = translate(query)
        # 제품 정보 가져오기
        print(eng_query)
        query_embedding=embedder.embed_text(eng_query)
        best_idx = fetch_products(query_embedding)
        product = item_list[best_idx]
        if product:
            st.write("### 검색 결과")
            
            # 이미지와 텍스트 영역 비율 설정
            col1, col2 = st.columns([1, 3])
            
            with col1:
                st.image(product['Image_paths'], width=150)  # 이미지 출력
                
            with col2:
                st.write(f"**[{product['Name']}]({product['Link']})**")
                st.write(f"💰 가격: {product['Price']}")
                st.markdown(f"[🔗 구매 링크]({product['Link']})")
        else:
            st.warning("검색 결과가 없습니다.")
    else:
        st.warning("키워드를 입력해주세요!")