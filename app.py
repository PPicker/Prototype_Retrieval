import os, faiss, psycopg2, streamlit as st, torch
from dotenv import load_dotenv
from boto3 import client as boto3_client
from embedder import Embedder
from gemini_utils.categorize import categorize
from gemini_utils.translate import translate
from utils.aws import get_s3_client
from typing import List, Dict
from pgvector.psycopg2 import register_vector

# ───────────────────────────────────────────────────────────────
# 1) 앱 최상단: 세션 스테이트 초기화 (한 번만 실행)
# ───────────────────────────────────────────────────────────────
if "page" not in st.session_state:
    st.session_state["page"] = "search"  # "search" or "detail"
    st.session_state["product"] = None
    st.session_state["query"] = ""
    st.session_state["results"] = []


# ── 설정 ─────────────────────────────────────────────────────────────
load_dotenv(".env.prod")
TOP_K = 3


@st.cache_resource
def load_resources():
    embedder = Embedder()
    conn = psycopg2.connect(
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT", "5432"),
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        options="-c search_path=public",
    )
    conn.autocommit = True
    s3 = get_s3_client()
    bucket = os.getenv("AWS_S3_BUCKET_NAME")
    return embedder, conn, s3, bucket


embedder, conn, s3, bucket = load_resources()
register_vector(conn)


def search_and_fetch(q_emb: torch.Tensor, k: int = TOP_K) -> List[Dict]:
    """
    1) pgvector의 inner-product (<#>)로 top-k id 검색
    2) 곧바로 name, price, link, thumbnail_key 가져오기
    3) presigned URL 생성
    """
    # 1) Tensor → Python 리스트
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT
                p.id,
                p.name,
                p.original_price AS price,
                p.url        AS link,
                p.thumbnail_key
            FROM products AS p
            ORDER BY p.embedding <#> %s
            LIMIT %s;
            """,
            (q_emb, k),
        )
        cols = [c.name for c in cur.description]
        rows = [dict(zip(cols, r)) for r in cur.fetchall()]

    # 2) presigned URL 생성
    for r in rows:
        key = r.pop("thumbnail_key")
        if key:
            r["image_url"] = s3.generate_presigned_url(
                "get_object",
                Params={"Bucket": bucket, "Key": key},
                ExpiresIn=3600,
            )
        else:
            r["image_url"] = None

    return rows


# ── 상세용 DB 조회 ───────────────────────────────────────────────────
def get_product_detail(prod_id: int) -> dict | None:
    # 1) products 메타
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT name,
                   original_price AS price,
                   url as link,
                   thumbnail_key
              FROM products
             WHERE id = %s
            """,
            (prod_id,),
        )
        meta = cur.fetchone()
        if not meta:
            return None
        name, price, link, thumb_key = meta

    # 2) products_images 에서 모든 key (order_index 순)
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT key
              FROM product_images
             WHERE product_id = %s
             ORDER BY order_index
            """,
            (prod_id,),
        )
        img_keys = [row[0] for row in cur.fetchall()]
        print(img_keys)

    return {
        "name": name,
        "price": price,
        "link": link,
        "thumbnail_key": thumb_key,
        "image_keys": img_keys,
    }


# ───────────────────────────────────────────────────────────────
# 5) 콜백: 페이지 전환
# ───────────────────────────────────────────────────────────────
def do_search():
    """검색 수행 후 결과 세션에 저장"""
    q = st.session_state["query"].strip()
    if not q:
        st.warning("검색어를 입력해주세요")
        return
    # eng = translate(q)
    # category = categorize(q)

    q_emb = embedder.embed(q)
    st.session_state["results"] = search_and_fetch(q_emb, TOP_K)


def go_to_detail(prod_id: int):
    st.session_state["page"] = "detail"
    st.session_state["product"] = prod_id


def go_to_search():
    st.session_state["page"] = "search"


# ───────────────────────────────────────────────────────────────
# 6) 페이지 렌더링
# ───────────────────────────────────────────────────────────────
def page_search():
    st.title("🛍️ 벡터 검색 상품 추천")
    st.text_input("검색어를 입력하세요", key="query")
    st.button("🔍 검색", on_click=do_search)

    results = st.session_state.get("results", [])
    if not results:
        return

    for p in results:
        c1, c2 = st.columns([1, 3])
        with c1:
            st.image(p["image_url"], width=120)
            st.button(
                "자세히 보기 ▶",
                key=f"detail_{p['id']}",
                on_click=go_to_detail,
                args=(p["id"],),
            )
        with c2:
            st.markdown(f"**{p['name']}**")
            st.write(f"💰 {int(p['price']):,} 원")
            st.markdown(f"[구매 링크 ▶]({p['link']})")
        st.markdown("---")


def page_detail():
    prod_id = st.session_state["product"]
    data = get_product_detail(prod_id)
    if not data:
        st.error("상품을 찾을 수 없습니다.")
        st.button("⬅ 돌아가기", on_click=go_to_search)
        return

    # presigned URL 리스트 생성
    urls = []
    if data["thumbnail_key"]:
        urls.append(
            s3.generate_presigned_url(
                "get_object",
                Params={"Bucket": bucket, "Key": data["thumbnail_key"]},
                ExpiresIn=3600,
            )
        )
    for k in data["image_keys"]:
        urls.append(
            s3.generate_presigned_url(
                "get_object", Params={"Bucket": bucket, "Key": k}, ExpiresIn=3600
            )
        )

    # ── 1) 페이지 제목 & 상품 정보 ──
    st.title("상품 상세 보기")
    st.markdown(f"## {data['name']}")
    st.write(f"###  {int(data['price']):,} 원")
    st.markdown(f"[ 구매하러 가기 ▶]({data['link']})")
    st.markdown("---")

    # ── 2) 이미지들 (정보 아래로 배치) ──
    for url in urls:
        st.image(url, use_container_width=True)
        st.markdown("---")

    # ── 3) 뒤로가기 버튼 ──
    st.button("⬅ 검색 결과로 돌아가기", on_click=go_to_search)


# ───────────────────────────────────────────────────────────────
# 7) 라우터: 페이지 결정
# ───────────────────────────────────────────────────────────────
if st.session_state["page"] == "search":
    page_search()
else:
    page_detail()
