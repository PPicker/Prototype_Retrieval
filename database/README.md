# PPicker Database Setup

이 레포지토리는 PPicker 프로젝트의 PostgreSQL 기반 데이터베이스 설정을 위한 구성 파일을 담고 있습니다.  
Docker를 이용해 로컬 또는 원격 서버에서 손쉽게 실행 가능한 개발/운영 환경을 제공합니다.

---

## 📦 테이블 구조

### 1. `brands`

| 컬럼명             | 타입     | 설명                       |
|------------------|----------|----------------------------|
| `id`             | SERIAL   | Primary Key                |
| `name`           | TEXT     | 브랜드 원래 이름            |
| `name_normalized`| TEXT     | 정규화된 브랜드 이름 (검색용) |
| `platform`       | TEXT     | 수집된 플랫폼 명 (예: musinsa) |
| `url`            | TEXT     | 브랜드 페이지 링크         |
| `description`    | TEXT     | 브랜드 설명                 |
| `created_at`     | TIMESTAMP| 생성일                     |
| `updated_at`     | TIMESTAMP| 수정일                     |

---

### 2. `products`

| 컬럼명               | 타입     | 설명                     |
|--------------------|----------|--------------------------|
| `id`               | SERIAL   | Primary Key              |
| `name`             | TEXT     | 상품 전체 이름           |
| `brand`            | TEXT     | 브랜드 이름              |
| `brand_normalized` | TEXT     | 정규화된 브랜드 이름     |
| `product_name_normalized` | TEXT | 정규화된 상품명         |
| `category`         | TEXT     | 카테고리                 |
| `url`              | TEXT     | 상품 상세 페이지 URL     |
| `description_detail` | TEXT   | 상세 설명 (html 기반)     |
| `description_semantic` | TEXT | 의도 기반 설명           |
| `original_price`   | INTEGER  | 정가                     |
| `discounted_price` | INTEGER  | 할인 가격                |
| `sold_out`         | BOOLEAN  | 품절 여부                |
| `thumbnail_url`    | TEXT     | 대표 이미지 URL          |
| `created_at`       | TIMESTAMP| 생성일                   |
| `updated_at`       | TIMESTAMP| 수정일                   |

---

### 3. `product_images`

| 컬럼명          | 타입     | 설명                          |
|---------------|----------|-------------------------------|
| `id`          | SERIAL   | Primary Key                   |
| `product_id`  | INTEGER  | 외래 키 (products 테이블 참조) |
| `url`         | TEXT     | 이미지 URL (예: S3 링크)       |
| `is_thumbnail`| BOOLEAN  | 대표 이미지 여부              |
| `order_index` | INTEGER  | 이미지 순서                   |
| `clothing_only`| BOOLEAN | 옷만 있는 이미지인지 여부      |
| `created_at`  | TIMESTAMP| 생성일                        |

---

## 🐳 Docker 환경 구성

### `docker-compose.yml`

PostgreSQL 15 버전을 사용하며, 로컬 디렉토리를 DB 볼륨과 초기화 스크립트로 마운트합니다.

```yaml
version: '3.8'

services:
  postgres:
    image: postgres:15
    container_name: ppicker_postgres
    restart: always
    env_file:
      - .env
    ports:
      - "${POSTGRES_PORT}:5432"
    volumes:
      - ./pgdata:/var/lib/postgresql/data
      - ./init:/docker-entrypoint-initdb.d

volumes:
  pgdata:
```
> 실제 환경 변수는 `.env` 파일에 정의하며, 이 파일은 `.gitignore`에 포함되어야 합니다.


---

## 🚀 실행 방법

```bash
# 1. 처음 실행 또는 환경 변수 변경 시 (초기화 포함)
docker compose down -v
rm -rf ./pgdata
docker compose up -d

# 2. 이후 일반 실행
docker compose up -d
```
> init/ 폴더 내의 .sql, .sh 파일들이 컨테이너 최초 실행 시 자동으로 실행됩니다.


## 🛠️ 초기화 스크립트 구성

### `init/init_schema.sql`
- 테이블 정의 스크립트 (brands, products, product_images)

### `init/create_user.sh`
- 환경 변수 기반 사용자 계정 생성
- `chmod +x`로 실행 권한이 있어야 정상 실행됩니다

```bash
chmod +x init/create_user.sh
```
## 🔐 보안 주의사항

- `.env` 파일에는 비밀번호 및 유저 정보가 포함되어 있으므로 **절대 git에 커밋하지 마세요.**
- `.gitignore`에 다음 내용을 포함해야 합니다:

```gitignore
.env
pgdata/
```


- 운영 환경에서는 일반 유저 계정 (`APP_USER`) 을 통해서만 DB에 접근하도록 구성하는 것이 안전합니다.

---

## 📁 디렉토리 구조

```bash
database/
│
├── docker-compose.yml
├── .env               # (git에 포함 X)
├── .gitignore
├── init/
│   ├── init_schema.sql
│   └── create_user.sh
│── pgdata/            # DB 데이터 볼륨 (git에 포함 X)