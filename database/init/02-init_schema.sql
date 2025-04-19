DROP TABLE IF EXISTS product_images;
DROP TABLE IF EXISTS products;
DROP TABLE IF EXISTS brands;

CREATE TABLE brands (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    name_normalized TEXT NOT NULL,
    platform TEXT NOT NULL,
    url TEXT,
    description TEXT,
    created_at TIMESTAMP DEFAULT now(),
    updated_at TIMESTAMP DEFAULT now(),
    UNIQUE(name, platform)
);

CREATE TABLE products (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    brand TEXT,
    brand_normalized TEXT,
    product_name_normalized TEXT NOT NULL,
    category TEXT,
    url TEXT,
    description_detail TEXT,
    description_semantic TEXT,
    original_price INTEGER,
    discounted_price INTEGER,
    sold_out BOOLEAN,
    thumbnail_url TEXT,
    created_at TIMESTAMP DEFAULT now(),
    updated_at TIMESTAMP DEFAULT now(),
    UNIQUE(name, brand)
);

CREATE TABLE product_images (
    id SERIAL PRIMARY KEY,
    product_id INTEGER REFERENCES products(id) ON DELETE CASCADE,
    url TEXT NOT NULL,
    is_thumbnail BOOLEAN DEFAULT FALSE,
    order_index INTEGER,
    clothing_only BOOLEAN,
    created_at TIMESTAMP DEFAULT now()
);
