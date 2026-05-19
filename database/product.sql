CREATE TABLE IF NOT EXISTS products (
    id          SERIAL PRIMARY KEY,
    제품이름    VARCHAR(255) NOT NULL,
    자산번호    VARCHAR(255) NOT NULL,
    필요인원수  INTEGER      NOT NULL DEFAULT 1,
    created_at  TIMESTAMP    DEFAULT NOW(),
    updated_at  TIMESTAMP    DEFAULT NOW()
);
