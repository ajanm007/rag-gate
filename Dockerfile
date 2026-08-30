# Multi-stage build: compile the release binary in a full Rust image, ship
# only the binary in a slim runtime image.
FROM rust:1.90-slim AS builder

# reqwest's TLS backend (openssl-sys) needs pkg-config + OpenSSL dev headers
# to compile against the system OpenSSL.
RUN apt-get update && apt-get install -y --no-install-recommends \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY Cargo.toml Cargo.lock ./
COPY src ./src

RUN cargo build --release

FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home appuser
COPY --from=builder /app/target/release/rag-gate /usr/local/bin/rag-gate
USER appuser

# Every knob is a RAGGATE_* env var (see README) — these are the defaults,
# spelled out so `docker run -e RAGGATE_UPSTREAM_URL=...` is self-documenting.
ENV RAGGATE_LISTEN_ADDR=0.0.0.0:8080 \
    RAGGATE_UPSTREAM_URL=https://api.openai.com

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD curl -sf http://127.0.0.1:8080/healthz || exit 1

CMD ["rag-gate"]
