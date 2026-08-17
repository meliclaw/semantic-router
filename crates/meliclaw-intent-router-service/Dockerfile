# Multi-stage: build tools stay out of the published image (guía Meliclaw §3.2).
FROM rust:1.82-bookworm AS build
WORKDIR /app
COPY Cargo.toml Cargo.lock ./
COPY crates ./crates
RUN cargo build --release -p meliclaw-intent-router-service

FROM debian:bookworm-slim
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*
COPY --from=build /app/target/release/meliclaw-intent-router-service /usr/local/bin/meliclaw-intent-router-service
EXPOSE 8091
USER nobody
ENV MELICLAW_INTENT_BIND=0.0.0.0:8091
ENTRYPOINT ["/usr/local/bin/meliclaw-intent-router-service"]
