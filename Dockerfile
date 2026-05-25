FROM alpine:latest AS builder

RUN apk add --no-cache gcc musl-dev zlib-dev curl wget

WORKDIR /app
COPY src ./src

RUN wget -q https://github.com/zanfranceschi/rinha-de-backend-2026/raw/main/resources/references.json.gz -O references.json.gz

RUN gcc -O3 -mavx2 src/kmeans_preprocess.c -o preprocess -lz -lm -fopenmp

# Pre-compute K-Means and generate dataset.bin during image build phase
RUN ./preprocess

RUN gcc -O3 -mavx2 src/server.c -o server -lm

FROM alpine:latest

RUN apk add --no-cache libgcc

WORKDIR /app

COPY --from=builder /app/dataset.bin ./dataset.bin
COPY --from=builder /app/server ./server

EXPOSE 8080

CMD ["./server"]
