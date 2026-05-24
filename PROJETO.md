# inemaupsk — Servidor/Worker de Upscaling

Projeto separado do `inemaimg`. Foco único: **upscaling de imagens em batch**, mantendo fidelidade ao original (sem alucinação generativa).

---

## Motivação

- Carga prevista: **~11 milhões de fotos** pra processar.
- Em 1 GPU, mesmo otimista (2s/foto), são ~255 dias rodando 24/7.
- Se rodasse junto com `inemaimg`, o lock de GPU travaria a geração realtime (consumida pelo `timesmkt3`) por meses.
- Solução: serviço isolado, container próprio, lock próprio, ciclo de vida próprio.

## Decisão arquitetural

- **Projeto separado** em `~/projetos/inemaupsk/`.
- Mesma stack do `inemaimg` (Docker aarch64/CUDA13/Blackwell sm_120, NGC PyTorch arm64-sbsa, FastAPI, padrão de loaders) — copiar o padrão, não importar como dependência.
- **Spandrel** (https://github.com/chaiNNer-org/spandrel) como motor de SR — carrega pesos `.pth` de qualquer arquitetura (Real-ESRGAN, ESRGAN, SwinIR, HAT, DAT, etc).
- Pesos baixados do **OpenModelDB** (https://openmodeldb.info) pra `models/upscale/`.
- **Descartar Upscayl** como dependência — é Electron + binário Vulkan x86_64, incompatível com aarch64/CUDA. Útil só como referência de catálogo de modelos.

## Caso de uso

> "Mandar foto via qualquer agente e devolver em alta resolução mantendo todas consistência."

- Input: imagem arbitrária (foto realista principalmente).
- Output: imagem ampliada (2x ou 4x) sem inventar conteúdo novo.
- Consumidor inicial: pipeline batch sobre as 11M fotos.
- Consumidor futuro: qualquer agente (n8n, timesmkt3, etc) via HTTP.

## MVP

- **1 modelo só** no v1: `4x-UltraSharp` (foto realista, default).
- Endpoint `POST /upscale`:
  ```
  {
    "image": "<base64 PNG/JPG>",
    "scale": 4
  }
  ```
- Sem `model` exposto no v1 (adiciona quando tiver 2º modelo).
- Sem `tile_size` exposto — calcula internamente pela VRAM livre.
- Limite hard: input ≤4096px no lado maior, recusa com 413 acima.
- PNG na saída pra não acumular compressão JPEG.

## Modo batch (essencial pro caso real)

API HTTP é pra ad-hoc/integração. Pra 11M fotos, precisa de **worker batch** que:
- Lê fila/diretório de input.
- Processa N em paralelo (limitado por VRAM).
- Salva em diretório/storage de output.
- Tem pause/resume e checkpoint (reprocessar 11M do zero se quebrar = inviável).
- Idempotente: não reprocessa o que já existe.
- Métricas: throughput (fotos/min), ETA, falhas.

Decisões pendentes (ver "Perguntas em aberto" abaixo).

## Impacto operacional

- **GPU**: mesma do `inemaimg` (DGX Spark Blackwell sm_120) compartilhada via runtime nvidia. Sem MPS configurado → containers competem por tempo. Solução: rodar batch em **janela** (noite/fim de semana) com pause-on-demand, ou em GPU separada.
- **VRAM**: SR é leve (~2-4 GB ativo, 50-300 MB de pesos). Convive bem.
- **Disco**: 11M × ~16x tamanho (4x linear) = aumento massivo de storage. Definir antes onde guardar saída.
- **Latência típica esperada**: 1024→4096 em 2-4s, 2048→4096 em 1-2s.

## Perguntas em aberto (responder antes de codar)

1. **Prazo**: precisa terminar em semanas, meses ou "quando der"?
2. **Origem das 11M fotos**: S3? Disco local? Banco? URL list? (afeta I/O — pode virar gargalo antes da GPU).
3. **Tamanho médio das fotos**: 500×500? 2K? 4K? Muda latência em 10x.
4. **Tem outra GPU disponível** ou só o DGX Spark?
5. **Saída**: substitui original ou guarda os dois? Onde salva?
6. **Reprocessamento**: tudo de uma vez ou contínuo (fotos novas chegando)?
7. **Janela de execução**: 24/7 ou só horário ocioso do DGX Spark?
8. **Foto tem tipo único** (todas fotos realistas) ou tem mistura (foto + arte + texto)? Se misturado, vai precisar de mais de 1 modelo.

## Próximos passos (ordem sugerida)

1. Responder as 8 perguntas acima.
2. Decidir batch design: fila (Redis/SQS) vs varredura de diretório vs DB.
3. Setup do projeto: `Dockerfile` (copiar de `inemaimg`), `requirements.txt` com `spandrel`, `compose.yml`.
4. Baixar `4x-UltraSharp` pra `models/upscale/`.
5. Implementar `loader.py` (1 modelo, infer com tiling), `worker.py` (batch loop), `server.py` (HTTP opcional pra ad-hoc).
6. Smoke test: 1 foto 1024×1024 → 4096×4096.
7. Benchmark real: 100 fotos representativas pra medir throughput.
8. Calcular ETA real com base no benchmark.
9. Definir storage de saída.
10. Subir batch piloto (10k fotos) antes do full run.

## Referências

- Doc original de análise no `inemaimg`: `~/projetos/inemaimg/UPSCAYL.md`
- Padrão de loaders/Docker a copiar: `~/projetos/inemaimg/`
- Spandrel: https://github.com/chaiNNer-org/spandrel
- OpenModelDB: https://openmodeldb.info
- 4x-UltraSharp: procurar no OpenModelDB
