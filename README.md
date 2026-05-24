# inemaupsk

Servidor de **upscaling de imagens** (super-resolution) baseado em
[Spandrel](https://github.com/chaiNNer-org/spandrel), com 4 modelos ESRGAN-family
prontos, API HTTP estilo `inemaimg`, e uma UI web pra testar / comparar lado a lado.

> Projeto irmão de `~/projetos/inemaimg/`. Separado pra não travar a GPU
> compartilhada durante o batch de **~11M fotos** (ver `PROJETO.md`).

---

## Stack

- **NGC PyTorch 26.03** arm64-sbsa (Blackwell sm_120, CUDA 13) — mesma base do `inemaimg`
- **FastAPI + uvicorn**
- **Spandrel** carrega qualquer `.pth` ESRGAN-family (RRDBNet, SwinIR, HAT, DAT…)
- **Pesos do OpenModelDB** (script de download incluso)
- **Docker Compose** isolado, lock próprio, porta `8002`

## Modelos inclusos (v0.2)

| modelo | melhor pra | notas |
|---|---|---|
| `4x-UltraSharp` | foto realista (default) | agressivo em detalhe, pode "plasticar" pele |
| `RealESRGAN_x4plus` | foto JPEG-degradada | baseline robusto, treinado com simulação de degradação |
| `4x_NMKD-Siax_200k` | foto comprimida | bom equilíbrio detalhe / artefato |
| `4x_foolhardy_Remacri` | nitidez agressiva | mais contrastado |

Todos são **4x** (output linear = 4× input). Sem inferência multi-escala no v1.

---

## Setup

```bash
git clone <repo> ~/projetos/inemaupsk
cd ~/projetos/inemaupsk

# baixa os 4 pesos (.pth) — ~256 MB total, idempotente
./models/download.sh

# sobe
docker compose up -d --build
```

Verifica:

```bash
curl -s http://localhost:8002/health | jq
# {"status":"ok", "available":[...4 models...], "cached":[], "default":"4x-UltraSharp.pth", ...}
```

UI: **http://localhost:8002**

---

## API HTTP

Mesmo padrão do `inemaimg`: JSON, base64 entrada/saída, lock asyncio serializa GPU.

### `POST /upscale` — um modelo

```json
{ "image": "<base64 PNG/JPEG>", "model": "4x-UltraSharp" }
```

`model` é opcional (default `4x-UltraSharp`), aceita com ou sem `.pth`.

Resposta:

```json
{
  "model": "4x-UltraSharp.pth",
  "image": "<base64 PNG>",
  "input_size": [867, 600],
  "output_size": [3468, 2400],
  "elapsed_s": 4.48
}
```

### `POST /upscale_all` — todos os modelos em sequência

```json
{ "image": "<base64>" }
```

Resposta: `{ "input_size": [...], "results": [{model, image, output_size, elapsed_s}, ...] }`

### `GET /models` — lista modelos disponíveis
### `GET /health` — status, cache, VRAM alocada

### Limites

- Input long side ≤ **4096 px** → HTTP `413` acima.
- Output sempre **PNG** (não acumula compressão JPEG).
- **Sem auth no v1** — `0.0.0.0:8002` aberto na LAN. Fecha pra `127.0.0.1` ou põe API key antes de expor pra fora.

---

## Exemplos de chamada

### curl

```bash
# único modelo
curl -sS -X POST http://localhost:8002/upscale \
  -H 'Content-Type: application/json' \
  -d "{\"image\":\"$(base64 -w0 foto.jpg)\",\"model\":\"RealESRGAN_x4plus\"}" \
  | jq -r .image | base64 -d > foto_4x.png

# todos pra comparar
curl -sS -X POST http://localhost:8002/upscale_all \
  -H 'Content-Type: application/json' \
  -d "{\"image\":\"$(base64 -w0 foto.jpg)\"}" \
  | jq '.results[] | {model, elapsed_s, output_size}'
```

### Python

```python
import base64, requests

with open("foto.jpg", "rb") as f:
    b64 = base64.b64encode(f.read()).decode()

r = requests.post(
    "http://localhost:8002/upscale",
    json={"image": b64, "model": "4x-UltraSharp"},
    timeout=300,
)
r.raise_for_status()
data = r.json()
open("foto_4x.png", "wb").write(base64.b64decode(data["image"]))
print(data["elapsed_s"], "s ->", data["output_size"])
```

### n8n / agentes

HTTP Request node → `POST http://<host>:8002/upscale`, body JSON com `image`
em base64. Retorno em `image` (base64 PNG) — decodifica num Function node ou
encaminha pra próximo step que aceita base64.

---

## UI web (`/`)

`http://localhost:8002` — playground minimalista:

1. **Imagem** — arrastar, clicar pra escolher, ou colar (Ctrl+V do clipboard)
2. **Modo** — toggle `um modelo` ↔ `todos (comparar)`
3. **Modelo** (só no modo "um") — radio button
4. **Upscale** — botão verde; cards lado a lado com tempo + botão download.
   Clica na imagem pra lightbox em tela cheia.

Modelos carregam **lazy** (1ª chamada) e ficam **cacheados** em VRAM.

---

## Latência observada (DGX Spark / GB10)

| input | output | por modelo |
|---|---|---|
| 150×100 | 600×400 | ~0.1 s |
| 256×256 | 1024×1024 | ~0.85 s |
| 867×600 | 3468×2400 | ~4.5 s |

`/upscale_all` é ~4× isso (4 modelos em sequência sob lock).

---

## Layout

```
.
├── server.py            # FastAPI: /upscale, /upscale_all, /models, /health, /
├── loaders/
│   └── upscale.py       # UpscaleLoader (Spandrel + tiled inference)
├── web/index.html       # Playground UI (live-mounted)
├── models/
│   ├── download.sh      # baixa os 4 pesos
│   └── upscale/*.pth    # weights (gitignored)
├── outputs/             # debug dumps (gitignored exceto .gitkeep)
├── Dockerfile           # NGC PyTorch 26.03-py3
├── docker-compose.yml   # nvidia runtime, porta 8002, mem cap 8g
├── requirements.txt
├── PROJETO.md           # decisões + perguntas em aberto (ler antes de mudar arquitetura)
└── CLAUDE.md            # contexto pra futuras sessões Claude Code
```

---

## Operação

- **GPU compartilhada** com `inemaimg`, sem MPS → containers competem por tempo.
  Pra batch grande (11M fotos), rodar em janela ociosa ou GPU separada (ver `PROJETO.md`).
- **mem_limit: 8g** no compose (SR é leve; `inemaimg` usa até 30g).
- **Lock asyncio** garante que chamadas concorrentes enfileiram, não OOMam.
- **Tiling interno** (`tile=512, pad=16`) permite inputs até 4096px sem estourar VRAM.

## Roadmap

- Endpoint multipart `POST /upscale_file` (upload direto, sem base64)
- Header `X-API-Key` opcional
- Worker batch (fila/diretório, checkpoint, idempotência) — bloqueado nas
  8 perguntas em `PROJETO.md`
- 2× modelos especializados (arte/texto/anime) quando aparecer a demanda
