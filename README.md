# doc-read-ai

Leitura autônoma de documentos (PDF e imagem) com agentes LangGraph, arquitetura hexagonal e schemas Pydantic dinâmicos.

Não existe uma lista fixa de tipos de documento. O sistema descobre sozinho o que recebeu (nota fiscal, fatura, boleto, RG, contrato, currículo...), desenha o schema de dados daquele documento e extrai os dados nesse schema.

## Fluxo

```
            ┌──────────────┐   inválido
arquivo ──▶ │ orchestrator │ ───────────▶ REJECTED
            └──────┬───────┘
                   │ válido
            ┌──────▼───────┐
            │  classifier  │  "isso é uma NF-e" (reaproveita o id de um tipo já registrado)
            └──────┬───────┘
            ┌──────▼─────────┐  tipo conhecido
            │ resolve_schema │ ────────────────────────┐
            └──────┬─────────┘                         │
                   │ tipo novo                         │
            ┌──────▼───────┐                           │
            │   analyst    │  desenha o schema e salva │
            └──────┬───────┘  no registro              │
            ┌──────▼───────┐ ◀─────────────────────────┘
            │ build_schema │  FieldSpec -> pydantic.create_model (determinístico)
            └──────┬───────┘
            ┌──────▼───────┐
            │  extractor   │  preenche o modelo dinâmico
            └──────┬───────┘
                   ▼
               EXTRACTED  (data + json_schema + análise)
```

Qualquer erro em um nó (API, recusa, schema inválido) encerra o grafo com status `FAILED` e a mensagem em `errors`.

### Registro de schemas

O primeiro documento de um tipo define o schema daquele tipo; os próximos reaproveitam. Isso mantém a saída estável (toda NF-e sai com os mesmos campos, no mesmo formato) e pula o analista, a etapa mais cara. O classificador recebe o catálogo de tipos registrados e reutiliza o id exato quando o documento é de um tipo conhecido.

Os schemas ficam em `schemas/<tipo>.json` (fora do git por padrão). Dá para revisar e editar à mão.

```bash
uv run doc-read schemas list                          # tipos registrados
uv run doc-read schemas show nota_fiscal_eletronica   # campos de um tipo
uv run doc-read schemas rm nota_fiscal_eletronica     # força redesenhar no próximo documento
```

Para desligar (schema novo a cada documento): `schema_registry.enabled: false` no YAML.

## Arquitetura (hexagonal + strategy)

```
src/doc_read_ai/
├── domain/              # núcleo: entidades, portas, erros. Não depende de nada externo
│   ├── document.py      # Document (bytes + media type)
│   ├── analysis.py      # ValidationVerdict, DocumentAnalysis, FieldSpec, ExtractionResult
│   ├── config.py        # AgentConfig / ProviderConfig / AppConfig
│   └── ports.py         # LLMPort, LLMFactoryPort, SchemaRegistryPort, DocumentLoaderPort
├── application/         # casos de uso: só conhece as portas
│   ├── agents/          # orchestrator, classifier, analyst, extractor
│   ├── prompts.py       # prompts padrão (sobrescrevíveis via YAML)
│   ├── schema_builder.py# FieldSpec -> modelo Pydantic em runtime
│   ├── graph.py         # StateGraph do LangGraph
│   └── use_cases.py     # ProcessDocumentUseCase
├── infrastructure/      # adapters
│   ├── llm/             # AnthropicLLM + LLMRegistry (strategy por provedor)
│   ├── loaders/         # LocalFileLoader (detecção por magic bytes)
│   ├── registry/        # registro de schemas (JSON em disco / memória)
│   ├── observability/   # OpenTelemetry: providers, exportadores, spans locais
│   ├── persistence/     # histórico de execuções (SQLite)
│   └── config/          # leitura do YAML e do .env
├── interfaces/
│   ├── cli.py           # adapter de entrada (Typer)
│   └── api/             # adapter de entrada (FastAPI)
└── bootstrap.py         # composition root: único lugar que conhece as classes concretas
```

**Trocar de provedor:** escreva um adapter que implemente `LLMPort` (ex.: `OpenAILLM`), registre em `build_default_llm_factory` e mude `provider:` no YAML. Agentes, grafo e domínio não mudam. Cada adapter decide como enviar PDF/imagem para o seu provedor, por isso o `Document` chega até ele cru.

**Um provedor/modelo por agente:** cada agente em `config/agents.yml` tem `provider`, `model`, `max_tokens`, `params` (opções do provedor) e `system_prompt` opcional.

## Como rodar

```bash
uv sync
# cole sua chave em .env -> ANTHROPIC_API_KEY=...

uv run doc-read agents                      # mostra a config dos agentes
uv run doc-read process samples/nota.pdf    # um arquivo
uv run doc-read process samples/            # pasta inteira (concorrente)
```

O JSON de cada documento vai para `output/<nome>.json`.

```bash
uv run pytest      # testes com LLM fake, sem chamadas de rede
uv run ruff check
```

## API (FastAPI)

```bash
uv run doc-read-api            # http://127.0.0.1:8000  · docs interativas em /docs
```

| Rota | O que faz |
|---|---|
| `POST /v1/documents` | Envia 1+ arquivos (multipart `files`). Processa em segundo plano e devolve 202 com os runs (`?wait=true` espera o fim) |
| `GET /v1/runs` · `GET /v1/runs/{id}` | Histórico e resultado de cada execução |
| `GET /v1/runs/{id}/events` | Progresso ao vivo por SSE (um evento por etapa do grafo + `done`) |
| `GET /v1/runs/{id}/trace` | Trace OpenTelemetry da execução |
| `DELETE /v1/runs/{id}` | Apaga a execução e o resultado (direito de eliminação, LGPD) |
| `GET/PUT/DELETE /v1/schemas/...` | Registro de schemas por tipo de documento |
| `GET /v1/agents` | Agentes, provedores e modelos (sem segredos) |
| `GET /v1/observability/summary` | Governança: execuções, tokens, custo por agente/modelo, latência por etapa e como os dados são tratados |
| `GET /health` · `GET /ready` | Liveness e readiness |

O arquivo enviado nunca é gravado em disco (só o hash SHA-256). O OpenAPI em `/openapi.json` serve para gerar o cliente do frontend.

## Observabilidade (OpenTelemetry)

Tudo é OTel padrão, sem SDK de fornecedor na aplicação. Cada documento gera um trace:

```
POST /v1/documents                      (FastAPI)
└─ doc_read.process_document            (caso de uso)
   ├─ doc_read.step validate
   │  └─ invoke_agent orchestrator
   │     └─ anthropic.messages.create   (SDK da Anthropic, com gen_ai.usage.*)
   ├─ doc_read.step classify ...
   └─ doc_read.step extract ...
```

Métricas: `doc_read.documents`, `doc_read.document.duration`, `doc_read.step.duration`, `doc_read.llm.tokens`. Os logs carregam o `trace_id`.

Para onde vai:
- **Local (sempre):** os spans são gravados em `data/telemetry.db` e alimentam `/v1/observability/*`.
- **Qualquer backend OTLP** (collector, Grafana, Honeycomb, Datadog, Langfuse...): variáveis `OTEL_EXPORTER_OTLP_*` padrão (veja `.env.example`). Sem endpoint, nada sai da máquina.

Privacidade: nome de arquivo nunca vai para a telemetria; prompts e respostas não são exportados (modo `metadata_only` do SDK); o banco local descarta atributos de conteúdo mesmo se o modo for alterado.

## Formatos aceitos

PDF, PNG, JPEG, WEBP e GIF. O tipo é detectado pelos bytes do arquivo, não pela extensão.

## Privacidade

Documentos processados e resultados nunca vão para o git: `samples/`, `output/` e `schemas/` são ignorados, assim como qualquer PDF ou imagem no repositório (veja `.gitignore`). Os dados dos testes são fictícios. A chave da API fica só no `.env` local.

Os documentos são enviados ao provedor de LLM configurado (Anthropic, por padrão) para processamento.

## Licença

MIT
