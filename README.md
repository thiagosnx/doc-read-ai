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
            │   analyst    │  "isso é uma NF-e; ela tem emitente, itens, impostos..."
            └──────┬───────┘  -> DocumentAnalysis.fields (list[FieldSpec])
                   │
            ┌──────▼───────┐
            │ build_schema │  FieldSpec -> pydantic.create_model (determinístico)
            └──────┬───────┘
                   │
            ┌──────▼───────┐
            │  extractor   │  preenche o modelo dinâmico com structured output
            └──────┬───────┘
                   ▼
               EXTRACTED  (data + json_schema + análise)
```

Qualquer erro em um nó (API, recusa, schema inválido) encerra o grafo com status `FAILED` e a mensagem em `errors`.

## Arquitetura (hexagonal + strategy)

```
src/doc_read_ai/
├── domain/              # núcleo: entidades, portas, erros. Não depende de nada externo
│   ├── document.py      # Document (bytes + media type)
│   ├── analysis.py      # ValidationVerdict, DocumentAnalysis, FieldSpec, ExtractionResult
│   ├── config.py        # AgentConfig / ProviderConfig / AppConfig
│   └── ports.py         # LLMPort, LLMFactoryPort, DocumentLoaderPort
├── application/         # casos de uso: só conhece as portas
│   ├── agents/          # orchestrator, analyst, extractor
│   ├── prompts.py       # prompts padrão (sobrescrevíveis via YAML)
│   ├── schema_builder.py# FieldSpec -> modelo Pydantic em runtime
│   ├── graph.py         # StateGraph do LangGraph
│   └── use_cases.py     # ProcessDocumentUseCase
├── infrastructure/      # adapters
│   ├── llm/             # AnthropicLLM + LLMRegistry (strategy por provedor)
│   ├── loaders/         # LocalFileLoader (detecção por magic bytes)
│   └── config/          # leitura do YAML e do .env
├── interfaces/cli.py    # adapter de entrada (Typer)
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

## Formatos aceitos

PDF, PNG, JPEG, WEBP e GIF. O tipo é detectado pelos bytes do arquivo, não pela extensão.

## Privacidade

Documentos processados e resultados nunca vão para o git: `samples/` e `output/` são ignorados, assim como qualquer PDF ou imagem no repositório (veja `.gitignore`). Os dados dos testes são fictícios. A chave da API fica só no `.env` local.

Os documentos são enviados ao provedor de LLM configurado (Anthropic, por padrão) para processamento.

## Licença

MIT
