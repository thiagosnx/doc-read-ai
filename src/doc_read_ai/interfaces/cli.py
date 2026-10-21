"""CLI (adapter de entrada): `doc-read process` e `doc-read agents`."""

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Annotated, Any, NoReturn

import typer
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table
from rich.tree import Tree

from doc_read_ai.bootstrap import (
    build_schema_registry,
    build_use_case,
    configure_logging,
    load_config,
)
from doc_read_ai.domain.analysis import (
    ExtractionResult,
    FieldSpec,
    PipelineStatus,
    SchemaSource,
)
from doc_read_ai.domain.config import AppConfig
from doc_read_ai.domain.errors import DocReadError
from doc_read_ai.infrastructure.config import Settings
from doc_read_ai.infrastructure.loaders import SUPPORTED_EXTENSIONS
from doc_read_ai.infrastructure.observability import configure_observability

app = typer.Typer(
    help="Leitura autônoma de documentos (PDF/imagem) com agentes de IA.",
    no_args_is_help=True,
    add_completion=False,
)
# No Windows a saída redirecionada usa cp1252 e quebra os acentos; força UTF-8.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")

console = Console()

schemas_app = typer.Typer(help="Registro de schemas: um schema estável por tipo de documento.")
app.add_typer(schemas_app, name="schemas")

SCHEMA_SOURCE_LABEL = {
    SchemaSource.REGISTRY: "reaproveitado do registro",
    SchemaSource.DESIGNED: "tipo novo, schema criado e salvo no registro",
}

STATUS_STYLE = {
    PipelineStatus.EXTRACTED: ("green", "EXTRAÍDO"),
    PipelineStatus.REJECTED: ("yellow", "REJEITADO"),
    PipelineStatus.FAILED: ("red", "FALHOU"),
}

ConfigOption = Annotated[
    Path | None, typer.Option("--config", "-c", help="YAML de agentes (padrão: DOC_READ_CONFIG)")
]


@app.command()
def process(
    path: Annotated[Path, typer.Argument(help="Arquivo ou pasta (não recursivo) com documentos")],
    out: Annotated[Path, typer.Option("--out", "-o", help="Pasta de saída dos JSON")] = Path(
        "output"
    ),
    config: ConfigOption = None,
    concurrency: Annotated[
        int, typer.Option("--concurrency", "-n", min=1, help="Documentos em paralelo")
    ] = 3,
) -> None:
    """Classifica e extrai os dados de um ou mais documentos."""
    configure_logging(Settings().log_level)
    files = _collect_files(path)
    app_config = _load_config_or_exit(config)
    _ensure_api_keys(app_config)
    # Antes do use case: os clientes do SDK só emitem spans se o provider global já existir.
    telemetry = configure_observability(service_name="doc-read-ai-cli", pricing=app_config.pricing)

    try:
        use_case = build_use_case(config_path=config)
    except DocReadError as exc:
        _fail(escape(str(exc)))

    out.mkdir(parents=True, exist_ok=True)
    console.print(f"[bold]Processando {len(files)} arquivo(s)[/] (concorrência: {concurrency})\n")
    try:
        results = asyncio.run(_process_all(use_case, files, out, concurrency))
    finally:
        telemetry.shutdown()  # descarrega os exportadores em lote antes de sair

    counts = {s: sum(r.status == s for r in results) for s in PipelineStatus}
    console.print(
        f"\n[green]{counts[PipelineStatus.EXTRACTED]} extraído(s)[/] · "
        f"[yellow]{counts[PipelineStatus.REJECTED]} rejeitado(s)[/] · "
        f"[red]{counts[PipelineStatus.FAILED]} com falha[/] · JSON em [bold]{out}[/]"
    )
    if counts[PipelineStatus.FAILED]:
        raise typer.Exit(code=1)


@app.command()
def agents(config: ConfigOption = None) -> None:
    """Lista os agentes configurados (provedor, modelo, parâmetros)."""
    app_config = _load_config_or_exit(config)
    table = Table(title="Agentes configurados")
    for col in ("Agente", "Provedor", "Modelo", "max_tokens", "params"):
        table.add_column(col, overflow="fold")
    for agent in app_config.agents.values():
        params = ", ".join(f"{k}={v}" for k, v in agent.params.items()) or "-"
        table.add_row(agent.name, agent.provider, agent.model, str(agent.max_tokens), params)
    console.print(table)


@schemas_app.command("list")
def schemas_list(config: ConfigOption = None) -> None:
    """Lista os tipos de documento com schema registrado."""
    registry = build_schema_registry(_load_config_or_exit(config))
    schemas = registry.list_all()
    if not schemas:
        console.print("[dim]Nenhum schema registrado ainda.[/]")
        return
    table = Table(title="Schemas registrados")
    for col in ("Tipo", "Nome", "Categoria", "Campos", "Criado em"):
        table.add_column(col, overflow="fold")
    for s in schemas:
        created = s.created_at.astimezone().strftime("%Y-%m-%d %H:%M")
        table.add_row(
            s.document_type, s.document_type_label, s.category, str(len(s.fields)), created
        )
    console.print(table)


@schemas_app.command("show")
def schemas_show(
    document_type: Annotated[str, typer.Argument(help="Id do tipo, ex: nota_fiscal_eletronica")],
    config: ConfigOption = None,
) -> None:
    """Mostra os campos do schema de um tipo de documento."""
    registry = build_schema_registry(_load_config_or_exit(config))
    schema = registry.get(document_type)
    if schema is None:
        _fail(f"Nenhum schema registrado para '{document_type}'. Veja `doc-read schemas list`.")
    tree = Tree(f"[bold]{escape(schema.document_type_label)}[/] [dim]({schema.document_type})[/]")
    for spec in schema.fields:
        _add_field(tree, spec)
    console.print(tree)


@schemas_app.command("rm")
def schemas_rm(
    document_type: Annotated[str, typer.Argument(help="Id do tipo, ex: nota_fiscal_eletronica")],
    config: ConfigOption = None,
) -> None:
    """Remove o schema de um tipo; o próximo documento desse tipo terá o schema redesenhado."""
    registry = build_schema_registry(_load_config_or_exit(config))
    if not registry.delete(document_type):
        _fail(f"Nenhum schema registrado para '{document_type}'.")
    console.print(f"Schema '{document_type}' removido.")


def _add_field(parent: Tree, spec: FieldSpec) -> None:
    kind = spec.type.value
    if spec.type == "list" and not spec.subfields:
        kind = f"list[{(spec.item_type or 'string')}]"
    required = "" if spec.required else " [dim](opcional)[/]"
    node = parent.add(
        f"[cyan]{spec.name}[/]: {kind}{required} [dim]— {escape(spec.description)}[/]"
    )
    for sub in spec.subfields:
        sub_required = "" if sub.required else " [dim](opcional)[/]"
        node.add(
            f"[cyan]{sub.name}[/]: {sub.type.value}{sub_required} "
            f"[dim]— {escape(sub.description)}[/]"
        )


# --- execução ---------------------------------------------------------------------------


async def _process_all(use_case, files: list[Path], out: Path, concurrency: int):
    semaphore = asyncio.Semaphore(concurrency)

    async def run(file: Path) -> ExtractionResult:
        async with semaphore:
            try:
                result = await use_case.execute(file)
            except Exception as exc:  # o caso de uso não deveria levantar, mas não derruba o lote
                result = ExtractionResult(
                    filename=file.name,
                    media_type="desconhecido",
                    status=PipelineStatus.FAILED,
                    errors=[f"{type(exc).__name__}: {exc}"],
                )
        target = out / f"{file.stem}.json"
        target.write_text(
            json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        _render_result(result, target)
        return result

    return await asyncio.gather(*(run(f) for f in files))


def _collect_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if path.is_dir():
        files = sorted(p for p in path.iterdir() if p.is_file() and _is_supported(p))
        if not files:
            _fail(
                f"Nenhum documento suportado em {path} ({', '.join(sorted(SUPPORTED_EXTENSIONS))})"
            )
        return files
    _fail(f"Caminho não encontrado: {path}")


def _is_supported(path: Path) -> bool:
    ext = path.suffix.lower()
    return ext in SUPPORTED_EXTENSIONS or ext.lstrip(".") in SUPPORTED_EXTENSIONS


def _load_config_or_exit(config: Path | None) -> AppConfig:
    try:
        return load_config(config)
    except DocReadError as exc:
        _fail(escape(str(exc)))


def _ensure_api_keys(app_config: AppConfig) -> None:
    """Só cobra a chave de provedores que declaram api_key_env."""
    missing = set()
    for agent in app_config.agents.values():
        provider = app_config.providers.get(agent.provider)
        if (
            provider
            and provider.api_key_env
            and not os.environ.get(provider.api_key_env, "").strip()
        ):
            missing.add(provider.api_key_env)
    if missing:
        names = ", ".join(sorted(missing))
        _fail(
            f"Chave de API ausente: {names}.\n"
            f"Abra o arquivo [bold].env[/] na raiz do projeto e cole a chave, ex.:\n"
            f"  {sorted(missing)[0]}=sk-...\n"
            "(use o .env.example como modelo)"
        )


def _fail(message: str, code: int = 2) -> NoReturn:
    console.print(f"[bold red]Erro:[/] {message}")
    raise typer.Exit(code=code)


# --- renderização -----------------------------------------------------------------------


def _render_result(result: ExtractionResult, target: Path) -> None:
    color, label = STATUS_STYLE[result.status]
    lines: list[str] = [f"[bold {color}]{label}[/]  [dim]{result.media_type}[/]"]
    if result.verdict:
        lines.append(f"[bold]Validação:[/] {escape(result.verdict.reason)}")
    if result.analysis:
        a = result.analysis
        lines.append(f"[bold]Tipo:[/] {escape(a.document_type_label)} [dim]({a.confidence:.0%})[/]")
        lines.append(f"[bold]Resumo:[/] {escape(a.summary)}")
    if result.schema_source:
        source = SCHEMA_SOURCE_LABEL[result.schema_source]
        lines.append(f"[bold]Schema:[/] {result.analysis.document_type} [dim]({source})[/]")
    for error in result.errors:
        lines.append(f"[red]• {escape(error)}[/]")
    lines.append(f"[dim]JSON: {target}[/]")

    console.print(
        Panel("\n".join(lines), title=f"[bold]{escape(result.filename)}[/]", border_style=color)
    )
    if result.data:
        console.print(_data_tree(result.data))


def _data_tree(data: dict[str, Any]) -> Tree:
    tree = Tree("[bold]Dados extraídos[/]")
    _add_node(tree, data)
    return tree


def _add_node(parent: Tree, value: Any) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if isinstance(item, dict | list) and item:
                _add_node(parent.add(f"[cyan]{escape(str(key))}[/]"), item)
            else:
                parent.add(f"[cyan]{escape(str(key))}[/]: {_scalar(item)}")
    elif isinstance(value, list):
        if all(not isinstance(i, dict | list) for i in value):
            parent.add(", ".join(_scalar(i) for i in value))
            return
        for idx, item in enumerate(value, start=1):
            if isinstance(item, dict | list):
                _add_node(parent.add(f"[magenta]#{idx}[/]"), item)
            else:
                parent.add(f"[magenta]#{idx}[/]: {_scalar(item)}")


def _scalar(value: Any) -> str:
    if value is None:
        return "[dim]—[/]"
    if isinstance(value, dict | list):
        return "[dim]vazio[/]"
    if isinstance(value, bool):
        return "sim" if value else "não"
    return escape(str(value))
