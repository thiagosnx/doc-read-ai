"""Configuração da observabilidade (.env + variáveis de ambiente).

Os exportadores OTLP NÃO são configurados aqui: usam só as variáveis padrão do OpenTelemetry,
lidas direto do ambiente pelo SDK. Um sinal só ganha exportador OTLP quando tem endpoint e o
seu `OTEL_<SINAL>_EXPORTER` não é `none`:

    OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318         # base para os três sinais
    OTEL_EXPORTER_OTLP_TRACES_ENDPOINT=...                    # ou um endpoint por sinal
    OTEL_EXPORTER_OTLP_HEADERS=Authorization=Bearer%20xyz
    OTEL_TRACES_EXPORTER / OTEL_METRICS_EXPORTER / OTEL_LOGS_EXPORTER=otlp|none

Langfuse aceita só traces:

    OTEL_EXPORTER_OTLP_TRACES_ENDPOINT=https://cloud.langfuse.com/api/public/otel/v1/traces
    OTEL_EXPORTER_OTLP_HEADERS=Authorization=Basic%20<base64 de pk-lf-...:sk-lf-...>
    OTEL_METRICS_EXPORTER=none
    OTEL_LOGS_EXPORTER=none
"""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class ObservabilitySettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    otel_sdk_disabled: bool = False  # OTEL_SDK_DISABLED (padrão OTel): desliga tudo
    otel_service_name: str | None = None  # OTEL_SERVICE_NAME (padrão OTel): vence o do código
    doc_read_telemetry_db: Path = Path("data/telemetry.db")  # spans locais (governança)
    doc_read_telemetry_local: bool = True  # grava os spans no SQLite local
    doc_read_otel_console: bool = False  # imprime os spans no console (depuração)
