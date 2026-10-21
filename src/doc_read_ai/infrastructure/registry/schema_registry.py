"""Adapters de SchemaRegistryPort: um JSON por tipo de documento em disco, ou em memória."""

import logging
import os
import re
from pathlib import Path

from pydantic import ValidationError

from doc_read_ai.domain.analysis import RegisteredSchema
from doc_read_ai.domain.ports import SchemaRegistryPort

logger = logging.getLogger(__name__)

# document_type vira nome de arquivo: só aceita o formato normalizado (sem path traversal).
_VALID_TYPE_ID = re.compile(r"^[a-z0-9_]+$")


class JsonFileSchemaRegistry(SchemaRegistryPort):
    """Salva `<diretorio>/<document_type>.json`. Fácil de revisar, editar à mão e versionar."""

    def __init__(self, directory: Path) -> None:
        self._dir = directory

    def _path(self, document_type: str) -> Path:
        if not _VALID_TYPE_ID.match(document_type):
            raise ValueError(f"document_type inválido para o registro: {document_type!r}")
        return self._dir / f"{document_type}.json"

    def get(self, document_type: str) -> RegisteredSchema | None:
        path = self._path(document_type)
        return self._read(path) if path.is_file() else None

    def save(self, schema: RegisteredSchema) -> None:
        path = self._path(schema.document_type)
        self._dir.mkdir(parents=True, exist_ok=True)
        # Escreve num temporário e troca: um processo concorrente nunca lê JSON pela metade.
        tmp = path.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(schema.model_dump_json(indent=2), encoding="utf-8")
        tmp.replace(path)
        logger.info("Schema '%s' salvo em %s", schema.document_type, path)

    def list_all(self) -> list[RegisteredSchema]:
        if not self._dir.is_dir():
            return []
        schemas = (self._read(p) for p in sorted(self._dir.glob("*.json")))
        return [s for s in schemas if s is not None]

    def delete(self, document_type: str) -> bool:
        path = self._path(document_type)
        if not path.is_file():
            return False
        path.unlink()
        return True

    @staticmethod
    def _read(path: Path) -> RegisteredSchema | None:
        try:
            return RegisteredSchema.model_validate_json(path.read_text(encoding="utf-8"))
        except (OSError, ValidationError) as exc:
            # Um arquivo editado errado não derruba o pipeline: o tipo é tratado como novo.
            logger.warning("Ignorando schema inválido em %s: %s", path, exc)
            return None


class InMemorySchemaRegistry(SchemaRegistryPort):
    """Registro que dura só o processo: usado quando o registro está desligado e nos testes."""

    def __init__(self) -> None:
        self._schemas: dict[str, RegisteredSchema] = {}

    def get(self, document_type: str) -> RegisteredSchema | None:
        return self._schemas.get(document_type)

    def save(self, schema: RegisteredSchema) -> None:
        self._schemas[schema.document_type] = schema

    def list_all(self) -> list[RegisteredSchema]:
        return list(self._schemas.values())

    def delete(self, document_type: str) -> bool:
        return self._schemas.pop(document_type, None) is not None
