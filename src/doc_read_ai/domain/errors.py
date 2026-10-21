class DocReadError(Exception):
    """Base de todos os erros do sistema."""


class UnsupportedDocumentError(DocReadError):
    """Formato/tamanho de arquivo não aceito (aceitamos só PDF e imagens)."""


class LLMError(DocReadError):
    """Falha do provedor de LLM: erro de API, recusa, truncamento ou saída fora do schema."""


class ProviderNotRegisteredError(DocReadError):
    """O YAML referencia um provider que não tem adapter registrado."""


class SchemaBuildError(DocReadError):
    """O schema proposto pelo analista não pôde virar um modelo Pydantic válido."""
