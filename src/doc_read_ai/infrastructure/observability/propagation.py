"""Propaga atributos do span raiz para os descendentes, para filtrar qualquer span por execução."""

from opentelemetry import trace
from opentelemetry.context import Context
from opentelemetry.sdk.trace import Span, SpanProcessor

# Atributos do span raiz que todo span descendente herda (inclusive os spans do SDK).
PROPAGATED_ATTRIBUTES = ("doc_read.run_id",)


class AttributePropagationProcessor(SpanProcessor):
    """Copia atributos do pai para o filho no início do span."""

    def on_start(self, span: Span, parent_context: Context | None = None) -> None:
        parent = trace.get_current_span(parent_context)
        attributes = getattr(parent, "attributes", None) or {}
        for key in PROPAGATED_ATTRIBUTES:
            if key in attributes and key not in (span.attributes or {}):
                span.set_attribute(key, attributes[key])
