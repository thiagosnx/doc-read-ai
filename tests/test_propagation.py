"""O run_id do span raiz chega a todos os descendentes, inclusive os spans do SDK do LLM."""

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from doc_read_ai.infrastructure.observability.propagation import AttributePropagationProcessor


def test_run_id_propagates_to_descendants() -> None:
    sink = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(AttributePropagationProcessor())
    provider.add_span_processor(SimpleSpanProcessor(sink))
    tracer = provider.get_tracer("doc_read_ai")

    with tracer.start_as_current_span(
        "doc_read.process_document", attributes={"doc_read.run_id": "run-9"}
    ):
        with tracer.start_as_current_span("doc_read.step extract"):
            with tracer.start_as_current_span("anthropic.messages.create"):
                pass
    with tracer.start_as_current_span("fora_de_execucao"):
        pass

    spans = {s.name: s.attributes for s in sink.get_finished_spans()}
    assert spans["doc_read.step extract"]["doc_read.run_id"] == "run-9"
    assert spans["anthropic.messages.create"]["doc_read.run_id"] == "run-9"
    assert "doc_read.run_id" not in spans["fora_de_execucao"]
