from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

provider = TracerProvider()

# ⚠️ Note: NO http:// prefix, and insecure=True for plaintext gRPC
exporter = OTLPSpanExporter(
    endpoint="192.168.68.151:4317",
    insecure=True,
)
provider.add_span_processor(SimpleSpanProcessor(exporter))
trace.set_tracer_provider(provider)

tracer = trace.get_tracer("connectivity-test")
with tracer.start_as_current_span("test-span") as span:
    span.set_attribute("project", "mem0-langgraph-dual-scope")
    print("✅ Span exported successfully")
