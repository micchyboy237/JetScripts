from jet.logger import CustomLogger
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
log_file = os.path.join(script_dir, f"{os.path.splitext(os.path.basename(__file__))[0]}.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
# Open Telemetry

AutoGen has native support for [open telemetry](https://opentelemetry.io/). This allows you to collect telemetry data from your application and send it to a telemetry backend of your choosing.

These are the components that are currently instrumented:
- Runtime (Single Threaded Agent Runtime, Worker Agent Runtime)

## Instrumenting your application
To instrument your application, you will need an sdk and an exporter. You may already have these if your application is already instrumented with open telemetry.

## Clean instrumentation

If you do not have open telemetry set up in your application, you can follow these steps to instrument your application.
"""
logger.info("# Open Telemetry")

pip install opentelemetry-sdk

"""
Depending on your open telemetry collector, you can use grpc or http to export your telemetry.
"""
logger.info("Depending on your open telemetry collector, you can use grpc or http to export your telemetry.")

pip install opentelemetry-exporter-otlp-proto-http
pip install opentelemetry-exporter-otlp-proto-grpc

"""
Next, we need to get a tracer provider:
"""
logger.info("Next, we need to get a tracer provider:")


def configure_oltp_tracing(endpoint: str = None) -> trace.TracerProvider:
    tracer_provider = TracerProvider(resource=Resource({"service.name": "my-service"}))
    processor = BatchSpanProcessor(OTLPSpanExporter())
    tracer_provider.add_span_processor(processor)
    trace.set_tracer_provider(tracer_provider)

    return tracer_provider

"""
Now you can send the trace_provider when creating your runtime:
"""
logger.info("Now you can send the trace_provider when creating your runtime:")

single_threaded_runtime = SingleThreadedAgentRuntime(tracer_provider=tracer_provider)
worker_runtime = GrpcWorkerAgentRuntime(tracer_provider=tracer_provider)

"""
And that's it! Your application is now instrumented with open telemetry. You can now view your telemetry data in your telemetry backend.

### Existing instrumentation

If you have open telemetry already set up in your application, you can pass the tracer provider to the runtime when creating it:
"""
logger.info("### Existing instrumentation")


tracer_provider = trace.get_tracer_provider()

single_threaded_runtime = SingleThreadedAgentRuntime(tracer_provider=tracer_provider)
worker_runtime = GrpcWorkerAgentRuntime(tracer_provider=tracer_provider)

logger.info("\n\n[DONE]", bright=True)