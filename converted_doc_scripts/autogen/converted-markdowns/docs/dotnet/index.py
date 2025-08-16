from jet.logger import CustomLogger
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
---
_disableAffix: true
---

<div class="center">
    <h1>AutoGen .NET</h1>
    <p class="subheader">
    A <i>.NET</i> framework for building AI agents and applications
    </p>
</div>

<div class="row">
  <div class="col-sm-6">
    <div class="card">
      <div class="card-body">
        <h5 class="card-title">Core</h5>
<p>

[![dotnet-ci](https://github.com/microsoft/autogen/actions/workflows/dotnet-build.yml/badge.svg)](https://github.com/microsoft/autogen/actions/workflows/dotnet-build.yml)
[![NuGet version](https://badge.fury.io/nu/Microsoft.AutoGen.Contracts.svg)](https://badge.fury.io/nu/Microsoft.AutoGen.Contracts)
[![NuGet version](https://badge.fury.io/nu/Microsoft.AutoGen.Core.svg)](https://badge.fury.io/nu/Microsoft.AutoGen.Core)
[![NuGet version](https://badge.fury.io/nu/Microsoft.AutoGen.Core.Grpc.svg)](https://badge.fury.io/nu/Microsoft.AutoGen.Core.Grpc)
[![NuGet version](https://badge.fury.io/nu/Microsoft.AutoGen.RuntimeGateway.Grpc.svg)](https://badge.fury.io/nu/Microsoft.AutoGen.RuntimeGateway.Grpc)
[![NuGet version](https://badge.fury.io/nu/Microsoft.AutoGen.AgentHost.svg)](https://badge.fury.io/nu/Microsoft.AutoGen.AgentHost)

</p>
        <p class="card-text">An event-driven programming framework for building scalable multi-agent AI systems.</p>

- Deterministic and dynamic agentic workflows for business processes
- Research on multi-agent collaboration
- Distributed agents for multi-language applications
- integration with event-driven, cloud native applications

*Start here if you are building workflows or distributed agent systems*

<p>
<div class="highlight">
<pre id="codecell0" tabindex="0">
"""
logger.info("A <i>.NET</i> framework for building AI agents and applications")

dotnet add package Microsoft.AutoGen.Contracts
dotnet add package Microsoft.AutoGen.Core

dotnet add package Microsoft.AutoGen.RuntimeGateway.Grpc
dotnet add package Microsoft.AutoGen.AgentHost

dotnet add package Microsoft.AutoGen.Agents
dotnet add package Microsoft.AutoGen.Extensions.Aspire
dotnet add package Microsoft.AutoGen.Extensions.MEAI
dotnet add package Microsoft.AutoGen.Extensions.SemanticKernel

"""
</pre></div></p>
<p>
        <a href="core/index.md" class="btn btn-primary">Get started</a>
      </div>
    </div>
  </div>
  <div class="col-sm-6">
    <div class="card">
      <div class="card-body">
        <h5 class="card-title">AgentChat</h5>
        <p class="card-text">A programming framework for building conversational single and multi-agent applications. Built on Core.</p>
        <a href="#" class="btn btn-primary disabled">Coming soon</a>
      </div>
    </div>
  </div>
</div>
"""

logger.info("\n\n[DONE]", bright=True)