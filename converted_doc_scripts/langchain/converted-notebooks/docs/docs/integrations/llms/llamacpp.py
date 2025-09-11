from jet.logger import logger
from langchain_community.llms import LlamaCpp
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
from langchain_core.prompts import PromptTemplate
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger.basicConfig(filename=log_file)
logger.info(f"Logs: {log_file}")

PERSIST_DIR = f"{OUTPUT_DIR}/chroma"
os.makedirs(PERSIST_DIR, exist_ok=True)


"""
# Llama.cpp

[llama-cpp-python](https://github.com/abetlen/llama-cpp-python) is a Python binding for [llama.cpp](https://github.com/ggerganov/llama.cpp).

It supports inference for [many LLMs](https://github.com/ggerganov/llama.cpp#description) models, which can be accessed on [Hugging Face](https://huggingface.co/TheBloke).

This notebook goes over how to run `llama-cpp-python` within LangChain.

**Note: new versions of `llama-cpp-python` use GGUF model files (see [here](https://github.com/abetlen/llama-cpp-python/pull/633)).**

This is a breaking change.
 
To convert existing GGML models to GGUF you can run the following in [llama.cpp](https://github.com/ggerganov/llama.cpp):

```
python ./convert-llama-ggmlv3-to-gguf.py --eps 1e-5 --input models/openorca-platypus2-13b.ggmlv3.q4_0.bin --output models/openorca-platypus2-13b.gguf.q4_0.bin
```

## Installation

There are different options on how to install the llama-cpp package: 
- CPU usage
- CPU + GPU (using one of many BLAS backends)
- Metal GPU (MacOS with Apple Silicon Chip) 

### CPU only installation
"""
logger.info("# Llama.cpp")

# %
p
i
p

i
n
s
t
a
l
l

-
-
u
p
g
r
a
d
e

-
-
q
u
i
e
t


l
l
a
m
a
-
c
p
p
-
p
y
t
h
o
n

"""
### Installation with OpenBLAS / cuBLAS / CLBlast

`llama.cpp` supports multiple BLAS backends for faster processing. Use the `FORCE_CMAKE=1` environment variable to force the use of cmake and install the pip package for the desired BLAS backend ([source](https://github.com/abetlen/llama-cpp-python#installation-with-openblas--cublas--clblast)).

Example installation with cuBLAS backend:
"""
logger.info("### Installation with OpenBLAS / cuBLAS / CLBlast")

# !
C
M
A
K
E
_
A
R
G
S
=
"
-
D
G
G
M
L
_
C
U
D
A
=
o
n
"

F
O
R
C
E
_
C
M
A
K
E
=
1

p
i
p

i
n
s
t
a
l
l

l
l
a
m
a
-
c
p
p
-
p
y
t
h
o
n

"""
**IMPORTANT**: If you have already installed the CPU only version of the package, you need to reinstall it from scratch. Consider the following command:
"""

# !
C
M
A
K
E
_
A
R
G
S
=
"
-
D
G
G
M
L
_
C
U
D
A
=
o
n
"

F
O
R
C
E
_
C
M
A
K
E
=
1

p
i
p

i
n
s
t
a
l
l

-
-
u
p
g
r
a
d
e

-
-
f
o
r
c
e
-
r
e
i
n
s
t
a
l
l

l
l
a
m
a
-
c
p
p
-
p
y
t
h
o
n

-
-
n
o
-
c
a
c
h
e
-
d
i
r

"""
### Installation with Metal

`llama.cpp` supports Apple silicon first-class citizen - optimized via ARM NEON, Accelerate and Metal frameworks. Use the `FORCE_CMAKE=1` environment variable to force the use of cmake and install the pip package for the Metal support ([source](https://github.com/abetlen/llama-cpp-python/blob/main/docs/install/macos.md)).

Example installation with Metal Support:
"""
logger.info("### Installation with Metal")

# !
C
M
A
K
E
_
A
R
G
S
=
"
-
D
L
L
A
M
A
_
M
E
T
A
L
=
o
n
"

F
O
R
C
E
_
C
M
A
K
E
=
1

p
i
p

i
n
s
t
a
l
l

l
l
a
m
a
-
c
p
p
-
p
y
t
h
o
n

"""
**IMPORTANT**: If you have already installed a cpu only version of the package, you need to reinstall it from scratch: consider the following command:
"""

# !
C
M
A
K
E
_
A
R
G
S
=
"
-
D
L
L
A
M
A
_
M
E
T
A
L
=
o
n
"

F
O
R
C
E
_
C
M
A
K
E
=
1

p
i
p

i
n
s
t
a
l
l

l
l
a
m
a
-
c
p
p
-
p
y
t
h
o
n

-
-
f
o
r
c
e
-
r
e
i
n
s
t
a
l
l

-
-
n
o
-
b
i
n
a
r
y

:
a
l
l
:

-
-
n
o
-
c
a
c
h
e
-
d
i
r

"""
### Installation with Windows

It is stable to install the `llama-cpp-python` library by compiling from the source. You can follow most of the instructions in the repository itself but there are some windows specific instructions which might be useful.

Requirements to install the `llama-cpp-python`,

- git
- python
- cmake
- Visual Studio Community (make sure you install this with the following settings)
    - Desktop development with C++
    - Python development
    - Linux embedded development with C++

1. Clone git repository recursively to get `llama.cpp` submodule as well 

```
git clone --recursive -j8 https://github.com/abetlen/llama-cpp-python.git
```

2. Open up a command Prompt and set the following environment variables.


```
set FORCE_CMAKE=1
set CMAKE_ARGS=-DGGML_CUDA=OFF
```
If you have an NVIDIA GPU make sure `DGGML_CUDA` is set to `ON`

#### Compiling and installing

Now you can `cd` into the `llama-cpp-python` directory and install the package

```
python -m pip install -e .
```

**IMPORTANT**: If you have already installed a cpu only version of the package, you need to reinstall it from scratch: consider the following command:
"""
logger.info("### Installation with Windows")

# !
p
y
t
h
o
n

-
m

p
i
p

i
n
s
t
a
l
l

-
e

.

-
-
f
o
r
c
e
-
r
e
i
n
s
t
a
l
l

-
-
n
o
-
c
a
c
h
e
-
d
i
r

"""
## Usage

Make sure you are following all instructions to [install all necessary model files](https://github.com/ggerganov/llama.cpp).

You don't need an `API_TOKEN` as you will run the LLM locally.

It is worth understanding which models are suitable to be used on the desired machine.

[TheBloke's](https://huggingface.co/TheBloke) Hugging Face models have a `Provided files` section that exposes the RAM required to run models of different quantisation sizes and methods (eg: [Llama2-7B-Chat-GGUF](https://huggingface.co/TheBloke/Llama-2-7b-Chat-GGUF#provided-files)).

This [github issue](https://github.com/facebookresearch/llama/issues/425) is also relevant to find the right model for your machine.
"""
logger.info("## Usage")


"""
**Consider using a template that suits your model! Check the models page on Hugging Face etc. to get a correct prompting template.**
"""

template = """Question: {question}

Answer: Let's work this out in a step by step way to be sure we have the right answer."""

prompt = PromptTemplate.from_template(template)

callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

"""
### CPU

Example using a LLaMA 2 7B model
"""
logger.info("### CPU")

llm = LlamaCpp(
    model_path="/Users/rlm/Desktop/Code/llama.cpp/models/openorca-platypus2-13b.gguf.q4_0.bin",
    temperature=0.75,
    max_tokens=2000,
    top_p=1,
    callback_manager=callback_manager,
    verbose=True,  # Verbose is required to pass to the callback manager
)

question = """
Question: A rap battle between Stephen Colbert and John Oliver
"""
llm.invoke(question)

"""
Example using a LLaMA v1 model
"""
logger.info("Example using a LLaMA v1 model")

llm = LlamaCpp(
    model_path="./ggml-model-q4_0.bin", callback_manager=callback_manager, verbose=True
)

llm_chain = prompt | llm

question = "What NFL team won the Super Bowl in the year Justin Bieber was born?"
llm_chain.invoke({"question": question})

"""
### GPU

If the installation with BLAS backend was correct, you will see a `BLAS = 1` indicator in model properties.

Two of the most important parameters for use with GPU are:

- `n_gpu_layers` - determines how many layers of the model are offloaded to your GPU.
- `n_batch` - how many tokens are processed in parallel. 

Setting these parameters correctly will dramatically improve the evaluation speed (see [wrapper code](https://github.com/langchain-ai/langchain/blob/master/libs/community/langchain_community/llms/llamacpp.py) for more details).
"""
logger.info("### GPU")

n_gpu_layers = -1  # The number of layers to put on the GPU. The rest will be on the CPU. If you don't know how many layers there are, you can use -1 to move all to GPU.
n_batch = 512  # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.

llm = LlamaCpp(
    model_path="/Users/rlm/Desktop/Code/llama.cpp/models/openorca-platypus2-13b.gguf.q4_0.bin",
    n_gpu_layers=n_gpu_layers,
    n_batch=n_batch,
    callback_manager=callback_manager,
    verbose=True,  # Verbose is required to pass to the callback manager
)

llm_chain = prompt | llm
question = "What NFL team won the Super Bowl in the year Justin Bieber was born?"
llm_chain.invoke({"question": question})

"""
### Metal

If the installation with Metal was correct, you will see a `NEON = 1` indicator in model properties.

Two of the most important GPU parameters are:

- `n_gpu_layers` - determines how many layers of the model are offloaded to your Metal GPU.
- `n_batch` - how many tokens are processed in parallel, default is 8, set to bigger number.
- `f16_kv` - for some reason, Metal only support `True`, otherwise you will get error such as `Asserting on type 0
GGML_ASSERT: .../ggml-metal.m:706: false && "not implemented"`

Setting these parameters correctly will dramatically improve the evaluation speed (see [wrapper code](https://github.com/langchain-ai/langchain/blob/master/libs/community/langchain_community/llms/llamacpp.py) for more details).
"""
logger.info("### Metal")

n_gpu_layers = 1  # The number of layers to put on the GPU. The rest will be on the CPU. If you don't know how many layers there are, you can use -1 to move all to GPU.
n_batch = 512  # Should be between 1 and n_ctx, consider the amount of RAM of your Apple Silicon Chip.
llm = LlamaCpp(
    model_path="/Users/rlm/Desktop/Code/llama.cpp/models/openorca-platypus2-13b.gguf.q4_0.bin",
    n_gpu_layers=n_gpu_layers,
    n_batch=n_batch,
    f16_kv=True,  # MUST set to True, otherwise you will run into problem after a couple of calls
    callback_manager=callback_manager,
    verbose=True,  # Verbose is required to pass to the callback manager
)

"""
The console log will show the following log to indicate Metal was enable properly.

```
ggml_metal_init: allocating
ggml_metal_init: using MPS
...
```

You also could check `Activity Monitor` by watching the GPU usage of the process, the CPU usage will drop dramatically after turn on `n_gpu_layers=1`. 

For the first call to the LLM, the performance may be slow due to the model compilation in Metal GPU.

### Grammars

We can use [grammars](https://github.com/ggerganov/llama.cpp/blob/master/grammars/README.md) to constrain model outputs and sample tokens based on the rules defined in them.

To demonstrate this concept, we've included [sample grammar files](https://github.com/langchain-ai/langchain/tree/master/libs/langchain/langchain/llms/grammars), that will be used in the examples below.

Creating gbnf grammar files can be time-consuming, but if you have a use-case where output schemas are important, there are two tools that can help:
- [Online grammar generator app](https://grammar.intrinsiclabs.ai/) that converts TypeScript interface definitions to gbnf file.
- [Python script](https://github.com/ggerganov/llama.cpp/blob/master/examples/json-schema-to-grammar.py) for converting json schema to gbnf file. You can for example create `pydantic` object, generate its JSON schema using `.schema_json()` method, and then use this script to convert it to gbnf file.

In the first example, supply the path to the specified `json.gbnf` file in order to produce JSON:
"""
logger.info("### Grammars")

n_gpu_layers = 1  # The number of layers to put on the GPU. The rest will be on the CPU. If you don't know how many layers there are, you can use -1 to move all to GPU.
n_batch = 512  # Should be between 1 and n_ctx, consider the amount of RAM of your Apple Silicon Chip.
llm = LlamaCpp(
    model_path="/Users/rlm/Desktop/Code/llama.cpp/models/openorca-platypus2-13b.gguf.q4_0.bin",
    n_gpu_layers=n_gpu_layers,
    n_batch=n_batch,
    f16_kv=True,  # MUST set to True, otherwise you will run into problem after a couple of calls
    callback_manager=callback_manager,
    verbose=True,  # Verbose is required to pass to the callback manager
    grammar_path="/Users/rlm/Desktop/Code/langchain-main/langchain/libs/langchain/langchain/llms/grammars/json.gbnf",
)

# %%capture captured --no-stdout
result = llm.invoke("Describe a person in JSON format:")

"""
We can also supply `list.gbnf` to return a list:
"""
logger.info("We can also supply `list.gbnf` to return a list:")

n_gpu_layers = 1
n_batch = 512
llm = LlamaCpp(
    model_path="/Users/rlm/Desktop/Code/llama.cpp/models/openorca-platypus2-13b.gguf.q4_0.bin",
    n_gpu_layers=n_gpu_layers,
    n_batch=n_batch,
    f16_kv=True,  # MUST set to True, otherwise you will run into problem after a couple of calls
    callback_manager=callback_manager,
    verbose=True,
    grammar_path="/Users/rlm/Desktop/Code/langchain-main/langchain/libs/langchain/langchain/llms/grammars/list.gbnf",
)

# %%capture captured --no-stdout
result = llm.invoke("List of top-3 my favourite books:")

logger.info("\n\n[DONE]", bright=True)