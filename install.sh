#!/bin/bash

# Install vLLM with CUDA 11.8.
export VLLM_VERSION=0.2.7
export PYTHON_VERSION=39
pip install https://github.com/vllm-project/vllm/releases/download/v${VLLM_VERSION}/vllm-${VLLM_VERSION}-cp${PYTHON_VERSION}-cp${PYTHON_VERSION}-manylinux1_x86_64.whl -i https://mirrors.aliyun.com/pypi/simple/

pip install langchain modelscope tiktoken requests -i https://mirrors.aliyun.com/pypi/simple/