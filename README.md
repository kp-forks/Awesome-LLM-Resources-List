# 🌟 Awesome LLM Resources

A Curated Collection of LLM resources. 💡✨

**🌐 Updated: 22nd of June 2025**

### 'Serverless' Hosting of Private/OS Models

| Platform/Tool                   | Rel. | Scale Down | OS 🔓 | GH | Start | One-Click | Dev Exp. | Free-Tier |
| --------------------------------| -------- | -------------| -------------- | ----------|----------------|---------------|-------------------------|--------------|
| [Beam.Cloud](https://www.beam.cloud/) | 2021     | > 1 min      | [![GitHub Repo stars](https://img.shields.io/github/stars/beam-cloud/beta9?style=flat-square&color=purple)](https://github.com/beam-cloud/beta9) | [![GitHub followers](https://img.shields.io/github/followers/beam-cloud?style=flat-square&color=teal)](https://github.com/beam-cloud) | [Helpers](https://github.com/ilsilfverskiold/Awesome-LLM-Resources-List/tree/main/helpers/scripts/beam) | ❌ | 👍 | 🆓 15h |
| [Baseten](https://www.baseten.com/) | 2019     | > 15 min     | 🔴            | [![GitHub followers](https://img.shields.io/github/followers/basetenlabs?style=flat-square&color=teal)](https://github.com/basetenlabs) | [Guide](https://docs.baseten.co/deploy/guides/private-model) | 🟡 | 👍 | $30 |
| [Modal](https://modal.com/)      | 2021     | < 1 min      | 🔴            | [![GitHub followers](https://img.shields.io/github/followers/modal-labs?style=flat-square&color=teal)](https://github.com/modal-labs) | [Helpers](https://github.com/ilsilfverskiold/Awesome-LLM-Resources-List/tree/main/helpers/scripts/modal) | ❌ | 👍 | $30/m |
| [HF Endpoints](https://ui.endpoints.huggingface.co/) | 2023 | > 15 min     | 🔴            | [![GitHub followers](https://img.shields.io/github/followers/huggingface?style=flat-square&color=teal)](https://github.com/huggingface) | None Needed             | ✅ | 😓 | ❌ |
| [Replicate](https://replicate.com/) | 2019     | < 1 min      | 🔴            | [![GitHub followers](https://img.shields.io/github/followers/replicate?style=flat-square&color=teal)](https://github.com/replicate) | [Guide](https://replicate.com/docs/guides/push-a-transformers-model) | 🟡 | 🤷 | ❌ |
| [Sagemaker (Serverless)](https://aws.amazon.com/sagemaker/) | 2017 | N/A          | 🔴            | [![GitHub followers](https://img.shields.io/github/followers/aws?style=flat-square&color=teal)](https://github.com/aws/amazon-sagemaker-examples) | N/A                     | ❌ | ❌ | 300,000s |
| [Lambda w/ EFS (AWS)](https://aws.amazon.com/pm/lambda/) | 2014 | < 1 min     | 🔴            | [![GitHub followers](https://img.shields.io/github/followers/awsdocs?style=flat-square&color=teal)](https://github.com/awsdocs/aws-lambda-developer-guide) | [Guide](https://aws.amazon.com/blogs/compute/hosting-hugging-face-models-on-aws-lambda/) | ❌ | ❌ | ✅ |
| [RunPod Serverless](https://www.runpod.io/serverless-gpu) | 2022 | > 30s       | 🔴            | [![GitHub followers](https://img.shields.io/github/followers/runpod?style=flat-square&color=teal)](https://github.com/runpod) | N/A                     | ❌ | 🤷 | ❌ |
| [BentoML](https://www.bentoml.com/) | 2019     | > 5 min      | [![GitHub Repo stars](https://img.shields.io/github/stars/bentoml/BentoML?style=flat-square&color=purple)](https://github.com/bentoml/BentoML) | [![GitHub followers](https://img.shields.io/github/followers/bentoml?style=flat-square&color=teal)](https://github.com/bentoml) | [Gallery](https://www.bentoml.com/gallery) | 🟡 | 👍 | 🆓 $10 |

It goes without saying that these platforms can usually do more than LLM serving**

### 🧮 Serverless Compute Pricing & Limits – Lambda vs Modal (on CPU)

| Platform               | 💵 Compute Unit                                     | 📥 Per-Request Fee                                 | 🆓 Free Tier                                         | ⏱️ Max Timeout                           | 🚦 Concurrency Limit                                              |
|------------------------|----------------------------------------------------|----------------------------------------------------|------------------------------------------------------|------------------------------------------|-------------------------------------------------------------------|
| **AWS Lambda + API GW**| GB-sec @ $0.000016667                              | $0.20/M Lambda + $1.00/M HTTP API calls            | 1M req + 400k GB-s/mo (12 mo) + 1M API calls/mo     | 15 min                                    | 1,000 per region (can request more)                               |
| **Modal**              | CPU-s @ $0.0000131 + GiB-s @ $0.00000222           | ❌ No per-request fee                              | $30/mo compute credits (Starter)                   | Func: 24h ⎮ HTTP: 150s → 303 redirect    | Starter: 100 containers / 200 req/s ⎮ Team: 1,000 containers      |


### Access Off-the-Shelf OS Models (via API):

| Platform/Tool                           | Released | GitHub |
| --------------------------------------- | -------- | ----------- |
| [Together.ai](https://Together.ai)                                | N/A      | 🔴          |
| [Fireworks.ai](https://Fireworks.ai)                            | N/A      | [![GitHub followers](https://img.shields.io/github/followers/fw-ai?style=flat-square&color=teal)](https://github.com/fw-ai) |
| [Replicate](https://replicate.com/)     | 2019     | [![GitHub followers](https://img.shields.io/github/followers/replicate?style=flat-square&color=teal)](https://github.com/replicate) |
| [Groq](https://groq.com/)               | N/A      | [![GitHub followers](https://img.shields.io/github/followers/groq?style=flat-square&color=teal)](https://github.com/groq) |
| [DeepInfra](https://deepinfra.com/)     | N/A      | [![GitHub followers](https://img.shields.io/github/followers/deepinfra?style=flat-square&color=teal)](https://github.com/deepinfra) |
| [Bedrock](https://aws.amazon.com/bedrock) | N/A    | [![GitHub followers](https://img.shields.io/github/followers/aws-samples?style=flat-square&color=teal)](https://github.com/aws-samples/amazon-bedrock-workshop) |
| [Lepton](https://www.lepton.ai/)        | N/A      | [![GitHub followers](https://img.shields.io/github/followers/leptonai?style=flat-square&color=teal)](https://github.com/leptonai) |
| [Fal.ai](https://fal.ai/)               | N/A      | [![GitHub followers](https://img.shields.io/github/followers/fal-ai?style=flat-square&color=teal)](https://github.com/fal-ai) |
| [VertexAI](https://cloud.google.com/vertex-ai) | N/A | [![GitHub followers](https://img.shields.io/github/followers/GoogleCloudPlatform?style=flat-square&color=teal)](https://github.com/GoogleCloudPlatform/vertex-ai-samples) |

### Local Inference 

| Framework                                         | Browser Chat 🖥️ | Organization      | Open Source | GitHub |
|-----------------------------------------------|------------------|-------------------|-------------|--------|
| [Llama.cpp](https://github.com/ggerganov/llama.cpp) | ❌               | ggerganov        | [![GitHub Repo stars](https://img.shields.io/github/stars/ggerganov/llama.cpp?style=flat-square&color=purple)](https://github.com/ggerganov/llama.cpp) | [![GitHub followers](https://img.shields.io/github/followers/ggerganov?style=flat-square&color=teal)](https://github.com/ggerganov) |
| [Ollama](https://ollama.com/)                  | ❌               | Ollama           | [![GitHub Repo stars](https://img.shields.io/github/stars/ollama/ollama?style=flat-square&color=purple)](https://github.com/ollama/ollama) | [![GitHub followers](https://img.shields.io/github/followers/ollama?style=flat-square&color=teal)](https://github.com/ollama) |
| [gpt4all](https://www.nomic.ai/gpt4all)        | ✅               | Nomic.ai         | [![GitHub Repo stars](https://img.shields.io/github/stars/nomic-ai/gpt4all?style=flat-square&color=purple)](https://github.com/nomic-ai/gpt4all) | [![GitHub followers](https://img.shields.io/github/followers/nomic-ai?style=flat-square&color=teal)](https://github.com/nomic-ai) |
| [LMStudio](https://lmstudio.ai/)               | ✅               | LMStudio AI      | 🔴 | [![GitHub followers](https://img.shields.io/github/followers/lmstudio-ai?style=flat-square&color=teal)](https://github.com/lmstudio-ai) |
| [OpenLLM](https://github.com/bentoml/OpenLLM)  | ✅               | BentoML          | [![GitHub Repo stars](https://img.shields.io/github/stars/bentoml/OpenLLM?style=flat-square&color=purple)](https://github.com/bentoml/OpenLLM) | [![GitHub followers](https://img.shields.io/github/followers/bentoml?style=flat-square&color=teal)](https://github.com/bentoml) |

### LLM Serving Frameworks

| Framework                                                      | Open Source                                                                                                                | GitHub                                                                                   |
|----------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------|
| [vLLM](https://github.com/vllm-project/vllm)                   | [![GitHub Repo stars](https://img.shields.io/github/stars/vllm-project/vllm?style=flat-square&color=purple)](https://github.com/vllm-project/vllm)                | [![GitHub followers](https://img.shields.io/github/followers/vllm-project?style=flat-square&color=teal)](https://github.com/vllm-project) |
| [OpenLLM](https://github.com/bentoml/OpenLLM)                  | [![GitHub Repo stars](https://img.shields.io/github/stars/bentoml/OpenLLM?style=flat-square&color=purple)](https://github.com/bentoml/OpenLLM)                    | [![GitHub followers](https://img.shields.io/github/followers/bentoml?style=flat-square&color=teal)](https://github.com/bentoml) |
| [TGI (Text Generation Inference)](https://github.com/huggingface/text-generation-inference) | [![GitHub Repo stars](https://img.shields.io/github/stars/huggingface/text-generation-inference?style=flat-square&color=purple)](https://github.com/huggingface/text-generation-inference) | [![GitHub followers](https://img.shields.io/github/followers/huggingface?style=flat-square&color=teal)](https://github.com/huggingface) |
| [TensorRT LLM](https://docs.nvidia.com/tensorrt-llm/index.html) | [![GitHub Repo stars](https://img.shields.io/github/stars/NVIDIA/TensorRT-LLM?style=flat-square&color=purple)](https://github.com/NVIDIA/TensorRT-LLM)            | [![GitHub followers](https://img.shields.io/github/followers/NVIDIA?style=flat-square&color=teal)](https://github.com/NVIDIA) |
| [Ray Serve](https://docs.ray.io/en/latest/serve/index.html)    | [![GitHub Repo stars](https://img.shields.io/github/stars/ray-project/ray?style=flat-square&color=purple)](https://github.com/ray-project/ray)                    | [![GitHub followers](https://img.shields.io/github/followers/ray-project?style=flat-square&color=teal)](https://github.com/ray-project) |
| [LMDeploy](https://github.com/InternLM/lmdeploy)               | [![GitHub Repo stars](https://img.shields.io/github/stars/InternLM/lmdeploy?style=flat-square&color=purple)](https://github.com/InternLM/lmdeploy)                | [![GitHub followers](https://img.shields.io/github/followers/InternLM?style=flat-square&color=teal)](https://github.com/InternLM) |
| [Ollama](https://github.com/ollama/ollama)                     | [![GitHub Repo stars](https://img.shields.io/github/stars/ollama/ollama?style=flat-square&color=purple)](https://github.com/ollama/ollama)                        | [![GitHub followers](https://img.shields.io/github/followers/ollama?style=flat-square&color=teal)](https://github.com/ollama) |
| [MLC-LLM](https://github.com/mlc-ai/mlc-llm)                   | [![GitHub Repo stars](https://img.shields.io/github/stars/mlc-ai/mlc-llm?style=flat-square&color=purple)](https://github.com/mlc-ai/mlc-llm)                      | [![GitHub followers](https://img.shields.io/github/followers/mlc-ai?style=flat-square&color=teal)](https://github.com/mlc-ai) |


### Building Open-Source LLM Web Chat UIs

| Tool                                            | Organization       | Description                                                                                     | Open Source                                                                                                                | GitHub                                                                                   |
|-------------------------------------------------|--------------------|-------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------|
| [Text Generation WebUI](https://github.com/oobabooga/text-generation-webui) | oobabooga          | A Gradio web UI for Large Language Models.                                                      | [![GitHub Repo stars](https://img.shields.io/github/stars/oobabooga/text-generation-webui?style=flat-square&color=purple)](https://github.com/oobabooga/text-generation-webui) | [![GitHub followers](https://img.shields.io/github/followers/oobabooga?style=flat-square&color=teal)](https://github.com/oobabooga) |
| [Jan AI](https://jan.ai/)                       | Jan HQ             | An open source alternative to ChatGPT that runs 100% offline on your computer. Multiple engine support (llama.cpp, TensorRT-LLM). | [![GitHub Repo stars](https://img.shields.io/github/stars/janhq/jan?style=flat-square&color=purple)](https://github.com/janhq/jan) | [![GitHub followers](https://img.shields.io/github/followers/janhq?style=flat-square&color=teal)](https://github.com/janhq) |
| [AnythingLLM](https://github.com/Mintplex-Labs/anything-llm) | Mintplex Labs      | The all-in-one Desktop & Docker AI application with built-in RAG, AI agents, and more.            | [![GitHub Repo stars](https://img.shields.io/github/stars/Mintplex-Labs/anything-llm?style=flat-square&color=purple)](https://github.com/Mintplex-Labs/anything-llm) | [![GitHub followers](https://img.shields.io/github/followers/Mintplex-Labs?style=flat-square&color=teal)](https://github.com/Mintplex-Labs) |
| [Superagent](https://github.com/superagent-ai/superagent) | Superagent AI      | Allows developers to add powerful AI assistants to their applications using LLMs and RAG.          | [![GitHub Repo stars](https://img.shields.io/github/stars/superagent-ai/superagent?style=flat-square&color=purple)](https://github.com/superagent-ai/superagent) | [![GitHub followers](https://img.shields.io/github/followers/superagent-ai?style=flat-square&color=teal)](https://github.com/superagent-ai) |
| [Bionic-GPT](https://bionic-gpt.com/)           | Bionic GPT         | A ChatGPT replacement offering generative AI advantages while maintaining strict data confidentiality. | [![GitHub Repo stars](https://img.shields.io/github/stars/bionic-gpt/bionic-gpt?style=flat-square&color=purple)](https://github.com/bionic-gpt/bionic-gpt) | [![GitHub followers](https://img.shields.io/github/followers/bionic-gpt?style=flat-square&color=teal)](https://github.com/bionic-gpt) |
| [Open WebUI](https://github.com/open-webui/open-webui) | Open WebUI         | A user-friendly web interface for interacting with Large Language Models (LLMs).                   | [![GitHub Repo stars](https://img.shields.io/github/stars/open-webui/open-webui?style=flat-square&color=purple)](https://github.com/open-webui/open-webui) | [![GitHub followers](https://img.shields.io/github/followers/open-webui?style=flat-square&color=teal)](https://github.com/open-webui) |
| [Xyne](https://github.com/xynehq/xyne)            | xynehq             | A sleek, minimal web chat interface for interacting with Large Language Models.                 | [![GitHub Repo stars](https://img.shields.io/github/stars/xynehq/xyne?style=flat-square&color=purple)](https://github.com/xynehq/xyne)            | [![GitHub followers](https://img.shields.io/github/followers/xynehq?style=flat-square&color=teal)](https://github.com/xynehq)            |
| [Assistant UI](https://github.com/assistant-ui/assistant-ui) | assistant-ui       | An open-source ChatGPT-like interface with a clean and responsive design.                        | [![GitHub Repo stars](https://img.shields.io/github/stars/assistant-ui/assistant-ui?style=flat-square&color=purple)](https://github.com/assistant-ui/assistant-ui) | [![GitHub followers](https://img.shields.io/github/followers/assistant-ui?style=flat-square&color=teal)](https://github.com/assistant-ui) |
| [Scira](https://github.com/zaidmukaddam/scira)    | zaidmukaddam       | An AI-powered search interface that leverages LLMs for intelligent search results.               | [![GitHub Repo stars](https://img.shields.io/github/stars/zaidmukaddam/scira?style=flat-square&color=purple)](https://github.com/zaidmukaddam/scira)    | [![GitHub followers](https://img.shields.io/github/followers/zaidmukaddam?style=flat-square&color=teal)](https://github.com/zaidmukaddam)    |
| [Onyx](https://github.com/onyx-dot-app/onyx)      | onyx-dot-app       | A customizable and extendable web chat UI for interacting with large language models.            | [![GitHub Repo stars](https://img.shields.io/github/stars/onyx-dot-app/onyx?style=flat-square&color=purple)](https://github.com/onyx-dot-app/onyx)      | [![GitHub followers](https://img.shields.io/github/followers/onyx-dot-app?style=flat-square&color=teal)](https://github.com/onyx-dot-app)      |
| [NextChat](https://github.com/ChatGPTNextWeb/NextChat) | ChatGPTNextWeb     | A Next.js-based, open-source ChatGPT clone for seamless web interaction.                         | [![GitHub Repo stars](https://img.shields.io/github/stars/ChatGPTNextWeb/NextChat?style=flat-square&color=purple)](https://github.com/ChatGPTNextWeb/NextChat) | [![GitHub followers](https://img.shields.io/github/followers/ChatGPTNextWeb?style=flat-square&color=teal)](https://github.com/ChatGPTNextWeb) |

### Rent GPUs (Fine-Tuning, Deploying, Training)

| Platform                                      | Templates                | Beginner Friendly | GitHub |
|-----------------------------------------------|--------------------------|-------------------|--------|
| [Brev.dev](https://www.brev.dev/)             | Fine-tuning              | ❌                | [![GitHub followers](https://img.shields.io/github/followers/brevdev?style=flat-square&color=teal)](https://github.com/brevdev) |
| [Modal](https://modal.com/)                   | Fine-tuning              | ❌                | [![GitHub followers](https://img.shields.io/github/followers/modal-labs?style=flat-square&color=teal)](https://github.com/modal-labs) |
| [Hyperbolic AI](https://hyperbolic.xyz/)      | None                     | ❌                | [![GitHub followers](https://img.shields.io/github/followers/HyperbolicLabs?style=flat-square&color=teal)](https://github.com/HyperbolicLabs) |
| [RunPod](https://www.runpod.io/)              | None                     | ❌                | [![GitHub followers](https://img.shields.io/github/followers/runpod?style=flat-square&color=teal)](https://github.com/runpod) |
| [Paperspace](https://www.paperspace.com/)     | Fine-tuning              | ✅                | [![GitHub followers](https://img.shields.io/github/followers/Paperspace?style=flat-square&color=teal)](https://github.com/Paperspace) |
| [Colab](https://colab.research.google.com/)   | Small models only        | ✅                | [![GitHub followers](https://img.shields.io/github/followers/googlecolab?style=flat-square&color=teal)](https://github.com/googlecolab) |


### Fine-Tuning with No-Code UI

| Tool                                           | Beginner Friendly | Open Source | GitHub |
|------------------------------------------------|-------------------|-------------|--------|
| [Together.ai](https://www.together.ai/products#fine-tuning) | ✅               | ❌          | N/A    |
| [Hugging Face AutoTrain](https://huggingface.co/autotrain) | ✅               | ❌          | [![GitHub Repo stars](https://img.shields.io/github/stars/huggingface/autotrain-advanced?style=flat-square&color=purple)](https://github.com/huggingface/autotrain-advanced) |
| [AutoML](https://github.com/mljar/automl-app)  | ❌               | ✅          | [![GitHub Repo stars](https://img.shields.io/github/stars/mljar/automl-app?style=flat-square&color=purple)](https://github.com/mljar/automl-app) |
| [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) | ❌               | ✅          | [![GitHub Repo stars](https://img.shields.io/github/stars/hiyouga/LLaMA-Factory?style=flat-square&color=purple)](https://github.com/hiyouga/LLaMA-Factory) |
| [H2O LLM Studio](https://github.com/h2oai/h2o-llmstudio) | ✅               | ✅          | [![GitHub Repo stars](https://img.shields.io/github/stars/h2oai/h2o-llmstudio?style=flat-square&color=purple)](https://github.com/h2oai/h2o-llmstudio) |

### Fine-Tuning Frameworks

| Framework                                 | Open Source | GitHub |
|-------------------------------------------|-------------|--------|
| [Axolotl](https://axolotl.ai/)            | [![GitHub Repo stars](https://img.shields.io/github/stars/axolotl-ai-cloud/axolotl?style=flat-square&color=purple)](https://github.com/axolotl-ai-cloud/axolotl) | [![GitHub followers](https://img.shields.io/github/followers/axolotl-ai-cloud?style=flat-square&color=teal)](https://github.com/axolotl-ai-cloud) |
| [Unsloth](https://unsloth.ai/)            | [![GitHub Repo stars](https://img.shields.io/github/stars/unslothai/unsloth?style=flat-square&color=purple)](https://github.com/unslothai/unsloth) | [![GitHub followers](https://img.shields.io/github/followers/unslothai?style=flat-square&color=teal)](https://github.com/unslothai) |


### OS Agentic/AI Workflow

| Framework                                        | Open Source                                                                                                                | Beginner Friendly | Released | GitHub                                                                                   |
|--------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------|-------------------|----------|------------------------------------------------------------------------------------------|
| [LangChain](https://www.langchain.com/)          | [![GitHub Repo stars](https://img.shields.io/github/stars/langchain-ai/langchain?style=flat-square&color=purple)](https://github.com/langchain-ai/langchain) | ✅               | 2022     | [![GitHub followers](https://img.shields.io/github/followers/langchain-ai?style=flat-square&color=teal)](https://github.com/langchain-ai) |
| [LlamaIndex](https://docs.llamaindex.ai/en/stable/use_cases/agents/) | [![GitHub Repo stars](https://img.shields.io/github/stars/run-llama/llama_index?style=flat-square&color=purple)](https://github.com/run-llama/llama_index) | ❌               | 2023     | [![GitHub followers](https://img.shields.io/github/followers/run-llama?style=flat-square&color=teal)](https://github.com/run-llama) |
| [Swarms](https://swarms.world/)                  | [![GitHub Repo stars](https://img.shields.io/github/stars/kyegomez/swarms?style=flat-square&color=purple)](https://github.com/kyegomez/swarms) | ❌               | 2023     | [![GitHub followers](https://img.shields.io/github/followers/kyegomez?style=flat-square&color=teal)](https://github.com/kyegomez) |
| [CrewAI](https://www.crewai.com/)               | [![GitHub Repo stars](https://img.shields.io/github/stars/crewaiinc/crewai?style=flat-square&color=purple)](https://github.com/crewaiinc/crewai) | ✅               | 2023     | [![GitHub followers](https://img.shields.io/github/followers/crewaiinc?style=flat-square&color=teal)](https://github.com/crewaiinc) |
| [Autogen](https://microsoft.github.io/autogen/0.2/) | [![GitHub Repo stars](https://img.shields.io/github/stars/microsoft/autogen?style=flat-square&color=purple)](https://github.com/microsoft/autogen) | ✅               | 2023     | [![GitHub followers](https://img.shields.io/github/followers/microsoft?style=flat-square&color=teal)](https://github.com/microsoft) |
| [AutoChain](https://autochain.forethought.ai/)   | [![GitHub Repo stars](https://img.shields.io/github/stars/Forethought-Technologies/AutoChain?style=flat-square&color=purple)](https://github.com/Forethought-Technologies/AutoChain) | ❌               | 2023     | [![GitHub followers](https://img.shields.io/github/followers/Forethought-Technologies?style=flat-square&color=teal)](https://github.com/Forethought-Technologies) |
| [SuperAGI](https://superagi.com/)                | [![GitHub Repo stars](https://img.shields.io/github/stars/TransformerOptimus/SuperAGI?style=flat-square&color=purple)](https://github.com/TransformerOptimus/SuperAGI) | ❌               | 2023     | [![GitHub followers](https://img.shields.io/github/followers/TransformerOptimus?style=flat-square&color=teal)](https://github.com/TransformerOptimus) |
| [AILegion](https://github.com/eumemic/ai-legion)  | [![GitHub Repo stars](https://img.shields.io/github/stars/eumemic/ai-legion?style=flat-square&color=purple)](https://github.com/eumemic/ai-legion) | ❌               | 2023     | [![GitHub followers](https://img.shields.io/github/followers/eumemic?style=flat-square&color=teal)](https://github.com/eumemic) |
| [MemGPT (Letta)](https://www.letta.com/)         | [![GitHub Repo stars](https://img.shields.io/github/stars/cpacker/MemGPT?style=flat-square&color=purple)](https://github.com/cpacker/MemGPT) | ❌               | 2023     | [![GitHub followers](https://img.shields.io/github/followers/cpacker?style=flat-square&color=teal)](https://github.com/cpacker) |
| [uAgents](https://pypi.org/project/uagents/)     | [![GitHub Repo stars](https://img.shields.io/github/stars/fetchai/uAgents?style=flat-square&color=purple)](https://github.com/fetchai/uAgents) | ❌               | 2023     | [![GitHub followers](https://img.shields.io/github/followers/fetchai?style=flat-square&color=teal)](https://github.com/fetchai) |
| [AGiXT](https://github.com/Josh-XT/AGiXT)          | [![GitHub Repo stars](https://img.shields.io/github/stars/Josh-XT/AGiXT?style=flat-square&color=purple)](https://github.com/Josh-XT/AGiXT) | ❌               | 2023     | [![GitHub followers](https://img.shields.io/github/followers/Josh-XT?style=flat-square&color=teal)](https://github.com/Josh-XT) |
| [Dify](https://dify.ai/)                         | [![GitHub Repo stars](https://img.shields.io/github/stars/langgenius/dify?style=flat-square&color=purple)](https://github.com/langgenius/dify) | ✅               | 2024     | [![GitHub followers](https://img.shields.io/github/followers/langgenius?style=flat-square&color=teal)](https://github.com/langgenius) |
| [TaskingAI](https://www.tasking.ai/)             | [![GitHub Repo stars](https://img.shields.io/github/stars/TaskingAI/TaskingAI?style=flat-square&color=purple)](https://github.com/TaskingAI/TaskingAI) | ✅               | 2024     | [![GitHub followers](https://img.shields.io/github/followers/TaskingAI?style=flat-square&color=teal)](https://github.com/TaskingAI) |
| [Bee Agent Framework](https://i-am-bee.github.io/bee-agent-framework/#/) | [![GitHub Repo stars](https://img.shields.io/github/stars/i-am-bee/bee-agent-framework?style=flat-square&color=purple)](https://github.com/i-am-bee/bee-agent-framework) | ❌               | 2024     | [![GitHub followers](https://img.shields.io/github/followers/i-am-bee?style=flat-square&color=teal)](https://github.com/i-am-bee) |
| [Swarms](https://swarms.world/)                  | [![GitHub Repo stars](https://img.shields.io/github/stars/kyegomez/swarms?style=flat-square&color=purple)](https://github.com/kyegomez/swarms) | ❌               | 2024     | [![GitHub followers](https://img.shields.io/github/followers/kyegomez?style=flat-square&color=teal)](https://github.com/kyegomez) |
| [IoA](https://openbmb.github.io/IoA/)             | [![GitHub Repo stars](https://img.shields.io/github/stars/OpenBMB/IoA?style=flat-square&color=purple)](https://github.com/OpenBMB/IoA) | ❌               | 2024     | [![GitHub followers](https://img.shields.io/github/followers/OpenBMB?style=flat-square&color=teal)](https://github.com/OpenBMB) |
| [Upsonic](https://github.com/Upsonic/Upsonic)      | [![GitHub Repo stars](https://img.shields.io/github/stars/Upsonic/Upsonic?style=flat-square&color=purple)](https://github.com/Upsonic/Upsonic) | ❌               | 2024     | [![GitHub followers](https://img.shields.io/github/followers/Upsonic?style=flat-square&color=teal)](https://github.com/Upsonic) |
| [Parlant](https://github.com/emcie-co/parlant)     | [![GitHub Repo stars](https://img.shields.io/github/stars/emcie-co/parlant?style=flat-square&color=purple)](https://github.com/emcie-co/parlant) | ❌               | 2024     | [![GitHub followers](https://img.shields.io/github/followers/emcie-co?style=flat-square&color=teal)](https://github.com/emcie-co) |
| [Rig](https://github.com/0xPlaygrounds/rig)        | [![GitHub Repo stars](https://img.shields.io/github/stars/0xPlaygrounds/rig?style=flat-square&color=purple)](https://github.com/0xPlaygrounds/rig) | ❌               | 2024     | [![GitHub followers](https://img.shields.io/github/followers/0xPlaygrounds?style=flat-square&color=teal)](https://github.com/0xPlaygrounds) |
| [eliza](https://github.com/elizaOS/eliza) | [![GitHub Repo stars](https://img.shields.io/github/stars/elizaOS/eliza?style=flat-square&color=purple)](https://github.com/elizaOS/eliza) | ✅ | 2024 | [![GitHub followers](https://img.shields.io/github/followers/elizaOS?style=flat-square&color=teal)](https://github.com/elizaOS) |
| [TensorZero](https://www.tensorzero.com/) | [![GitHub Repo stars](https://img.shields.io/github/stars/tensorzero/tensorzero?style=flat-square&color=purple)](https://github.com/tensorzero/tensorzero) | ❌ | 2024 | [![GitHub followers](https://img.shields.io/github/followers/tensorzero?style=flat-square&color=teal)](https://github.com/tensorzero/tensorzero) |
| [AgentDock](https://agentdock.ai/) | [![GitHub Repo stars](https://img.shields.io/github/stars/AgentDock/AgentDock?style=flat-square&color=purple)](https://github.com/AgentDock/AgentDock) | ✅ | 2025 | [![GitHub followers](https://img.shields.io/github/followers/AgentDock?style=flat-square&color=teal)](https://github.com/AgentDock) |

### Top Agentic Frameworks

| Framework                                                 | Open Source                                                                                                                | Beginner Friendly | Released | GitHub                                                                                   |
|-----------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------|-------------------|----------|------------------------------------------------------------------------------------------|
| [LangGraph](https://www.langchain.com/langgraph)          | [![GitHub Repo stars](https://img.shields.io/github/stars/langchain-ai/langgraph?style=flat-square&color=purple)](https://github.com/langchain-ai/langgraph) | ❌               | 2023     | [![GitHub followers](https://img.shields.io/github/followers/langchain-ai?style=flat-square&color=teal)](https://github.com/langchain-ai) |
| [Flowise](https://flowiseai.com/)                         | [![GitHub Repo stars](https://img.shields.io/github/stars/FlowiseAI/Flowise?style=flat-square&color=purple)](https://github.com/FlowiseAI/Flowise) | ✅               | 2023     | [![GitHub followers](https://img.shields.io/github/followers/FlowiseAI?style=flat-square&color=teal)](https://github.com/FlowiseAI) |
| [Langroid](https://langroid.github.io/langroid/)          | [![GitHub Repo stars](https://img.shields.io/github/stars/langroid/langroid?style=flat-square&color=purple)](https://github.com/langroid/langroid) | ❌               | 2023     | [![GitHub followers](https://img.shields.io/github/followers/langroid?style=flat-square&color=teal)](https://github.com/langroid) |
| [smolagents](https://github.com/huggingface/smolagents)   | [![GitHub Repo stars](https://img.shields.io/github/stars/huggingface/smolagents?style=flat-square&color=purple)](https://github.com/huggingface/smolagents) | ❌               | 2024     | [![GitHub followers](https://img.shields.io/github/followers/huggingface?style=flat-square&color=teal)](https://github.com/huggingface) |
| [Semantic Kernel](https://github.com/microsoft/semantic-kernel) | [![GitHub Repo stars](https://img.shields.io/github/stars/microsoft/semantic-kernel?style=flat-square&color=purple)](https://github.com/microsoft/semantic-kernel) | ❌                | 2023     | [![GitHub followers](https://img.shields.io/github/followers/microsoft?style=flat-square&color=teal)](https://github.com/microsoft) |
| [Atomic Agents](https://github.com/BrainBlend-AI/atomic-agents) | [![GitHub Repo stars](https://img.shields.io/github/stars/BrainBlend-AI/atomic-agents?style=flat-square&color=purple)](https://github.com/BrainBlend-AI/atomic-agents) | ❌               | 2024     | [![GitHub followers](https://img.shields.io/github/followers/BrainBlend-AI?style=flat-square&color=teal)](https://github.com/BrainBlend-AI) |
| [Agno](https://github.com/agno-agi/agno)    | [![GitHub Repo stars](https://img.shields.io/github/stars/pydantic/pydantic-ai?style=flat-square&color=purple)](https://github.com/agno-agi/agno) | ✅                | 2024     | [![GitHub followers](https://img.shields.io/github/followers/pydantic?style=flat-square&color=teal)](https://github.com/agno-agi) |
| [PydanticAI](https://github.com/pydantic/pydantic-ai)     | [![GitHub Repo stars](https://img.shields.io/github/stars/pydantic/pydantic-ai?style=flat-square&color=purple)](https://github.com/pydantic/pydantic-ai) | ❌               | 2024     | [![GitHub followers](https://img.shields.io/github/followers/pydantic?style=flat-square&color=teal)](https://github.com/pydantic) |
| [Mastra](https://github.com/mastra-ai/mastra)             | [![GitHub Repo stars](https://img.shields.io/github/stars/mastra-ai/mastra?style=flat-square&color=purple)](https://github.com/mastra-ai/mastra) | ✅               | 2025     | [![GitHub followers](https://img.shields.io/github/followers/mastra-ai?style=flat-square&color=teal)](https://github.com/mastra-ai) |

### Agentic Frameworks: Core Capabilities
| Framework      | Memory & RAG                                  | Multimodality                                 | Multi-agent Support                           | Observability                                   |
|----------------|-----------------------------------------------|-----------------------------------------------|-----------------------------------------------|-------------------------------------------------|
| **AgentDock**  | Built-in RAG system; knowledge base integration | 🟢 Multi-modal (text, voice, tools, APIs)    | Visual workflow orchestration & agent chains  | Comprehensive LLM traceability & credit tracking |
| **Agno**       | Integrated memory & vector DB/RAG             | 🟢 Native (text, image, audio, video)         | Supervisor-worker roles                       | Built-in cloud dashboard/logging                |
| **LangGraph**  | Persistent state; easy external integration   | 🔸 Primarily text; extendable via nodes       | Hierarchical orchestration                    | LangSmith integration & graph editor            |
| **SmolAgents** | Built-in short-term; custom long-term         | 🔸 Vision agents via VLMs                     | Modular multi-agent composition               | Minimal; relies on external logging             |
| **Mastra**     | Persistent workflows; native RAG pipelines    | 🟢 Multi-modal via integrations               | Native multi-agent workflows                  | Built-in OpenTelemetry dashboards               |
| **Pydantic AI**| DI-based memory & RAG integration             | 🔸 Text-first; multimodal via custom DI       | Type-safe manual orchestration                | Limited; Python logging/OpenTelemetry           |
| **Atomic Agents**| Per-agent memory & RAG (vector DB)            | 🟢 Native multi-modal                         | Explicit chaining of workflows                | Minimal; external instrumentation recommended   |
| **Autogen**    | Short-term built-in; external long-term       | 🔸 Text-mainly; extensible                    | Emergent, free-form collaboration             | Moderate; internal logging, no dashboard        |
| **CrewAI**     | Stateful memory & team-based RAG              | 🟢 Diverse modalities (text, image, etc.)     | Supervisor-led multi-team workflows           | Integrated dashboards for logging & monitoring  |

Please see this [google sheet](https://docs.google.com/spreadsheets/d/1zjcww1w0vARZz9Z6GDxNMp-PKyg7iRyNYAnDo59HjzI/edit?usp=sharing) with more columns. 

### Visual AI Agent Builders

| Tool | Organization | Description | Open Source | GitHub |
|------|--------------|-------------|-------------|--------|
| [Rivet](https://github.com/Ironclad/rivet) | Ironclad | A visual builder to design and deploy AI agent workflows. | [![GitHub Repo stars](https://img.shields.io/github/stars/Ironclad/rivet?style=flat-square&color=purple)](https://github.com/Ironclad/rivet) | [![GitHub followers](https://img.shields.io/github/followers/Ironclad?style=flat-square&color=teal)](https://github.com/Ironclad) |
| [PySpur](https://github.com/PySpur-Dev/pyspur) | PySpur-Dev | A tool to build and visualize AI agents seamlessly. | [![GitHub Repo stars](https://img.shields.io/github/stars/PySpur-Dev/pyspur?style=flat-square&color=purple)](https://github.com/PySpur-Dev/pyspur) | [![GitHub followers](https://img.shields.io/github/followers/PySpur-Dev?style=flat-square&color=teal)](https://github.com/PySpur-Dev) |
| [Flowise](https://github.com/FlowiseAI/Flowise) | FlowiseAI | A no‑code, visual platform for designing AI agent workflows. | [![GitHub Repo stars](https://img.shields.io/github/stars/FlowiseAI/Flowise?style=flat-square&color=purple)](https://github.com/FlowiseAI/Flowise) | [![GitHub followers](https://img.shields.io/github/followers/FlowiseAI?style=flat-square&color=teal)](https://github.com/FlowiseAI) |

### 💬 Model Call Pricing for agent systems (Text-only (2000 tokens in, 100 token out), Flat Rate)

| Model               | 💵 $ / Call | 💯 100 Calls | 🧮 1,000 Calls | 🔁 30,000 Calls |
|---------------------|------------:|-------------:|---------------:|----------------:|
| **Gemini Flash 2.0**| $0.00000    | $0.02        | $0.24          | $7.20           |
| **GPT-4o mini**     | $0.00144    | $0.14        | $1.44          | $43.20          |
| **GPT-4.1**         | $0.00480    | $0.48        | $4.80          | $144.00         |
| **Gemini Pro 2.5**  | $0.00350    | $0.35        | $3.50          | $105.00         |
| **Claude Haiku 3.5**| $0.00200    | $0.20        | $2.00          | $60.00          |
| **Claude Sonnet 4** | $0.00750    | $0.75        | $7.50          | $225.00         |
| **GPT-4o**          | $0.01200    | $1.20        | $12.00         | $360.00         |
| **OpenAI o3**       | $0.02400    | $2.40        | $24.00         | $720.00         |

### Agentic Tools (for “building”)

| Tool | Organization | Description | Open Source | GitHub |
|------|--------------|-------------|-------------|--------|
| [browser-use](https://github.com/browser-use/browser-use) | browser-use | Integrates browser functionalities into agentic workflows. | [![GitHub Repo stars](https://img.shields.io/github/stars/browser-use/browser-use?style=flat-square&color=purple)](https://github.com/browser-use/browser-use) | [![GitHub followers](https://img.shields.io/github/followers/browser-use?style=flat-square&color=teal)](https://github.com/browser-use) |
| [code2prompt](https://github.com/mufeedvh/code2prompt) | mufeedvh | Converts code snippets into actionable prompts for development. | [![GitHub Repo stars](https://img.shields.io/github/stars/mufeedvh/code2prompt?style=flat-square&color=purple)](https://github.com/mufeedvh/code2prompt) | [![GitHub followers](https://img.shields.io/github/followers/mufeedvh?style=flat-square&color=teal)](https://github.com/mufeedvh) |
| [note-gen](https://github.com/codexu/note-gen) | codexu | Automatically generates notes and documentation from your code. | [![GitHub Repo stars](https://img.shields.io/github/stars/codexu/note-gen?style=flat-square&color=purple)](https://github.com/codexu/note-gen) | [![GitHub followers](https://img.shields.io/github/followers/codexu?style=flat-square&color=teal)](https://github.com/codexu) |
| [refly](https://github.com/refly-ai/refly) | refly-ai | Automates code refactoring and prompt generation tasks. | [![GitHub Repo stars](https://img.shields.io/github/stars/refly-ai/refly?style=flat-square&color=purple)](https://github.com/refly-ai/refly) | [![GitHub followers](https://img.shields.io/github/followers/refly-ai?style=flat-square&color=teal)](https://github.com/refly-ai) |
| [potpie](https://github.com/potpie-ai/potpie) | potpie-ai | A toolkit for prototyping and building AI agent pipelines. | [![GitHub Repo stars](https://img.shields.io/github/stars/potpie-ai/potpie?style=flat-square&color=purple)](https://github.com/potpie-ai/potpie) | [![GitHub followers](https://img.shields.io/github/followers/potpie-ai?style=flat-square&color=teal)](https://github.com/potpie-ai) |
| [AgentStack](https://github.com/AgentOps-AI/AgentStack) | AgentOps-AI | A comprehensive stack for constructing and deploying AI agents. | [![GitHub Repo stars](https://img.shields.io/github/stars/AgentOps-AI/AgentStack?style=flat-square&color=purple)](https://github.com/AgentOps-AI/AgentStack) | [![GitHub followers](https://img.shields.io/github/followers/AgentOps-AI?style=flat-square&color=teal)](https://github.com/AgentOps-AI) |
| [browser](https://github.com/lightpanda-io/browser) | lightpanda-io | A browser‑based tool designed for integrating agentic functionalities. | [![GitHub Repo stars](https://img.shields.io/github/stars/lightpanda-io/browser?style=flat-square&color=purple)](https://github.com/lightpanda-io/browser) | [![GitHub followers](https://img.shields.io/github/followers/lightpanda-io?style=flat-square&color=teal)](https://github.com/lightpanda-io) |
| [Memary](https://github.com/kingjulio8238/Memary) | kingjulio8238 | A memory module for retaining context in agent workflows. | [![GitHub Repo stars](https://img.shields.io/github/stars/kingjulio8238/Memary?style=flat-square&color=purple)](https://github.com/kingjulio8238/Memary) | [![GitHub followers](https://img.shields.io/github/followers/kingjulio8238?style=flat-square&color=teal)](https://github.com/kingjulio8238) |
| [open-canvas](https://github.com/langchain-ai/open-canvas) | langchain-ai | A visual interface for designing agent workflows with LangChain. | [![GitHub Repo stars](https://img.shields.io/github/stars/langchain-ai/open-canvas?style=flat-square&color=purple)](https://github.com/langchain-ai/open-canvas) | [![GitHub followers](https://img.shields.io/github/followers/langchain-ai?style=flat-square&color=teal)](https://github.com/langchain-ai) |
| [agent-service-toolkit](https://github.com/JoshuaC215/agent-service-toolkit) | JoshuaC215 | A toolkit for building and deploying agent-based services. | [![GitHub Repo stars](https://img.shields.io/github/stars/JoshuaC215/agent-service-toolkit?style=flat-square&color=purple)](https://github.com/JoshuaC215/agent-service-toolkit) | [![GitHub followers](https://img.shields.io/github/followers/JoshuaC215?style=flat-square&color=teal)](https://github.com/JoshuaC215) |


### Virtual Brains

| Tool | Organization | Description | Open Source | GitHub |
|------|--------------|-------------|-------------|--------|
| [Leon](https://github.com/leon-ai/leon) | leon-ai | An open‑source personal assistant and automation platform powered by AI. | [![GitHub Repo stars](https://img.shields.io/github/stars/leon-ai/leon?style=flat-square&color=purple)](https://github.com/leon-ai/leon) | [![GitHub followers](https://img.shields.io/github/followers/leon-ai?style=flat-square&color=teal)](https://github.com/leon-ai) |
| [Khoj](https://github.com/khoj-ai/khoj) | khoj-ai | A virtual brain for organizing and retrieving your knowledge using AI. | [![GitHub Repo stars](https://img.shields.io/github/stars/khoj-ai/khoj?style=flat-square&color=purple)](https://github.com/khoj-ai/khoj) | [![GitHub followers](https://img.shields.io/github/followers/khoj-ai?style=flat-square&color=teal)](https://github.com/khoj-ai) |


### AI Agents

| Framework                                           | Organization                 | Open Source                                                                                                                | Released | GitHub                                                                                   |
|-----------------------------------------------------|------------------------------|-----------------------------------------------------------------------------------------------------------------------------|----------|------------------------------------------------------------------------------------------|
| [GPT Engineer](https://gptengineer.app/)            | GPT Engineer Org             | [![GitHub Repo stars](https://img.shields.io/github/stars/gpt-engineer-org/gpt-engineer?style=flat-square&color=purple)](https://github.com/gpt-engineer-org/gpt-engineer) | 2023     | [![GitHub followers](https://img.shields.io/github/followers/gpt-engineer-org?style=flat-square&color=teal)](https://github.com/gpt-engineer-org) |
| [XAgent](https://github.com/OpenBMB/XAgent)         | OpenBMB                      | [![GitHub Repo stars](https://img.shields.io/github/stars/OpenBMB/XAgent?style=flat-square&color=purple)](https://github.com/OpenBMB/XAgent) | 2023     | [![GitHub followers](https://img.shields.io/github/followers/OpenBMB?style=flat-square&color=teal)](https://github.com/OpenBMB) |
| [Bolt.new](https://github.com/stackblitz/bolt.new)  | StackBlitz                   | [![GitHub Repo stars](https://img.shields.io/github/stars/stackblitz/bolt.new?style=flat-square&color=purple)](https://github.com/stackblitz/bolt.new)  | 2023     | [![GitHub followers](https://img.shields.io/github/followers/stackblitz?style=flat-square&color=teal)](https://github.com/stackblitz)  |
| [Goose](https://github.com/block/goose)             | Block                        | [![GitHub Repo stars](https://img.shields.io/github/stars/block/goose?style=flat-square&color=purple)](https://github.com/block/goose)             | 2023     | [![GitHub followers](https://img.shields.io/github/followers/block?style=flat-square&color=teal)](https://github.com/block)             |
| [AI Hedge Fund](https://github.com/virattt/ai-hedge-fund) | virattt                      | [![GitHub Repo stars](https://img.shields.io/github/stars/virattt/ai-hedge-fund?style=flat-square&color=purple)](https://github.com/virattt/ai-hedge-fund) | 2023     | [![GitHub followers](https://img.shields.io/github/followers/virattt?style=flat-square&color=teal)](https://github.com/virattt) |
| [FinRobot](https://ai4finance.org/)                 | AI4Finance Foundation        | [![GitHub Repo stars](https://img.shields.io/github/stars/AI4Finance-Foundation/FinRobot?style=flat-square&color=purple)](https://github.com/AI4Finance-Foundation/FinRobot) | 2024     | [![GitHub followers](https://img.shields.io/github/followers/AI4Finance-Foundation?style=flat-square&color=teal)](https://github.com/AI4Finance-Foundation) |
| [STORM](https://storm.genie.stanford.edu/)          | Stanford OVAL                | [![GitHub Repo stars](https://img.shields.io/github/stars/stanford-oval/storm?style=flat-square&color=purple)](https://github.com/stanford-oval/storm) | 2024     | [![GitHub followers](https://img.shields.io/github/followers/stanford-oval?style=flat-square&color=teal)](https://github.com/stanford-oval) |
| [Multion](https://www.multion.ai/)                  | MULTI-ON                     | 🔴                                                                                                                          | N/A      | [![GitHub followers](https://img.shields.io/github/followers/MULTI-ON?style=flat-square&color=teal)](https://github.com/MULTI-ON) |
| [Minion](https://minion.ai/)                        | Minion AI                    | 🔴                                                                                                                          | N/A      | [![GitHub followers](https://img.shields.io/github/followers/minionai?style=flat-square&color=teal)](https://github.com/minionai) |

### Long-Term Memory

| Provider     | Community             | Founded     | GitHub                                                                                          | ⭐ Stars | Open Source                 |
|--------------|------------------------|-------------|--------------------------------------------------------------------------------------------------|---------|-----------------------------|
| Mem0         | 🚀 Fast-growing        | June 2023   | [![GitHub followers](https://img.shields.io/github/followers/mem0ai?style=flat-square&color=teal)](https://github.com/mem0ai/mem0)         | 35.2k   | ✅ Apache-2.0               |
| Letta        | 💬 Active dev community| Oct 2023    | [![GitHub followers](https://img.shields.io/github/followers/letta-ai?style=flat-square&color=teal)](https://github.com/letta-ai/letta)     | 17k     | ✅ Apache-2.0               |
| Zep          | 🤝 Moderate community  | Aug 2024    | [![GitHub followers](https://img.shields.io/github/followers/getzep?style=flat-square&color=teal)](https://github.com/getzep/graphiti)      | 11.6k   | ⚠️ Graphiti CE (Apache-2.0) |
| MemoRAG      | 🧪 Small research group| Sep 2024    | [![GitHub followers](https://img.shields.io/github/followers/qhjqhj00?style=flat-square&color=teal)](https://github.com/qhjqhj00/MemoRAG)    | 1.8k    | ✅ Apache-2.0               |
| Memary       | 🧠 Niche community     | April 2024  | [![GitHub followers](https://img.shields.io/github/followers/kingjulio8238?style=flat-square&color=teal)](https://github.com/kingjulio8238/Memary) | 2.3k    | ✅ MIT                      |
| Cognee       | 🔄 Moderate            | Aug 2023    | [![GitHub followers](https://img.shields.io/github/followers/topoteretes?style=flat-square&color=teal)](https://github.com/topoteretes/cognee)     | 5.8k    | ✅ Apache-2.0               |

#### Memory Features Comparison

| Provider  | Based   | Optional KG | Self-Editing / Agentic | Rolling Summaries            | Categories |
|-----------|---------|-------------|-------------------------|-------------------------------|------------|
| Mem0      | 🧮 Vector | ✅ Yes       | ✅ Yes                  | ❌ Not explicit               | ✅ Yes     |
| Letta     | 🧮 Vector | ⚠️ Partial  | ✅ Yes                  | ⚠️ Partial (memory blocks)    | ✅ Yes     |
| Zep       | 🧠 KG     |  -         | ✅ Yes                  | ✅ Auto chat summarization     | ✅ Yes     |
| MemoRAG   | 🧮 Vector | ❌ No       | ✅ Yes                  | ❌ Uses long-range model      | ❌ No      |
| Memary    | 🧠 KG     | -         | ✅ Yes                  | ⚠️ Plans “rewind” feature     | ✅ Yes     |
| Cognee    | 🧠 KG     | -         | ✅ Yes                  | ❌ No auto summaries          | ✅ Yes     |

#### Enterprise Security (Cloud-Based Use)

| Provider  | Enterprise Security                                                                                   |
|-----------|--------------------------------------------------------------------------------------------------------|
| Mem0      | 🔐 Hosted with encryption, org/project roles, GDPR-friendly delete. Uses Graphlit (SOC 2 not stated). |
| Letta     | ☁️ Self-hosted or managed server. User auth & ID-partitioned memory. Graphlit-based. No public SSO details. |
| Zep       | ✅ SOC 2 Type 2. Encrypted at rest/in transit, access controls, JWT, and deletion API ("Right to be Forgotten"). |
| MemoRAG   | 🏠 Self-host                                                                                           |
| Memary    | 🏠 Self-host                                                                                           |
| Cognee    | 🏠 Self-host                                                                                           |


#### Pricing by Monthly Message Volume

| Provider   | 1K msgs/mo       | 10K msgs/mo        | 100K msgs/mo             | 1M msgs/mo               |
|------------|------------------|---------------------|---------------------------|---------------------------|
| Mem0       | 🆓 Free           | 🆓 Free–$29         | 💵 $249                   | 🏢 Enterprise (custom)    |
| Letta      | 🆓 Free           | 💵 $20              | 💵 $750                   | 🏢 Enterprise (custom)    |
| Zep        | 🆓 Free           | 🆓 Free             | 💵 ~ $112.50              | 💵 ~ $1,237               |
| MemoRAG    | 💻 GPU Server (~$150–300/mo) | 💻 GPU Server (~$150–300/mo) | 💻 Multi-GPU ($500+)       | 🖥️ Cluster ($1K+/mo)       |
| Self-host  | 🖥️ Small VM (~$15/mo) | 🖥️ Small VM (~$15–20/mo) | 🖥️ Medium VM ($50–$100/mo) | 🖥️ Large VM ($200+/mo)     |

### Co-Pilots

| Framework                                  | Open Source                                                                                                                | GitHub                                                                                   |
|--------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------|
| [Aider](https://aider.chat/)               | [![GitHub Repo stars](https://img.shields.io/github/stars/Aider-AI/aider?style=flat-square&color=purple)](https://github.com/Aider-AI/aider) | [![GitHub followers](https://img.shields.io/github/followers/Aider-AI?style=flat-square&color=teal)](https://github.com/Aider-AI) |
| [Cursor](https://www.cursor.com/)          | [![GitHub Repo stars](https://img.shields.io/github/stars/getcursor/cursor?style=flat-square&color=purple)](https://github.com/getcursor/cursor) | [![GitHub followers](https://img.shields.io/github/followers/getcursor?style=flat-square&color=teal)](https://github.com/getcursor) |
| [Continue](https://docs.continue.dev/)     | [![GitHub Repo stars](https://img.shields.io/github/stars/continuedev/continue?style=flat-square&color=purple)](https://github.com/continuedev/continue) | [![GitHub followers](https://img.shields.io/github/followers/continuedev?style=flat-square&color=teal)](https://github.com/continuedev) |


### Voice API

| Framework                                  | Open Source                                                                                                                | GitHub                                                                                   |
|--------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------|
| [VAPI.ai](https://vapi.ai/)                | 🔴    | [![GitHub followers](https://img.shields.io/github/followers/vapiai?style=flat-square&color=teal)](https://github.com/vapiai) |
| [Bland.ai](https://www.bland.ai/)          | 🔴                                                                                                                          | N/A                                                                                      |
| [CallAnnie](https://callannie.ai/)         | 🔴                                                                                                                          | N/A                                                                                      |
| [RealtimeTTS](https://github.com/KoljaB/RealtimeTTS) | [![GitHub Repo stars](https://img.shields.io/github/stars/KoljaB/RealtimeTTS?style=flat-square&color=purple)](https://github.com/KoljaB/RealtimeTTS) | [![GitHub followers](https://img.shields.io/github/followers/KoljaB?style=flat-square&color=teal)](https://github.com/KoljaB) |
| [RealtimeSTT](https://github.com/KoljaB/RealtimeSTT) | [![GitHub Repo stars](https://img.shields.io/github/stars/KoljaB/RealtimeSTT?style=flat-square&color=purple)](https://github.com/KoljaB/RealtimeSTT) | [![GitHub followers](https://img.shields.io/github/followers/KoljaB?style=flat-square&color=teal)](https://github.com/KoljaB) |
| [Coqui TTS](https://github.com/coqui-ai/TTS) | [![GitHub Repo stars](https://img.shields.io/github/stars/coqui-ai/TTS?style=flat-square&color=purple)](https://github.com/coqui-ai/TTS) | [![GitHub followers](https://img.shields.io/github/followers/coqui-ai?style=flat-square&color=teal)](https://github.com/coqui-ai) |

### Open Source TTS Models

| Model | License | Stars/Likes | Downloads (Last Month) | Repository |
|----------------------------------------------|------------------------------|----------------|-------------------------|----------------|
| [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) | Apache 2.0 | ⭐ 3.16k (HF) | 📥 557,392 | [Hugging Face](https://huggingface.co/hexgrad/Kokoro-82M) |
| [Zonos-v0.1-transformer](https://huggingface.co/Zyphra/Zonos-v0.1-transformer) | Apache 2.0 | ⭐ 249 (HF) | 📥 24,240 | [Hugging Face](https://huggingface.co/Zyphra/Zonos-v0.1-transformer) |
| [XTTS-v2](https://huggingface.co/coqui/XTTS-v2) | Non-Commercial | ❤️ 368 (HF) | 📥 2,545,850 | [Hugging Face](https://huggingface.co/coqui/XTTS-v2) |
| [ChatTTS](https://github.com/2noise/ChatTTS) | AGPL-3.0 | N/A | N/A | [GitHub](https://github.com/2noise/ChatTTS) |
| [MeloTTS](https://github.com/myshell-ai/MeloTTS) | MIT | N/A | N/A | [GitHub](https://github.com/myshell-ai/MeloTTS) |

For more TTS models and rankings, check out the [TTS Leaderboard](https://huggingface.co/spaces/Pendrokar/TTS-Spaces-Arena).

### LLM Application Frameworks

| Tool | Organization | Description | Open Source | GitHub |
|------|--------------|-------------|-------------|--------|
| [Eino](https://github.com/cloudwego/eino) | CloudWeGo | A lightweight LLM application framework for scalable AI solutions. | [![GitHub Repo stars](https://img.shields.io/github/stars/cloudwego/eino?style=flat-square&color=purple)](https://github.com/cloudwego/eino) | [![GitHub followers](https://img.shields.io/github/followers/cloudwego?style=flat-square&color=teal)](https://github.com/cloudwego) |
| [Conversation Knowledge Mining Solution Accelerator](https://github.com/microsoft/Conversation-Knowledge-Mining-Solution-Accelerator) | Microsoft | A solution accelerator for integrating conversation intelligence and knowledge mining using LLMs. | [![GitHub Repo stars](https://img.shields.io/github/stars/microsoft/Conversation-Knowledge-Mining-Solution-Accelerator?style=flat-square&color=purple)](https://github.com/microsoft/Conversation-Knowledge-Mining-Solution-Accelerator) | [![GitHub followers](https://img.shields.io/github/followers/microsoft?style=flat-square&color=teal)](https://github.com/microsoft) |
| [Olmocr](https://github.com/allenai/olmocr) | AllenAI | An OCR framework optimized for integration with language models. | [![GitHub Repo stars](https://img.shields.io/github/stars/allenai/olmocr?style=flat-square&color=purple)](https://github.com/allenai/olmocr) | [![GitHub followers](https://img.shields.io/github/followers/allenai?style=flat-square&color=teal)](https://github.com/allenai) |
| [PDFMathTranslate](https://github.com/Byaidu/PDFMathTranslate) | Byaidu | A tool for converting and translating mathematical content in PDFs using LLMs. | [![GitHub Repo stars](https://img.shields.io/github/stars/Byaidu/PDFMathTranslate?style=flat-square&color=purple)](https://github.com/Byaidu/PDFMathTranslate) | [![GitHub followers](https://img.shields.io/github/followers/Byaidu?style=flat-square&color=teal)](https://github.com/Byaidu) |
| [Podcastfy](https://github.com/souzatharsis/podcastfy) | souzatharsis | A tool to generate podcasts from written content using LLMs. | [![GitHub Repo stars](https://img.shields.io/github/stars/souzatharsis/podcastfy?style=flat-square&color=purple)](https://github.com/souzatharsis/podcastfy) | [![GitHub followers](https://img.shields.io/github/followers/souzatharsis?style=flat-square&color=teal)](https://github.com/souzatharsis) |
| [Pandas AI](https://github.com/sinaptik-ai/pandas-ai) | sinaptik-ai | Brings LLM-powered analytics to pandas dataframes. | [![GitHub Repo stars](https://img.shields.io/github/stars/sinaptik-ai/pandas-ai?style=flat-square&color=purple)](https://github.com/sinaptik-ai/pandas-ai) | [![GitHub followers](https://img.shields.io/github/followers/sinaptik-ai?style=flat-square&color=teal)](https://github.com/sinaptik-ai) |
| [Ramalama](https://github.com/containers/ramalama) | containers | An LLM application framework for containerized deployment of AI solutions. | [![GitHub Repo stars](https://img.shields.io/github/stars/containers/ramalama?style=flat-square&color=purple)](https://github.com/containers/ramalama) | [![GitHub followers](https://img.shields.io/github/followers/containers?style=flat-square&color=teal)](https://github.com/containers) |
| [Robyn](https://github.com/facebookexperimental/Robyn) | facebookexperimental | A scalable framework for building LLM applications from Facebook Experimental. | [![GitHub Repo stars](https://img.shields.io/github/stars/facebookexperimental/Robyn?style=flat-square&color=purple)](https://github.com/facebookexperimental/Robyn) | [![GitHub followers](https://img.shields.io/github/followers/facebookexperimental?style=flat-square&color=teal)](https://github.com/facebookexperimental) |
| [ExtractThinker](https://github.com/enoch3712/ExtractThinker) | enoch3712 | A tool for extracting and synthesizing insights from textual data using LLMs. | [![GitHub Repo stars](https://img.shields.io/github/stars/enoch3712/ExtractThinker?style=flat-square&color=purple)](https://github.com/enoch3712/ExtractThinker) | [![GitHub followers](https://img.shields.io/github/followers/enoch3712?style=flat-square&color=teal)](https://github.com/enoch3712) |


### OS RAG Frameworks

| Framework                                           | Organization | Open Source                                                                                                                | Released | GitHub                                                                                   |
|-----------------------------------------------------|--------------|-----------------------------------------------------------------------------------------------------------------------------|----------|------------------------------------------------------------------------------------------|
| [Haystack](https://haystack.deepset.ai/)            | deepset.ai   | [![GitHub Repo stars](https://img.shields.io/github/stars/deepset-ai/haystack?style=flat-square&color=purple)](https://github.com/deepset-ai/haystack) | 2023     | [![GitHub followers](https://img.shields.io/github/followers/deepset-ai?style=flat-square&color=teal)](https://github.com/deepset-ai) |
| [RAGflow](https://ragflow.io/)                      | Infiniflow   | [![GitHub Repo stars](https://img.shields.io/github/stars/infiniflow/ragflow?style=flat-square&color=purple)](https://github.com/infiniflow/ragflow) | 2024     | [![GitHub followers](https://img.shields.io/github/followers/infiniflow?style=flat-square&color=teal)](https://github.com/infiniflow) |
| [txtai](https://neuml.github.io/txtai/)             | Neuml        | [![GitHub Repo stars](https://img.shields.io/github/stars/neuml/txtai?style=flat-square&color=purple)](https://github.com/neuml/txtai) | 2022     | [![GitHub followers](https://img.shields.io/github/followers/neuml?style=flat-square&color=teal)](https://github.com/neuml) |
| [LLM App](https://pathway.com/developers/templates) | Pathway      | [![GitHub Repo stars](https://img.shields.io/github/stars/pathwaycom/llm-app?style=flat-square&color=purple)](https://github.com/pathwaycom/llm-app) | 2023     | [![GitHub followers](https://img.shields.io/github/followers/pathwaycom?style=flat-square&color=teal)](https://github.com/pathwaycom) |
| [Cognita](https://cognita.truefoundry.com/)         | Truefoundry  | [![GitHub Repo stars](https://img.shields.io/github/stars/truefoundry/cognita?style=flat-square&color=purple)](https://github.com/truefoundry/cognita) | 2024     | [![GitHub followers](https://img.shields.io/github/followers/truefoundry?style=flat-square&color=teal)](https://github.com/truefoundry) |
| [R2R](https://r2r-docs.sciphi.ai/introduction)      | SciPhi AI    | [![GitHub Repo stars](https://img.shields.io/github/stars/SciPhi-AI/R2R?style=flat-square&color=purple)](https://github.com/SciPhi-AI/R2R) | 2024     | [![GitHub followers](https://img.shields.io/github/followers/SciPhi-AI?style=flat-square&color=teal)](https://github.com/SciPhi-AI) |
| [Raptor](https://arxiv.org/abs/2401.18059)          | Parth Sarthi | [![GitHub Repo stars](https://img.shields.io/github/stars/parthsarthi03/raptor?style=flat-square&color=purple)](https://github.com/parthsarthi03/raptor) | 2024     | [![GitHub followers](https://img.shields.io/github/followers/parthsarthi03?style=flat-square&color=teal)](https://github.com/parthsarthi03) |
| [LightRAG](https://github.com/HKUDS/LightRAG)        | HKUDS        | [![GitHub Repo stars](https://img.shields.io/github/stars/HKUDS/LightRAG?style=flat-square&color=purple)](https://github.com/HKUDS/LightRAG) | 2023     | [![GitHub followers](https://img.shields.io/github/followers/HKUDS?style=flat-square&color=teal)](https://github.com/HKUDS) |
| [PIKE-RAG](https://github.com/microsoft/PIKE-RAG)     | Microsoft    | [![GitHub Repo stars](https://img.shields.io/github/stars/microsoft/PIKE-RAG?style=flat-square&color=purple)](https://github.com/microsoft/PIKE-RAG) | 2024     | [![GitHub followers](https://img.shields.io/github/followers/microsoft?style=flat-square&color=teal)](https://github.com/microsoft) |
| [KAG](https://github.com/OpenSPG/KAG)                | OpenSPG      | [![GitHub Repo stars](https://img.shields.io/github/stars/OpenSPG/KAG?style=flat-square&color=purple)](https://github.com/OpenSPG/KAG) | 2024     | [![GitHub followers](https://img.shields.io/github/followers/OpenSPG?style=flat-square&color=teal)](https://github.com/OpenSPG) |
| [MemoRAG](https://github.com/qhjqhj00/MemoRAG)       | qhjqhj00     | [![GitHub Repo stars](https://img.shields.io/github/stars/qhjqhj00/MemoRAG?style=flat-square&color=purple)](https://github.com/qhjqhj00/MemoRAG) | 2023     | [![GitHub followers](https://img.shields.io/github/followers/qhjqhj00?style=flat-square&color=teal)](https://github.com/qhjqhj00) |

See [RAG_Techniques](https://github.com/NirDiamant/RAG_Techniques) if you get stuck (not always needed)

### 🔍 Vector DBs – FOSS, Performance, Pricing, DevX

| Vector DB        | License       | ⚡ Perf / Throughput                                                                 | ⏱️ Latency (Real-World)                                     | ☁️ Cloud Pricing / Free Tier                                                | 💻 Dev Experience                                                                                 |
|------------------|----------------|-------------------------------------------------------------------------------------|--------------------------------------------------------------|-------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------|
| **Qdrant**       | Apache 2.0     | 🥇 Highest RPS, lowest latency in single-node bench (≥4× vs prev. run)              | p95 < 10ms for 1M vecs (1 thread)                           | Always-on 1GB free; pay-go ≈ $0.014/hr                                     | REST + gRPC; 7 lang clients; filter-aware HNSW; hybrid support; Python “embedded” mode            |
| **Milvus / Zilliz Cloud** | Apache 2.0 | 🚀 Fastest index build; RPS trails Qdrant for high-dim vecs                        | p95 ≈ 10–20ms @ 1M 768-dim (DiskANN, vendor data)            | 5GB free; serverless $0.30/GB-mo; dedicated from $99/mo                    | New SDK v2 (async, schema cache); Python/Go/Java/Node support                                      |
| **Weaviate**     | BSD-3-Clause   | ⚙️ Least bench gains, but decent recall (95%+) and throughput                        | “Low-ms” claimed; users report 100–300ms if misconfigured    | Starts $25/mo; 14-day sandbox free                                         | GraphQL + REST; strong SDKs (Py/TS/Go/Java); easy RAG + hybrid templates                          |
| **pgvector**     | MIT            | 🔥 28× lower p95 & 16× higher QPS vs Pinecone s1 @ 99% recall (50M)                  | p95 < 50ms @ 50M 768-dim (Timescale test)                   | Neon/Supabase offer free Postgres with pgvector (0.5–1GB, ~200h CPU)       | Pure SQL; supports joins + ACID; great for hybrid text + dense queries                            |
| **Redis 8 Vector** | AGPLv3 / RSAL / SSPL | 🧵 3.4× higher QPS vs Qdrant, 4× vs Milvus @ ≥0.98 recall                           | Sub-ms avg, <10ms under load (vendor); 9.7× lower than Aurora+pgvector | Redis Cloud: 30MB free, pay-go from $5/mo; Flex $0.007/hr                  | Redis Vector Library + RAG helpers; OM clients for .NET/Py/JS; fast setup                         |

### 💾 Vector DB Cloud Pricing (2000-char Chunks, ~768-dim)

| # Chunks | Data Size   | 🟣 Milvus / Zilliz Cloud (Serverless)               | 🟢 Qdrant Cloud                                 | 🟡 Weaviate Cloud (“Standard”)                                 |
|----------|-------------|------------------------------------------------------|-------------------------------------------------|---------------------------------------------------------------|
| **10k**  | ~0.07 GB    | 🆓 Free – within 5 GB tier                           | 🆓 Free – fits in 1 GB RAM / 4 GB disk          | $25 base + $1.2 dim fee ≈ **$26**                              |
| **100k** | ~0.67 GB    | 🆓 Still under 5 GB                                 | 🆓 Fits with compression in 4 GB disk           | $25 + $12.0 dim fee ≈ **$37**                                  |
| **1M**   | ~6.7 GB     | 💵 ≈ $2 storage; add vCU fees or $99 dedicated      | 💵 Needs 10 GB cluster → ≈ **$20/mo**           | $25 + $120.5 dim fee ≈ **$145**                                |
| **10M**  | ~67 GB      | 💵 ≈ $20 storage; + compute: **$100–150 total**     | 💵 Needs 64+ GB → **$120–150/mo** estimate      | $25 + $1,204 dim fee ≈ **$1,230**                              |

### 🧠 Embedding Cost – OpenAI (Small Model, per Chunk Size)

| # Chunks     | 📏 1,000 Chars | 📏 2,000 Chars | 📏 3,000 Chars |
|--------------|---------------:|---------------:|---------------:|
| **1,000**     | $0.01          | $0.01          | $0.01          |
| **10,000**    | $0.05          | $0.10          | $0.15          |
| **100,000**   | $0.50          | $1.00          | $1.50          |
| **1,000,000** | $5.00          | $10.00         | $15.00         |
| **10,000,000**| $50.00         | $100.00        | $150.00        |


### AI Tools (for “using”)

| Tool | Organization | Description | Open Source | GitHub |
|------|--------------|-------------|-------------|--------|
| [magic-resume](https://github.com/JOYCEQL/magic-resume) | JOYCEQL | An AI-powered tool for generating resumes. | [![GitHub Repo stars](https://img.shields.io/github/stars/JOYCEQL/magic-resume?style=flat-square&color=purple)](https://github.com/JOYCEQL/magic-resume) | [![GitHub followers](https://img.shields.io/github/followers/JOYCEQL?style=flat-square&color=teal)](https://github.com/JOYCEQL) |
| [VideoCaptioner](https://github.com/WEIFENG2333/VideoCaptioner) | WEIFENG2333 | An AI tool for automatically generating video captions. | [![GitHub Repo stars](https://img.shields.io/github/stars/WEIFENG2333/VideoCaptioner?style=flat-square&color=purple)](https://github.com/WEIFENG2333/VideoCaptioner) | [![GitHub followers](https://img.shields.io/github/followers/WEIFENG2333?style=flat-square&color=teal)](https://github.com/WEIFENG2333) |
| [DeepSeekAI](https://github.com/DeepLifeStudio/DeepSeekAI) | DeepLifeStudio | Browser extension for invoking the DeepSeek AI large model. | [![GitHub Repo stars](https://img.shields.io/github/stars/DeepLifeStudio/DeepSeekAI?style=flat-square&color=purple)](https://github.com/DeepLifeStudio/DeepSeekAI) | [![GitHub followers](https://img.shields.io/github/followers/DeepLifeStudio?style=flat-square&color=teal)](https://github.com/DeepLifeStudio) |
| [logocreator](https://github.com/Nutlope/logocreator) | Nutlope | A tool for creating logos using AI. | [![GitHub Repo stars](https://img.shields.io/github/stars/Nutlope/logocreator?style=flat-square&color=purple)](https://github.com/Nutlope/logocreator) | [![GitHub followers](https://img.shields.io/github/followers/Nutlope?style=flat-square&color=teal)](https://github.com/Nutlope) |
| [blinkshot](https://github.com/Nutlope/blinkshot) | Nutlope | An AI-powered tool for capturing and enhancing screenshots. | [![GitHub Repo stars](https://img.shields.io/github/stars/Nutlope/blinkshot?style=flat-square&color=purple)](https://github.com/Nutlope/blinkshot) | [![GitHub followers](https://img.shields.io/github/followers/Nutlope?style=flat-square&color=teal)](https://github.com/Nutlope) |
| [pollinations](https://github.com/pollinations/pollinations) | pollinations | A tool for generating creative images and artwork using AI. | [![GitHub Repo stars](https://img.shields.io/github/stars/pollinations/pollinations?style=flat-square&color=purple)](https://github.com/pollinations/pollinations) | [![GitHub followers](https://img.shields.io/github/followers/pollinations?style=flat-square&color=teal)](https://github.com/pollinations) |
| [PromptWizard](https://github.com/microsoft/PromptWizard) | microsoft | A tool to generate, manage, and optimize prompts for AI models. | [![GitHub Repo stars](https://img.shields.io/github/stars/microsoft/PromptWizard?style=flat-square&color=purple)](https://github.com/microsoft/PromptWizard) | [![GitHub followers](https://img.shields.io/github/followers/microsoft?style=flat-square&color=teal)](https://github.com/microsoft) |
| [Open-Interface](https://github.com/AmberSahdev/Open-Interface) | AmberSahdev | Control Any Computer Using LLMs. | [![GitHub Repo stars](https://img.shields.io/github/stars/AmberSahdev/Open-Interface?style=flat-square&color=purple)](https://github.com/AmberSahdev/Open-Interface) | [![GitHub followers](https://img.shields.io/github/followers/AmberSahdev?style=flat-square&color=teal)](https://github.com/AmberSahdev) |
| [wut](https://github.com/shobrook/wut) | shobrook | LLM for the terminal | [![GitHub Repo stars](https://img.shields.io/github/stars/shobrook/wut?style=flat-square&color=purple)](https://github.com/shobrook/wut) | [![GitHub followers](https://img.shields.io/github/followers/shobrook?style=flat-square&color=teal)](https://github.com/shobrook) |


### Training/Optimization

| Tool | Organization | Description | Open Source | GitHub |
|------|--------------|-------------|-------------|--------|
| [transformerlab-app](https://github.com/transformerlab/transformerlab-app) | transformerlab | An application for training and optimizing transformer models. | [![GitHub Repo stars](https://img.shields.io/github/stars/transformerlab/transformerlab-app?style=flat-square&color=purple)](https://github.com/transformerlab/transformerlab-app) | [![GitHub followers](https://img.shields.io/github/followers/transformerlab?style=flat-square&color=teal)](https://github.com/transformerlab) |
| [fluxgym](https://github.com/cocktailpeanut/fluxgym) | cocktailpeanut | A gym environment for reinforcement learning training and optimization. | [![GitHub Repo stars](https://img.shields.io/github/stars/cocktailpeanut/fluxgym?style=flat-square&color=purple)](https://github.com/cocktailpeanut/fluxgym) | [![GitHub followers](https://img.shields.io/github/followers/cocktailpeanut?style=flat-square&color=teal)](https://github.com/cocktailpeanut) |
| [AutoGPTQ](https://github.com/AutoGPTQ/AutoGPTQ) | AutoGPTQ | A tool for automating GPT quantization and optimization. | [![GitHub Repo stars](https://img.shields.io/github/stars/AutoGPTQ/AutoGPTQ?style=flat-square&color=purple)](https://github.com/AutoGPTQ/AutoGPTQ) | [![GitHub followers](https://img.shields.io/github/followers/AutoGPTQ?style=flat-square&color=teal)](https://github.com/AutoGPTQ) |


### AI Models

| Tool | Organization | Description | Open Source | GitHub |
|------|--------------|-------------|-------------|--------|
| [WALDO](https://github.com/stephansturges/WALDO) | stephansturges | An AI model for visual reasoning and object detection. | [![GitHub Repo stars](https://img.shields.io/github/stars/stephansturges/WALDO?style=flat-square&color=purple)](https://github.com/stephansturges/WALDO) | [![GitHub followers](https://img.shields.io/github/followers/stephansturges?style=flat-square&color=teal)](https://github.com/stephansturges) |
| [Janus](https://github.com/deepseek-ai/Janus) | deepseek-ai | A multi-modal AI model for advanced data processing. | [![GitHub Repo stars](https://img.shields.io/github/stars/deepseek-ai/Janus?style=flat-square&color=purple)](https://github.com/deepseek-ai/Janus) | [![GitHub followers](https://img.shields.io/github/followers/deepseek-ai?style=flat-square&color=teal)](https://github.com/deepseek-ai) |
| [ModernBERT](https://github.com/AnswerDotAI/ModernBERT) | AnswerDotAI | A modernized version of BERT for natural language processing tasks. | [![GitHub Repo stars](https://img.shields.io/github/stars/AnswerDotAI/ModernBERT?style=flat-square&color=purple)](https://github.com/AnswerDotAI/ModernBERT) | [![GitHub followers](https://img.shields.io/github/followers/AnswerDotAI?style=flat-square&color=teal)](https://github.com/AnswerDotAI) |
| [Magma](https://github.com/microsoft/Magma) | microsoft | A scalable AI model for large-scale data analysis. | [![GitHub Repo stars](https://img.shields.io/github/stars/microsoft/Magma?style=flat-square&color=purple)](https://github.com/microsoft/Magma) | [![GitHub followers](https://img.shields.io/github/followers/microsoft?style=flat-square&color=teal)](https://github.com/microsoft) |
| [Cosmos-Nemotron](https://github.com/NVlabs/Cosmos-Nemotron) | NVlabs | An AI model for advanced image and video processing. | [![GitHub Repo stars](https://img.shields.io/github/stars/NVlabs/Cosmos-Nemotron?style=flat-square&color=purple)](https://github.com/NVlabs/Cosmos-Nemotron) | [![GitHub followers](https://img.shields.io/github/followers/NVlabs?style=flat-square&color=teal)](https://github.com/NVlabs) |
| [Paints-UNDO](https://github.com/lllyasviel/Paints-UNDO) | lllyasviel | An interactive AI model for image generation and editing. | [![GitHub Repo stars](https://img.shields.io/github/stars/lllyasviel/Paints-UNDO?style=flat-square&color=purple)](https://github.com/lllyasviel/Paints-UNDO) | [![GitHub followers](https://img.shields.io/github/followers/lllyasviel?style=flat-square&color=teal)](https://github.com/lllyasviel) |


### Monitoring / Evaluation

| Tool | Organization | Description | Open Source | GitHub |
|------|--------------|-------------|-------------|--------|
| [helicone](https://github.com/Helicone/helicone) | Helicone | A platform for monitoring and analyzing AI model performance. | [![GitHub Repo stars](https://img.shields.io/github/stars/Helicone/helicone?style=flat-square&color=purple)](https://github.com/Helicone/helicone) | [![GitHub followers](https://img.shields.io/github/followers/Helicone?style=flat-square&color=teal)](https://github.com/Helicone) |
| [langwatch](https://github.com/langwatch/langwatch) | langwatch | A tool for monitoring outputs and performance of language models. | [![GitHub Repo stars](https://img.shields.io/github/stars/langwatch/langwatch?style=flat-square&color=purple)](https://github.com/langwatch/langwatch) | [![GitHub followers](https://img.shields.io/github/followers/langwatch?style=flat-square&color=teal)](https://github.com/langwatch) |
| [shortest](https://github.com/antiwork/shortest) | antiwork | A tool for evaluating and optimizing AI-generated content. | [![GitHub Repo stars](https://img.shields.io/github/stars/antiwork/shortest?style=flat-square&color=purple)](https://github.com/antiwork/shortest) | [![GitHub followers](https://img.shields.io/github/followers/antiwork?style=flat-square&color=teal)](https://github.com/antiwork) |
| [deepeval](https://github.com/confident-ai/deepeval) | confident-ai | A framework for deep evaluation of AI models. | [![GitHub Repo stars](https://img.shields.io/github/stars/confident-ai/deepeval?style=flat-square&color=purple)](https://github.com/confident-ai/deepeval) | [![GitHub followers](https://img.shields.io/github/followers/confident-ai?style=flat-square&color=teal)](https://github.com/confident-ai) |


### Infrastructure

| Tool | Organization | Description | Open Source | GitHub |
|------|--------------|-------------|-------------|--------|
| [gpustack](https://github.com/gpustack/gpustack) | gpustack | A toolkit for managing GPU infrastructure for AI workloads. | [![GitHub Repo stars](https://img.shields.io/github/stars/gpustack/gpustack?style=flat-square&color=purple)](https://github.com/gpustack/gpustack) | [![GitHub followers](https://img.shields.io/github/followers/gpustack?style=flat-square&color=teal)](https://github.com/gpustack) |
| [harbor](https://github.com/av/harbor) | av | A repository for containerized AI infrastructure management. | [![GitHub Repo stars](https://img.shields.io/github/stars/av/harbor?style=flat-square&color=purple)](https://github.com/av/harbor) | [![GitHub followers](https://img.shields.io/github/followers/av?style=flat-square&color=teal)](https://github.com/av) |


### Research Papers on Chain-of-Thought Prompting

| Publication Date | Title | 🔗 | Authors | Organization | Technique |
|------------------|-------|----|---------|--------------|-----------|
| January 28, 2022 | Chain-of-Thought Prompting Elicits Reasoning in Large Language Models | [🔗](https://arxiv.org/abs/2201.11903) | Jason Wei, et al. | DeepMind | CoT Prompting |
| March 21, 2022 | Self-Consistency Improves Chain of Thought Reasoning in Language Models | [🔗](https://arxiv.org/abs/2203.11171) | Xuezhi Wang et al. | DeepMind | CoT with Self-Consistency |
| May 21, 2022 | Least-to-Most Prompting Enables Complex Reasoning in Large Language Models | [🔗](https://arxiv.org/abs/2205.10625) | Denny Zhou et al. | DeepMind | Least-to-Most Prompting |
| May 21, 2022 | Large Language Models are Zero-Shot Reasoners | [🔗](https://arxiv.org/abs/2205.11916) | Takeshi Kojima, et al. | DeepMind | Zero-shot-CoT |
| October 6, 2022 | ReAct: Synergizing Reasoning and Acting in Language Models | [🔗](https://arxiv.org/abs/2210.03629) | Shunyu Yao et al. | Princeton University | ReAct |
| April 1, 2023 | Teaching Large Language Models to Self-Debug | [🔗](https://arxiv.org/abs/2304.05128) | Xiang Lisa Li, et al. | DeepMind, Stanford University | Self-Debugging |
| May 6, 2023 | Plan-and-Solve Prompting: Improving Zero-Shot Chain-of-Thought Reasoning by Large Language Models | [🔗](https://arxiv.org/abs/2305.04091) | Lei Wang et al. | The Chinese University of Hong Kong, SenseTime Research | Plan-and-Solve Prompting |
| May 23, 2023 | Let’s Verify Step by Step | [🔗](https://arxiv.org/pdf/2305.20050) | Anya Goyal, et al. | DeepMind | Verification for CoT |
| October 3, 2023 | Large Language Models Cannot Self-Correct Reasoning Yet | [🔗](https://arxiv.org/abs/2310.01798) | Qingxiu Dong, et al. | The Chinese University of Hong Kong, Huawei Noah's Ark Lab | Self-Correction in LLMs |
| November 2023 | Universal Self-Consistency for Large Language Model Generation | [🔗](https://arxiv.org/pdf/2311.17311) | Xinyun Chen, Renat Aksitov, Uri Alon, Jie Ren, Kefan Xiao, Pengcheng Yin, Sushant Prakash, Charles Sutton, Xuezhi Wang, Denny Zhou | DeepMind | Universal Self-Consistency |
| May 17, 2023 | Tree of Thoughts: Deliberate Problem Solving with Large Language Models | [🔗](https://export.arxiv.org/abs/2305.10601) | Shunyu Yao, et al. | Princeton University, DeepMind | Tree-of-Thought |
| February 15, 2024 | Chain-of-Thought Reasoning Without Prompting | [🔗](https://arxiv.org/abs/2402.10200) | Xuezhi Wang, Denny Zhou | DeepMind | Chain-of-Thought Decoding |
| March 21, 2024 | ChainLM: Empowering Large Language Models with Improved Chain-of-Thought Prompting | [🔗](https://arxiv.org/abs/2403.14238) | Xiaoxue Cheng et al. | Renmin University of China | CoTGenius |
| June 2024 | Language Agent Tree Search Unifies Reasoning, Acting, and Planning in Language Models | [🔗](https://arxiv.org/pdf/2310.04406) | Andy Zhou, Kai Yan, Michal Shlapentokh-Rothman, Haohan Wang, Yu-Xiong Wang |  | Language Agent Tree Search (LATS) |
| May 2024 | Monte Carlo Tree Search Boosts Reasoning via Iterative Preference Learning | [🔗](https://arxiv.org/pdf/2405.00451) | Yuxi Xie, et al. | National University of Singapore, DeepMind | MCTS |
| September 18, 2024 | To CoT or not to CoT? Chain-of-thought helps mainly on math and symbolic reasoning | [🔗](https://arxiv.org/abs/2409.12183) | Zayne Sprague, et al. | The University of Texas at Austin, Johns Hopkins University, Princeton University | Meta-analysis of CoT |
| September 25, 2024 | Chain-of-Thoughtlessness? An Analysis of CoT in Planning | [🔗](https://arxiv.org/abs/2305.12147) | Kaya Stechly, et al. | Arizona State University | Analysis of CoT in Planning |
| October 18, 2024 | Supervised Chain of Thought | [🔗](https://arxiv.org/abs/2410.14198) | Xiang Zhang, Dujian Ding | University of British Columbia | Supervised Chain of Thought |
| October 24, 2024 | On examples: A Theoretical Understanding of Chain-of-Thought: Coherent Reasoning and Error-Aware Demonstration | [🔗](https://arxiv.org/abs/2410.16540) | Zhiqiang Hu, et al. | Amazon, Michigan State University | Theoretical Analysis of CoT |



### CoT Implementations

| Implementation | Link | Author | GitHub Stars | GitHub Followers |
|----------------|------|---------|--------------|------------------|
| CoT | [chain-of-thought-hub](https://github.com/FranxYao/chain-of-thought-hub) | Franx Yao | [![Stars](https://img.shields.io/github/stars/FranxYao/chain-of-thought-hub?style=flat-square&color=purple)](https://github.com/FranxYao/chain-of-thought-hub) | [![Followers](https://img.shields.io/github/followers/FranxYao?style=flat-square&color=teal)](https://github.com/FranxYao) |
| CoT | [optillm](https://github.com/codelion/optillm) | Codelion | [![Stars](https://img.shields.io/github/stars/codelion/optillm?style=flat-square&color=purple)](https://github.com/codelion/optillm) | [![Followers](https://img.shields.io/github/followers/codelion?style=flat-square&color=teal)](https://github.com/codelion) |
| CoT | [auto-cot](https://github.com/amazon-science/auto-cot) | Amazon Science | [![Stars](https://img.shields.io/github/stars/amazon-science/auto-cot?style=flat-square&color=purple)](https://github.com/amazon-science/auto-cot) | [![Followers](https://img.shields.io/github/followers/amazon-science?style=flat-square&color=teal)](https://github.com/amazon-science) |
| CoT | [g1](https://github.com/bklieger-groq/g1) | BKlieger Groq | [![Stars](https://img.shields.io/github/stars/bklieger-groq/g1?style=flat-square&color=purple)](https://github.com/bklieger-groq/g1) | [![Followers](https://img.shields.io/github/followers/bklieger-groq?style=flat-square&color=teal)](https://github.com/bklieger-groq) |
| Decoding CoT | [optillm/cot_decoding.py](https://github.com/codelion/optillm/blob/main/optillm/cot_decoding.py) | Codelion | [![Stars](https://img.shields.io/github/stars/codelion/optillm?style=flat-square&color=purple)](https://github.com/codelion/optillm) | [![Followers](https://img.shields.io/github/followers/codelion?style=flat-square&color=teal)](https://github.com/codelion) |
| Tree of Thoughts | [tree-of-thought-llm](https://github.com/princeton-nlp/tree-of-thought-llm) | Princeton NLP | [![Stars](https://img.shields.io/github/stars/princeton-nlp/tree-of-thought-llm?style=flat-square&color=purple)](https://github.com/princeton-nlp/tree-of-thought-llm) | [![Followers](https://img.shields.io/github/followers/princeton-nlp?style=flat-square&color=teal)](https://github.com/princeton-nlp) |
| Tree of Thoughts | [tree-of-thoughts](https://github.com/kyegomez/tree-of-thoughts) | Kye Gomez | [![Stars](https://img.shields.io/github/stars/kyegomez/tree-of-thoughts?style=flat-square&color=purple)](https://github.com/kyegomez/tree-of-thoughts) | [![Followers](https://img.shields.io/github/followers/kyegomez?style=flat-square&color=teal)](https://github.com/kyegomez) |
| Tree of Thoughts | [saplings](https://github.com/shobrook/saplings) | Shobrook | [![Stars](https://img.shields.io/github/stars/shobrook/saplings?style=flat-square&color=purple)](https://github.com/shobrook/saplings) | [![Followers](https://img.shields.io/github/followers/shobrook?style=flat-square&color=teal)](https://github.com/shobrook) |
| MCTS | [optillm/mcts.py](https://github.com/codelion/optillm/blob/main/optillm/mcts.py) | Codelion | [![Stars](https://img.shields.io/github/stars/codelion/optillm?style=flat-square&color=purple)](https://github.com/codelion/optillm) | [![Followers](https://img.shields.io/github/followers/codelion?style=flat-square&color=teal)](https://github.com/codelion) |
| Graph of Thoughts | [graph-of-thoughts](https://github.com/spcl/graph-of-thoughts) | SPCL | [![Stars](https://img.shields.io/github/stars/spcl/graph-of-thoughts?style=flat-square&color=purple)](https://github.com/spcl/graph-of-thoughts) | [![Followers](https://img.shields.io/github/followers/spcl?style=flat-square&color=teal)](https://github.com/spcl) |
| Other | [CPO](https://github.com/sail-sg/CPO) | SAIL SG | [![Stars](https://img.shields.io/github/stars/sail-sg/CPO?style=flat-square&color=purple)](https://github.com/sail-sg/CPO) | [![Followers](https://img.shields.io/github/followers/sail-sg?style=flat-square&color=teal)](https://github.com/sail-sg) |
| Other | [Everything-of-Thoughts-XoT](https://github.com/microsoft/Everything-of-Thoughts-XoT) | Microsoft | [![Stars](https://img.shields.io/github/stars/microsoft/Everything-of-Thoughts-XoT?style=flat-square&color=purple)](https://github.com/microsoft/Everything-of-Thoughts-XoT) | [![Followers](https://img.shields.io/github/followers/microsoft?style=flat-square&color=teal)](https://github.com/microsoft) |

### CoT Fine-Tuned Models & Datasets

#### Models
| Model Name | Author | Size | Link |
|------------|--------|------|------|
| CoT-T5-3B | KAIST AI | 3B | [🔗](https://huggingface.co/kaist-ai/CoT-T5-3B) |
| CoT-T5-11B | KAIST AI | 11B | [🔗](https://huggingface.co/kaist-ai/CoT-T5-11B) |
| Llama-3.2V-11B-cot | Xkev | 11B | [🔗](https://huggingface.co/Xkev/Llama-3.2V-11B-cot) |
| Llama-3.1-8B-Instruct-Reasoner-1o1_v0.3 | Lyte | 8B | [🔗](https://huggingface.co/Lyte/Llama-3.1-8B-Instruct-Reasoner-1o1_v0.3) |


#### Datasets
| Dataset Name | Author | Data Size | Likes | Link |
|--------------|--------|-----------|-------|------|
| chain-of-thought-sharegpt | Isaiah Bjork | 7.14k rows | 🌟 8 | [🔗](https://huggingface.co/datasets/isaiahbjork/chain-of-thought-sharegpt) |
| CoT-Collection | KAIST AI | 1.84 million rows | 🌟 122 | [🔗](https://huggingface.co/datasets/kaist-ai/CoT-Collection?row=4) |
| Reasoner-1o1-v0.3-HQ | Lyte | 370 rows | 🌟 7 | [🔗](https://huggingface.co/datasets/Lyte/Reasoner-1o1-v0.3-HQ) |
| OpenLongCoT-Pretrain | qq8933 | 103k rows | 🌟 86 | [🔗](https://huggingface.co/datasets/qq8933/OpenLongCoT-Pretrain?row=0) |


### Learning Resources

| Tool | Organization | Description | Open Source | GitHub |
|------|--------------|-------------|-------------|--------|
| [awesome-cursorrules](https://github.com/PatrickJS/awesome-cursorrules) | PatrickJS | A curated list of resources and guides on cursorrules. | [![GitHub Repo stars](https://img.shields.io/github/stars/PatrickJS/awesome-cursorrules?style=flat-square&color=purple)](https://github.com/PatrickJS/awesome-cursorrules) | [![GitHub followers](https://img.shields.io/github/followers/PatrickJS?style=flat-square&color=teal)](https://github.com/PatrickJS) |
| [ai-engineering-hub](https://github.com/patchy631/ai-engineering-hub) | patchy631 | A hub of AI engineering learning resources, tutorials, and best practices. | [![GitHub Repo stars](https://img.shields.io/github/stars/patchy631/ai-engineering-hub?style=flat-square&color=purple)](https://github.com/patchy631/ai-engineering-hub) | [![GitHub followers](https://img.shields.io/github/followers/patchy631?style=flat-square&color=teal)](https://github.com/patchy631) |
| [GenAI_Agents](https://github.com/NirDiamant/GenAI_Agents) | NirDiamant | Resources and examples for building Generative AI Agents. | [![GitHub Repo stars](https://img.shields.io/github/stars/NirDiamant/GenAI_Agents?style=flat-square&color=purple)](https://github.com/NirDiamant/GenAI_Agents) | [![GitHub followers](https://img.shields.io/github/followers/NirDiamant?style=flat-square&color=teal)](https://github.com/NirDiamant) |
| [learn-agentic-ai](https://github.com/panaversity/learn-agentic-ai) | panaversity | Learning materials for understanding and building agentic AI. | [![GitHub Repo stars](https://img.shields.io/github/stars/panaversity/learn-agentic-ai?style=flat-square&color=purple)](https://github.com/panaversity/learn-agentic-ai) | [![GitHub followers](https://img.shields.io/github/followers/panaversity?style=flat-square&color=teal)](https://github.com/panaversity) |
| [awesome-generative-ai](https://github.com/steven2358/awesome-generative-ai) | steven2358 | A curated list of generative AI resources and projects. | [![GitHub Repo stars](https://img.shields.io/github/stars/steven2358/awesome-generative-ai?style=flat-square&color=purple)](https://github.com/steven2358/awesome-generative-ai) | [![GitHub followers](https://img.shields.io/github/followers/steven2358?style=flat-square&color=teal)](https://github.com/steven2358) |
| [awesome-mcp-servers](https://github.com/punkpeye/awesome-mcp-servers) | punkpeye | A curated collection of awesome MCP servers resources. | [![GitHub Repo stars](https://img.shields.io/github/stars/punkpeye/awesome-mcp-servers?style=flat-square&color=purple)](https://github.com/punkpeye/awesome-mcp-servers) | [![GitHub followers](https://img.shields.io/github/followers/punkpeye?style=flat-square&color=teal)](https://github.com/punkpeye) |
| [GenAI-Showcase](https://github.com/mongodb-developer/GenAI-Showcase) | mongodb-developer | A showcase of innovative Generative AI projects. | [![GitHub Repo stars](https://img.shields.io/github/stars/mongodb-developer/GenAI-Showcase?style=flat-square&color=purple)](https://github.com/mongodb-developer/GenAI-Showcase) | [![GitHub followers](https://img.shields.io/github/followers/mongodb-developer?style=flat-square&color=teal)](https://github.com/mongodb-developer) |
| [well-architected-iac-analyzer](https://github.com/aws-samples/well-architected-iac-analyzer) | aws-samples | A tool to analyze and ensure well-architected Infrastructure as Code practices. | [![GitHub Repo stars](https://img.shields.io/github/stars/aws-samples/well-architected-iac-analyzer?style=flat-square&color=purple)](https://github.com/aws-samples/well-architected-iac-analyzer) | [![GitHub followers](https://img.shields.io/github/followers/aws-samples?style=flat-square&color=teal)](https://github.com/aws-samples) |
| [llama-cookbook](https://github.com/meta-llama/llama-cookbook) | meta-llama | A collection of recipes and guides for working with LLaMA models. | [![GitHub Repo stars](https://img.shields.io/github/stars/meta-llama/llama-cookbook?style=flat-square&color=purple)](https://github.com/meta-llama/llama-cookbook) | [![GitHub followers](https://img.shields.io/github/followers/meta-llama?style=flat-square&color=teal)](https://github.com/meta-llama) |
| [optillm](https://github.com/codelion/optillm) | codelion | Resources for optimizing LLM usage and performance. | [![GitHub Repo stars](https://img.shields.io/github/stars/codelion/optillm?style=flat-square&color=purple)](https://github.com/codelion/optillm) | [![GitHub followers](https://img.shields.io/github/followers/codelion?style=flat-square&color=teal)](https://github.com/codelion) |
| [cursor.directory](https://github.com/pontusab/cursor.directory) | pontusab | A directory of tools and resources related to cursor-based workflows. | [![GitHub Repo stars](https://img.shields.io/github/stars/pontusab/cursor.directory?style=flat-square&color=purple)](https://github.com/pontusab/cursor.directory) | [![GitHub followers](https://img.shields.io/github/followers/pontusab?style=flat-square&color=teal)](https://github.com/pontusab) |
| [GenAI_Agents](https://github.com/NirDiamant/GenAI_Agents) | NirDiamant | A curated collection of generative AI agents and related tools. | [![GitHub Repo stars](https://img.shields.io/github/stars/NirDiamant/GenAI_Agents?style=flat-square&color=purple)](https://github.com/NirDiamant/GenAI_Agents) | [![GitHub followers](https://img.shields.io/github/followers/NirDiamant?style=flat-square&color=teal)](https://github.com/NirDiamant) |


