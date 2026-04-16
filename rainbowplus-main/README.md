# 🌈 RainbowPlus

This tree is a **vendored fork** inside the parent **ToxSearch-S** repository: paths below are relative to **`rainbowplus-main/`** (run `pip install -e .` from this directory). Comparison and reproduction notes for joint experiments are at the repo root (`README.md`, `ARCHITECTURE.md`, `experiments/EXPERIMENT_PLAN.md`).

## 📋 Overview
This package implements the methods described in **"[RainbowPlus: Enhancing Adversarial Prompt Generation via Evolutionary Quality-Diversity Search](https://arxiv.org/abs/2504.15047)"**. Building upon the foundational insights of Rainbow Teaming and the MAP-Elites algorithm, **RainbowPlus** introduces key enhancements to the evolutionary quality-diversity (QD) paradigm. 

Specifically, **RainbowPlus** reimagines the archive as a dynamic, multi-individual container that stores diverse high-fitness prompts per cell, analogous to maintaining a population of elite solutions across behavioral niches. This enriched archive enables a broader evolutionary exploration of adversarial strategies. 

Furthermore, **RainbowPlus** employs a comprehensive fitness function that evaluates multiple candidate prompts in parallel using a probabilistic scoring mechanism, replacing traditional pairwise comparisons and enhancing both accuracy and computational efficiency. By integrating these evolutionary principles into its adaptive QD search, **RainbowPlus** achieves superior attack efficacy and prompt diversity, outperforming both QD-based methods and state-of-the-art red-teaming approaches.

![Diagram](/assets/diagram.png)

## ✨ Key Features

- 🏆 **State-of-the-Art Performance**: Achieves superior results compared to existing methods on HarmBench benchmark
- 🔄 **Universal Compatibility**: Supports both open-source and closed-source LLMs (OpenAI, vLLM)
- ⚡ **Computational Efficiency**: Completes an evaluation in just 1.45 hours on HarmBench
- 🛠️ **Flexible Configuration**: Highly customizable for various experimental settings

*RainbowPlus* achieves state-of-the-art results compared to other red-teaming methods.

![Results](/assets/results-sota.png)

## 📁 Repository Structure

```
├── configs/                  # Configuration files
│   ├── categories/           # Category definitions
│   ├── styles/               # Style definitions
│   ├── base.yml              # Base configuration
│   ├── base-openai.yml       # Configuration to run LLMs from OpenAI
│   ├── base-opensource.yml   # Configuration to run open-source LLMs
│   ├── SOTA_COMPARISON.md    # How base.yml matches ToxSearch-S RG/PG sampling
│   └── eval.yml              # Evaluation configuration
│
├── data/                     # Dataset storage
│
├── rainbowplus/              # Core package
│   ├── configs/              # Configuration utilities
│   ├── llms/                 # LLM integration modules
│   ├── scores/               # Fitness and similarity functions
│   ├── archive.py            # Archive management
│   ├── evaluate.py           # Current evaluation implementation
│   ├── evaluate_v0.py        # Evaluation implementation from old version
│   ├── get_scores.py         # Metrics extraction utilities
│   ├── prompts.py            # LLM prompt templates
│   ├── rainbowplus.py        # Main implementation
│   └── utils.py              # Utility functions
│
├── sbatch_rainbowplus.sh     # Slurm array launcher (used from ToxSearch-S `experiments/EXPERIMENT_PLAN.md`)
├── sh/                       # Shell scripts
│   └── run.sh                # All-in-one execution script
│
├── README.md                 # This documentation
└── setup.py                  # Package installation script
```

## 🚀 Getting Started

### 1️⃣ Environment Setup

Create and activate a Python virtual environment, then install the required dependencies:

```bash
python -m venv venv
source venv/bin/activate
pip install -e .
```

### 2️⃣ API Configuration

#### 🤗 Hugging Face Token (Optional)

Required for accessing certain resources from the Hugging Face Hub (e.g., Llama Guard):

```bash
export HF_AUTH_TOKEN="YOUR_HF_TOKEN"
```

Alternatively:

```bash
huggingface-cli login --token=YOUR_HF_TOKEN
```

#### 🔑 OpenAI API Key (Optional)

Required when using OpenAI models:

```bash
export OPENAI_API_KEY="YOUR_API_KEY"
```

## 📊 Usage

### 🧠 LLM Configuration

RainbowPlus supports two primary LLM integration methods:

#### 1️⃣ vLLM (Open-Source Models)

Example configuration for Qwen-2.5-7B-Instruct:

```yaml
target_llm:
  type_: vllm

  model_kwargs:
    model: Qwen/Qwen2.5-7B-Instruct
    trust_remote_code: True
    max_model_len: 2048
    gpu_memory_utilization: 0.5

  sampling_params:
    temperature: 0.6
    top_p: 0.9
    max_tokens: 1024
```

Additional parameters can be specified according to the [vLLM model documentation](https://docs.vllm.ai/en/latest/api/offline_inference/llm.html) and [sampling parameters documentation](https://docs.vllm.ai/en/latest/api/inference_params.html#sampling-parameters).

#### 2️⃣ OpenAI API (Closed-Source Models)

Example configuration for GPT-4o-mini:

```yaml
target_llm:
  type_: openai

  model_kwargs:
    model: gpt-4o-mini

  sampling_params:
    temperature: 0.6
    top_p: 0.9
    max_tokens: 1024
```

Additional parameters can be specified according to the [OpenAI API documentation](https://platform.openai.com/docs/api-reference/chat/create).

### SOTA comparison with ToxSearch-S (sampling alignment)

**`configs/base.yml`** (the default) uses `temperature`, `top_p`, `top_k`, and `max_tokens` aligned with ToxSearch-S **`RGConfig.yaml`** (target) and **`PGConfig.yaml`** (mutator). See **`configs/SOTA_COMPARISON.md`**. Set the local **`model:`** GGUF path and **`tokenizer:`** in `base.yml` (or `--target_llm` for the target path) and **`PERSPECTIVE_API_KEY`** when using the Perspective fitness scorer.

### 🧪 Running Experiments

Basic execution with default configuration:

```bash
python -m rainbowplus.rainbowplus \
    --config_file configs/base.yml \
    --num_samples 150 \
    --max_iters 400 \
    --sim_threshold 0.6 \
    --num_mutations 10 \
    --fitness_threshold 0.6 \
    --log_dir logs-sota \
    --dataset ./data/harmbench.json \
    --log_interval 50 \
    --shuffle True
```

For customized experiments, you can override target LLM and specific parameters:

```bash
python -m rainbowplus.rainbowplus \
    --config_file configs/base-opensource.yml \
    --num_samples -1 \
    --max_iters 400 \
    --sim_threshold 0.6 \
    --num_mutations 10 \
    --fitness_threshold 0.6 \
    --log_dir logs-sota \
    --dataset ./data/harmbench.json \
    --target_llm "TARGET MODEL" \
    --log_interval 50 \
    --shuffle True
```

#### ⚙️ Configuration Parameters

| Parameter | Description |
|-----------|-------------|
| `target_llm` | Target LLM identifier |
| `num_samples` | Number of initial seed prompts |
| `max_iters` | Maximum number of iteration steps |
| `sim_threshold` | Similarity threshold for prompt mutation |
| `num_mutations` | Number of prompt mutations per iteration |
| `fitness_threshold` | Minimum fitness score to add prompt to archive |
| `log_dir` | Directory for storing logs |
| `dataset` | Dataset path |
| `shuffle` | Whether to shuffle seed prompts |
| `log_interval` | Number of iterations between log saves |

### 🔄 Batch Processing Multiple Models

For evaluating multiple models sequentially:

```bash
MODEL_IDS="meta-llama/Llama-2-7b-chat-hf lmsys/vicuna-7b-v1.5 baichuan-inc/Baichuan2-7B-Chat Qwen/Qwen-7B-Chat"

for MODEL in $MODEL_IDS; do
    python -m rainbowplus.rainbowplus \
        --config_file configs/base-opensource.yml \
        --num_samples -1 \
        --max_iters 400 \
        --sim_threshold 0.6 \
        --num_mutations 10 \
        --fitness_threshold 0.6 \
        --log_dir logs-sota \
        --dataset ./data/harmbench.json \
        --target_llm $MODEL \
        --log_interval 50 \
        --shuffle True

    # Clean cache between model runs
    rm -r ~/.cache/huggingface/hub/
done
```

## 📊 Evaluation

After running experiments, evaluate the results:

```bash
MODEL_IDS="meta-llama/Llama-2-7b-chat-hf"  # For multiple models: MODEL_IDS="meta-llama/Llama-2-7b-chat-hf lmsys/vicuna-7b-v1.5"

for MODEL in $MODEL_IDS; do
    # Run evaluation
    python -m rainbowplus.evaluate \
        --config configs/eval.yml \
        --log_dir "./logs-sota/$MODEL/harmbench"

    # Extract metrics
    python rainbowplus/get_scores.py \
        --log_dir "./logs-sota/$MODEL/harmbench" \
        --keyword "global"
done
```

### Evaluation Parameters

| Parameter | Description |
|-----------|-------------|
| `config` | Path to configuration file |
| `log_dir` | Directory containing experiment logs |
| `keyword` | Keyword for global config file name (default: `global`), you can ignore this param |

### 📉 Output Metrics

Results are saved in JSON format:

```json
{
    "General": 0.79,
    "All": 0.8666092943201377
}
```

Where:
- `General`: Metrics calculated following standard methods
- `All`: Metrics calculated across all generated prompts

## ⚡ Streamlined Execution

For end-to-end execution, use the provided shell script [sh/run.sh](sh/run.sh):

```bash
bash sh/run.sh
```

- Modify common parameters (`log_dir, max_iters, ...`) in line [4-11](https://github.com/knoveleng/rainbowplus/blob/c4d679395d1dc5fd3d35da5e98f1568a06d3ee39/sh/run.sh#L4-L11).
- Modify target LLMs in line [56-77](https://github.com/knoveleng/rainbowplus/blob/c4d679395d1dc5fd3d35da5e98f1568a06d3ee39/sh/run.sh#L56-L67) for open-source models.
- Modify target LLMs in line [80-83](https://github.com/knoveleng/rainbowplus/blob/c4d679395d1dc5fd3d35da5e98f1568a06d3ee39/sh/run.sh#L80-L82) for closed-source models.

## 🔮 Next Features

- [ ] Support OpenAI as fitness function
- [ ] Deploy via FastAPI
- [ ] Support more LLMs

## 📝 Citation

```
@misc{dang2025rainbowplusenhancingadversarialprompt,
      title={RainbowPlus: Enhancing Adversarial Prompt Generation via Evolutionary Quality-Diversity Search}, 
      author={Quy-Anh Dang and Chris Ngo and Truong-Son Hy},
      year={2025},
      eprint={2504.15047},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2504.15047}, 
}
```
