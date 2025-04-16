# Awesome-Efficient-CoT-Reasoning-Summary

🔥 Recently, chain-of-thought (CoT) has become a crucial technique for accurate and explainable LLM reasoning! DeepSeek-R1 and OpenAI o1 demonstrate powerful reasoning ability with CoT.

✅ Many research efforts have been made to generate longer and more detailed thoughts before generating the final answer (the figure below shows the simple evidence of CoT lengthening).

👀 We can observe that the redundant reasoning steps are widespread. Also, recent research has found the overthinking phenomenon in CoT, which can be further improved.

❓ How to efficiently and effectively compress the CoTs or directly generate concise CoTs during inference while maintaining the reasoning performance is an important topic!

🎯 In this repo, I track the hotspots of efficient CoT reasoning, and more importantly, I also summarize the main contribution or method of these works.

😊 If I leave out any important papers, please let me know and I will include them as soon as possible.
I will actively maintain the repo for our brainstorming in this field.

![image](https://github.com/zwxandy/Efficient-CoT-Reasoning/blob/main/long_cot.png)

## Content
* [Recent Survey](#recent-survey)
* [Prompting-guided CoT Compression](#prompting-guided-cot-compression)
* [Latent-space CoT Reasoning](#latent-space-cot-reasoning)
* [Training-internized CoT Compression](#training-internized-cot-compression)
* [Inference-time CoT Compression](#inference-time-cot-compression)
* [Analysis of CoT Compression](#analysis-of-cot-compression)


## Recent Survey
* [Stop Overthinking: A Survey on Efficient Reasoning for Large Language Models](https://arxiv.org/pdf/2503.16419), arXiv 2025.3.20, [repo](https://github.com/Eclipsess/Awesome-Efficient-Reasoning-LLMs)
* [Harnessing the Reasoning Economy: A Survey of Efficient Reasoning for Large Language Models](https://arxiv.org/pdf/2503.24377v1), arXiv 2025.3.31, [repo](https://github.com/DevoAllen/Awesome-Reasoning-Economy-Papers)


## Prompting-guided CoT Compression

> This method directly uses prompts (or few-shot examples) to guide LLMs to generate concise CoT. It does not require model training, but may suffer from bad interpretability, limited CoT compression ratio, and poor reasoning performance.

| Title | Publish | Code |  Method |
|-----|-----|-----|-----|
| [Chain of Draft: Thinking Faster by Writing Less](https://arxiv.org/pdf/2502.18600) | arXiv 2025.2.25 | [code](https://github.com/sileix/chain-of-draft) | Use prompt "with 5 words at most". |
| [Skeleton-of-Thought: Prompting LLMs for Efficient Parallel Generation](https://arxiv.org/pdf/2307.15337) | ICLR 2024 | [code](https://github.com/imagination-research/sot) | Use prompts to generate the skeleton and then complete each point in parallel. Train a router to decide which question to use SoT. |
| [Sketch-of-Thought: Efficient LLM Reasoning with Adaptive Cognitive-Inspired Sketching](https://arxiv.org/pdf/2503.05179) | arXiv 2025.3.7 | [code](https://github.com/SimonAytes/SoT) | Design a prompting method with 3 paradigms and train a router to select. |
| [Break the Chain: Large Language Models Can be Shortcut Reasoners](https://arxiv.org/pdf/2406.06580v1) | arXiv 2024.6.4 | x | Propose zero-shot prompting strategies to encourage the use of shortcuts. Introduce ShortcutQA, a dataset designed to evaluate reasoning through shortcuts. |
| [Token-Budget-Aware LLM Reasoning](https://arxiv.org/pdf/2412.18547) | arXiv 2025.2.17 | [code](https://github.com/GeniusHTX/TALE) | Use LLM to estimate the optimal CoT budget and then use prompt to generate shorter CoT. |
| [How Well do LLMs Compress Their Own Chain-of-Thought? A Token Complexity Approach](https://arxiv.org/pdf/2503.01141) | arXiv 2025.3.3 | [code](https://github.com/Compressed-CoT/compressed-cot) | Find the relationship between CoT length and accuracy and; there is an intrinsic shortest CoT for successfully solving each task. |
| [Stepwise Perplexity-Guided Refinement for Efficient Chain-of-Thought Reasoning in Large Language Models](https://arxiv.org/pdf/2502.13260) | arXiv 2025.2.28 | x | Use PPL to identify critical reasoning steps. Refine demonstration examples in few-shot CoT or finetuning the model using selected examples that include only critical steps. |

## Latent-space CoT Reasoning

> This method directly performs reasoning process in the latent space rather than discrete CoT tokens. It can be very efficient for token saving but may be more difficult to train and lack supervision from immediate CoT tokens during training with next-token prediction.

| Title | Publish | Code |  Method |
|-----|-----|-----|-----|
| [CODI: Compressing Chain-of-Thought into Continuous Space via Self-Distillation](https://arxiv.org/pdf/2502.21074) | arXiv 2025.2.28 | x | Use standard CoT method to supervise the latent reasoning with hidden state distillation. |
| [SoftCoT: Soft Chain-of-Thought for Efficient Reasoning with LLMs](https://arxiv.org/pdf/2502.12134) | arXiv 2025.2.17 | x | Use a small model to generate a few latent tokens as the initial thoughts for LLM inference. Train a linear projection layer to align the representation space between two models.  |
| [Efficient Reasoning with Hidden Thinking](https://arxiv.org/pdf/2501.19201) ![image](https://img.shields.io/badge/Multimodal-orange) | arXiv 2025.1.31 | [code](https://github.com/shawnricecake/Heima) | For Multimodal LLMs, design a Encoder to condense each intermediate CoT into a single thinking token. Also design a Decoder to reconstruct reasoning processes that closely resemble the original CoTs. |
| [Reasoning with Latent Thoughts: On the Power of Looped Transformers](https://arxiv.org/pdf/2502.17416) ![image](https://img.shields.io/badge/Looped_Transformer-blue) | ICLR 2025 | x | Use looped Transformer to generate latent thoughts and simulate CoT reasoning. It can save the parameters but still requires the same total FLOPS to achieve similar performance with multiple loops. |
| [Enhancing Auto-regressive Chain-of-Thought through Loop-Aligned Reasoning](https://arxiv.org/pdf/2502.08482) ![image](https://img.shields.io/badge/Looped_Transformer-blue) | arXiv 2025.2.12 | [code](https://github.com/qifanyu/RELAY) | First train a looped Transformer (3-layer Bert) with explicit CoT alignment in each interation. Then use the looped model to generate new data and enhance auto-regressive CoT model (3-layer decoder) for length generalization ability. However, the model used in this work is too small. |

## Training-internized CoT Compression

> Without the intervention during inference, this method internizes shorter CoTs during training. During inference, LLMs can automatically generate more concise CoTs. However, how to efficiently construct effective training datasets with short CoTs is not easy.

| Title | Publish | Code |  Method |
|-----|-----|-----|-----|
| [C3oT: Generating Shorter Chain-of-Thought without Compromising Effectiveness](https://arxiv.org/pdf/2412.11664) | AAAI 2025 | x | To construct the training dataset, use GPT-4 to compress CoT, and pair "long/short" prompts with original/compressed CoTs. |
| [TokenSkip: Controllable Chain-of-Thought Compression in LLMs](https://arxiv.org/pdf/2502.12067) | arXiv 2025.2.17 | [code](https://github.com/hemingkx/TokenSkip) | Use bidirectional model to compress CoTs with different compression ratios, and pair the prompts (added with compression ratios) with compressed CoTs.  |
| [Token-Budget-Aware LLM Reasoning](https://arxiv.org/pdf/2412.18547) | arXiv 2025.2.17 | [code](https://github.com/GeniusHTX/TALE) | Use a search algorithm to estimate the optimal CoT budget and then use prompt to generate shorter CoT. Use SFT or DPO to train the LLM. |
| [LightThinker: Thinking Step-by-Step Compression](https://arxiv.org/pdf/2502.15589) | arXiv 2025.2.21 | [code](https://github.com/zjunlp/LightThinker) | Dynamically compress intermediate thoughts during reasoning. It is achieved by training the model on when and how to perform compression through data construction, mapping hidden states to condensed gist tokens. |
| [CoT-Valve: Length-Compressible Chain-of-Thought Tuning](https://arxiv.org/pdf/2502.09601) | arXiv 2025.2.13 | [code](https://github.com/horseee/CoT-Valve) | Train extra LoRA weights to control the CoT compression ratio with a hyper-parameter $\alpha$. |
| [Self-Training Elicits Concise Reasoning in Large Language Models](https://arxiv.org/pdf/2502.20122) | arXiv 2025.2.28 | [code](https://github.com/TergelMunkhbat/concise-reasoning) | Finetune LLMs by leveraging self-generated concise reasoning paths obtained by best-of-N sampling and few-shot conditioning. |
| [Stepwise Perplexity-Guided Refinement for Efficient Chain-of-Thought Reasoning in Large Language Models](https://arxiv.org/pdf/2502.13260) | arXiv 2025.2.28 | x | Use PPL to identify critical reasoning steps. Refine demonstration examples in few-shot CoT or finetuning the model using selected examples that include only critical steps. |
| [Can Language Models Learn to Skip Steps?](https://arxiv.org/pdf/2411.01855) | NeurIPS 2024 | [code](https://github.com/tengxiaoliu/LM_skip) | Iteratively train models to generate shorter and accurate reasoning paths. |
| [O1-Pruner: Length-Harmonizing Fine-Tuning for O1-Like Reasoning Pruning](https://arxiv.org/pdf/2501.12570) | arXiv 2025.1.29 | [code](https://github.com/StarDewXXX/O1-Pruner) | Two observations: 1) the relationship between length and accuracy varies significantly across problems, and high accuracy often persists in shorter lengths; 2) there is a consistent trend where shorter response lengths are associated with higher average accuracy rates. Use pre-sampling for each question and then use RL-style fine-tuning to encourage the model to generate shorter reasoning processes.  |

## Inference-time CoT Compression

> This method aims to compress the CoTs process during inference, but may suffer from extra computation to identify redundant CoTs.

| Title | Publish | Code |  Method |
|-----|-----|-----|-----|
| [Entropy-based Exploration Conduction for Multi-step Reasoning](https://arxiv.org/pdf/2503.15848) | arXiv 2025.3.20 | x | Use entropy of reasoning steps to select whether to deepen, expand or stop exploration. |
| [Efficient Long-Decoding Inference with Reasoning-Aware Attention Sparsity](https://arxiv.org/pdf/2502.11147) | arXiv 2025.2.16 | x | Identify a new attention pattern during the decoding stage of reasoning tasks, which is similar to KV cache compression like H2O. |
| [$\phi$-Decoding: Adaptive Foresight Sampling for Balanced Inference-Time Exploration and Exploitation](https://arxiv.org/pdf/2503.13288) | arXiv 2025.3.17 | [code](https://github.com/xufangzhi/phi-Decoding) |  |

## Analysis of CoT Compression 

| Title | Publish | Code |  Method |
|-----|-----|-----|-----|
| [To CoT or not to CoT? Chain-of-thought helps mainly on math and symbolic reasoning](https://arxiv.org/pdf/2409.12183) | arXiv 2024.10.29 | [code](https://github.com/Zayne-sprague/To-CoT-or-not-to-CoT) | CoT gives strong performance benefits primarily on tasks involving math or logic, with much smaller gains on other types of tasks. |
| [How Well do LLMs Compress Their Own Chain-of-Thought? A Token Complexity Approach](https://arxiv.org/pdf/2503.01141) | arXiv 2025.3.3 | [code](https://github.com/Compressed-CoT/compressed-cot) | Find the relationship between CoT length and accuracy and; there is an intrinsic shortest CoT for successfully solving each task. |

## Multimodal CoT Reasoning ![image](https://img.shields.io/badge/Multimodal-orange)
| Title | Publish | Code |  Method |
|-----|-----|-----|-----|
| [Learn to Explain: Multimodal Reasoning via Thought Chains for Science Question Answering](https://arxiv.org/pdf/2209.09513) | NeurIPS 2022 | [code](https://scienceqa.github.io) | A benchmark of Science Question Answering (ScienceQA) that consists of multimodal multiple choice questions with annotations of answers with lectures and explanations. |
| [Interleaved-Modal Chain-of-Thought](https://arxiv.org/pdf/2411.19488) | arXiv 2025.3.17 | [code](https://github.com/jungao1106/ICoT) | Generate interleaved-modal chain-of-thought (ICoT) with paired visual and textual information. Consider that the immediate visual information is usually part of the input image rather than generating extra image, and use an attention-based method to select a few visual tokens. |
| [Efficient Reasoning with Hidden Thinking](https://arxiv.org/pdf/2501.19201) | arXiv 2025.1.31 | [code](https://github.com/shawnricecake/Heima) | For Multimodal LLMs, design a Encoder to condense each intermediate CoT into a single thinking token. Also design a Decoder to reconstruct reasoning processes that closely resemble the original CoTs. |
