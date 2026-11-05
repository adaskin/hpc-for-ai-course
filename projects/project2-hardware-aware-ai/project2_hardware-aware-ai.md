# YZM 513 -- Project 2 (Final Project): Hardware-Aware AI Systems

## Weeks 8-15 | Proposal: Week 10 | Report and code: Week 14 | Presentations: Weeks 14-15

**Format:** Groups of 2  
**Deliverables:** One-page proposal &middot; Report (6-8 pages, PDF) &middot; Code repository (GitHub) &middot; In-class presentation (15 min + 5 min Q&A)  
**Weight:** 15% of course grade (10% report and code + 5% presentation)

---

## 0. The Problem

Every AI system runs on real hardware with a fixed budget: one GPU, a limited amount of memory, a few hours of compute. Within that budget, the same model can be slow or fast, run out of memory or fit, waste tokens or answer directly.

Your task is to pick one AI system, **measure** how it behaves on your hardware, **change** it in a controlled way, and **explain** what happened, with evidence.

> How can we improve the speed, memory use, quality, or behaviour of an AI system on a fixed hardware and compute budget?

Choose one of the tracks in Section 4 or propose your own. The tracks are starting points, not checklists: adapt the model, dataset, workload and methods to your interests and to the hardware you have.

**You do not need a dramatic speedup.** A careful negative result, a failed replication, or a well-supported explanation of why an optimisation did *not* help is a good project. What we grade is the quality of your evidence and reasoning, not the size of your improvement.

---

## 1. What Your Project Must Define

Whatever track you choose, your proposal and report must state:

| Item | Example |
|------|---------|
| **System** | Qwen2.5-0.5B-Instruct, served with Hugging Face `transformers` |
| **Workload** | 200 prompts of 128-1,024 tokens, 256 generated tokens each |
| **Hardware** | Colab T4 (16 GB), PyTorch 2.x, CUDA 12.x |
| **Baseline** | fp16, batch size 1, eager mode |
| **Question** | How do batch size and `torch.compile` change throughput and tail latency? |
| **Interventions** (at least 2) | (1) batch sizes 1-16, (2) `torch.compile`, (3) both |
| **Metrics** | Tokens/s, p50 and p95 latency, peak memory, GPU utilisation, output quality |

---

## 2. AI Usage Policy

AI assistants (ChatGPT, Claude, Copilot, Kimi, Qwen, etc.) are **encouraged** for coding, debugging, experimental design, data analysis, literature search and writing. Treat them as research tools:

- Test the code they generate and verify the claims and citations they give you.
- Every number in your report must come from **your own runs**, and the logs or scripts that produce it must be in your repository.
- You must be able to explain every figure and every code block you submit.
- **Penalty:** -15 points if you cannot explain a figure or code block you submitted.

In Track D, an AI system is also the *object* of study; Section 4 lists the extra logging rules for that track.

---

## 3. Constraints

- **Hardware.** Everything must run on **free Google Colab (T4 GPU)** or hardware you already have. If you use anything else (a lab GPU, Kaggle, your own machine), say so in the proposal and report its specifications. Compare conditions **only on the same hardware**.
- **Reproducibility.** Include `requirements.txt` (or an environment file) and a `README.md` with run instructions. Record library versions and the output of `nvidia-smi`.
- **Fair benchmarks.** Warm-up runs, **at least 5 timed runs**, report the median and the spread (min-max or interquartile range). Synchronise the GPU before you stop the clock (Week 6, Week 8).
- **Correctness first.** An optimised version must produce the same results as the baseline (within a stated tolerance), or you must measure how much quality it loses.
- **Live demo.** Show a short live demo (about 2 minutes). Colab GPUs are not guaranteed, so bring a recorded backup.

### Know your GPU: the Colab T4

| Property | NVIDIA T4 (Turing, compute capability 7.5) |
|---|---|
| Memory | 16 GB GDDR6, about **320 GB/s** |
| Compute | 40 SMs, 2,560 CUDA cores; about 8 TFLOP/s fp32, about 65 TFLOP/s fp16 with tensor cores |
| Supported well | fp16 autocast + `GradScaler`, int8 / 4-bit (bitsandbytes), `torch.compile`, Triton, Numba CUDA, CUDA C++ |
| **Not** supported natively | bf16, FlashAttention-2, fp8: these need Ampere or newer GPUs |

Plan around the T4. Many tutorials assume bf16 or FlashAttention-2; on the T4 use fp16 with loss scaling, and PyTorch's built-in `scaled_dot_product_attention`.

---

## 4. Tracks

| Track | Topic | Builds on |
|:-:|---|---|
| **A** | Memory-efficient training and fine-tuning | Profiling (Week 9), data pipelines (Week 10), distributed training (Weeks 11-12), mixed precision |
| **B** | Optimising model inference | Inference and serving (Week 13): quantisation, `torch.compile`, KV cache, batching |
| **C** | Custom GPU kernels | GPU architecture (Week 6), CUDA kernels (Week 8), memory hierarchy and profiling (Week 9) |
| **D** | AI-directed model diagnosis and weight editing | Measurement (Week 1), GPU inference cost (Weeks 6, 13) |
| **E** | Your own project | Any part of the course |

Some of these tools are taught **after** the proposal deadline. You are not expected to know them yet: choose by interest, and the lectures will catch up.

### Track A: Memory-Efficient Training

**Goal.** Train or fine-tune a model under limited GPU memory while keeping acceptable throughput and task quality.

Possible topics:

- Full fine-tuning of a small language or vision model vs. LoRA / QLoRA on a larger one
- Mixed precision (fp16 autocast + `GradScaler`)
- Gradient checkpointing and gradient accumulation
- Memory-efficient optimisers (e.g. 8-bit Adam)
- `DataLoader` settings and CPU-GPU transfer overlap (`num_workers`, `pin_memory`, prefetching)
- CPU offloading, where it makes sense

If you compare full fine-tuning with parameter-efficient fine-tuning, state exactly which parameters are trained and how many.

**Example question.**
> How do 4-bit quantisation, gradient checkpointing and micro-batch size affect peak memory, throughput and downstream accuracy during QLoRA fine-tuning?

**Measure:** peak GPU memory, samples or tokens per second, training and validation loss, downstream task quality, GPU utilisation or idle time, time spent in data loading and transfers.

Distributed methods (DDP, FSDP, ZeRO) are welcome but not required. Colab has one GPU; a 2-process DDP run on one machine (`gloo` backend, as in the distributed-training lab) or Kaggle's free 2 x T4 notebooks are acceptable ways to study them.

### Track B: Optimising Inference

**Goal.** Improve the latency, throughput or memory use of a model serving a defined workload.

Possible topics:

- Weight quantisation (int8, 4-bit)
- `torch.compile` or other graph compilation
- KV-cache size and memory, static vs. dynamic cache
- Static vs. dynamic batching; asynchronous request scheduling
- Continuous batching with an existing serving framework (check that it supports the T4 before you commit)
- Speculative decoding with a small draft model
- Effects of prompt length and generation length
- Quality-latency trade-offs

You do not need a production server. A controlled local benchmark with concurrent clients is enough.

**Example question.**
> How do quantisation and request batch size affect decode throughput, tail latency, memory use and output quality for a small open-weight language model?

**Measure:** time to first token, prefill and decode throughput, end-to-end latency including p95, requests or tokens per second, peak memory, output quality or task accuracy.

**Always report** prompt lengths, generation lengths, batch sizes, decoding settings, the warm-up procedure and the number of runs.

### Track C: Custom GPU Kernels

**Goal.** Implement a GPU operation yourself and explain its performance against existing implementations.

Suitable operations:

- RMSNorm or LayerNorm
- SwiGLU or another composite activation
- Applying RoPE
- Softmax or another small reduction
- A fused chain of element-wise operations (Week 8, Part VI)
- A specialised operation from your own research

Use Numba CUDA, CUDA C++, Triton, or another suitable framework.

You do not need to beat cuBLAS. Matching `torch.compile`, beating the eager PyTorch version, or a well-explained negative result is enough.

**Example question.**
> How does fusing a chain of element-wise operations change memory traffic and run time across tensor sizes?

**Required comparisons:**

1. A straightforward PyTorch (eager) baseline
2. Your custom kernel
3. A `torch.compile` version, where practical

**Required evidence:**

- Numerical correctness tests against the baseline (state your tolerance)
- Run time over **several input sizes**, from small to larger than the L2 cache
- Achieved bandwidth (GB/s) or FLOP/s compared with the T4's peak: a roofline placement of your operator
- A profiler trace (`torch.profiler`, Nsight Systems or Nsight Compute)
- An explanation in terms of bandwidth, occupancy, launch overhead or compute limits

If your operation is used in training, also check its backward pass and gradients (e.g. `torch.autograd.gradcheck` on small inputs).

### Track D: AI-Directed Model Diagnosis and Weight Editing

**Goal.** Investigate whether an AI system can diagnose and repair a failure in another model through targeted interventions, including direct weight edits, and what that repair costs and buys in compute.

**Background.** In *Sparks of Weight-Space Programming* (Takehi, Teoh, Zhang and Khattab, MIT CSAIL, November 2026), an LLM agent (OpenAI's GPT-6 Astra) inspected the weights of Qwen3.5-0.8B with Python and edited them **without gradient-based training**. Its edit changed 131,072 weights (about 0.016%) in layers 15 and 19 and raised MATH-500 accuracy from 7.9% to 48.1% under the authors' settings. Read the post carefully before you build on it:

- The gain came mostly from **removing repetitive looping**: before the edit, 93% of MATH-500 answers ran to the token limit. In the authors' words, fixing loops "lets the model finish answers it could already work out"; it did not add mathematical knowledge.
- On Gemma 3 270M, which rarely loops, the agent could **not** improve math accuracy.
- The post does not compare the edit with simpler fixes, such as changing decoding settings.

That last gap is a good project. The edited model and the complete logs are public (see References), so you can start from the published edit instead of reproducing the search.

**Why it belongs in this course.** A looping model burns GPU time on tokens that do not help. Measure the repair as a systems change: tokens per answer, latency per answer, and **accuracy per GPU-second**, not only accuracy. A weight edit adds no inference cost; an activation-steering hook or a longer prompt does. Measure that overhead.

Possible projects:

- Compare the published edit with decoding-only baselines (repetition or presence penalty, a lower token limit, a thinking budget, turning thinking off)
- Compare weight editing with activation steering or with a small LoRA
- Give an AI agent a controlled failure in a small model and a fixed budget, and test whether it finds and repairs it
- Test whether an edit generalises beyond the problems used to develop it
- Measure collateral damage on unrelated tasks
- Compare AI-proposed edits with simple human-designed interventions

You may use any AI system you have access to as the investigating agent. You do not need the original model or its headline result.

**Example question.**
> Does the published weight edit reduce tokens per answer and latency more than a decoding-only fix, and at what cost to accuracy on held-out problems and unrelated tasks?

**A manageable design:**

1. Choose a small model and define a measurable failure (e.g. the fraction of answers that hit the token limit).
2. Split problems into a **development set** (used by you or the agent) and a **held-out test set** (used only for the final numbers).
3. Measure the baseline behaviour.
4. Give the investigating agent a **fixed budget** of forward passes or experiments, and the same budget to every agent you compare.
5. Apply its edit or other intervention.
6. Compare with at least one simpler baseline.
7. Measure the target behaviour, regressions on unrelated tasks, and inference-time memory and compute overhead.

**Track D rules:**

- **Fit the T4.** The original work allowed 32,768-token answers and sampled each problem several times, which is far beyond a free Colab session. Cap the generation length (e.g. 2,048-4,096 tokens), use a subset (e.g. 100 problems), and state your settings. Your absolute numbers will not match the post; compare conditions under **your own fixed settings**.
- **Log the agent.** Save the full transcript, every piece of code it ran, and the number of forward passes it used.
- **Be consistent.** If you impose a "no gradient training" condition, apply it to every condition that claims to follow it.
- **Sandbox the agent.** Run it in a Colab or container session with no access to your accounts, passwords or API keys beyond what it needs.

A negative result, such as finding that a decoding change works as well as the weight edit, is a useful outcome.

### Track E: Your Own Project

Propose a different project connected to the course, or a combination of the tracks above. Examples:

- Efficient multimodal inference
- Memory-efficient retrieval or embedding search on the GPU (a GPU follow-up to Project 1)
- Energy-aware optimisation (measure power with `nvidia-smi`)
- Compiler-assisted kernel optimisation
- Reliability and reproducibility of GPU benchmarks
- Model editing for a specialised domain
- AI agents that tune training or inference settings automatically
- A performance problem from your own thesis work

Your proposal must state a research question, a baseline, an experimental plan and metrics. **Talk to the instructor before committing to a computationally expensive design.**

---

## 5. Common Requirements (All Tracks)

### 5.1 A Baseline

Run a baseline **before** you optimise anything. Describe the hardware, software versions, model, dataset, workload and configuration.

### 5.2 A Controlled Experiment

Change a small number of variables at a time, and measure their effects separately where you can. At minimum:

- One baseline condition
- **At least two** intervention or comparison conditions
- The same evaluation procedure for every condition
- Repeated measurements wherever results vary between runs

### 5.3 Required Measurements

For the baseline and every intervention, report:

| Measurement | How |
|---|---|
| **Wall-clock time** (or latency / throughput) | CUDA events or `time.perf_counter()` after `torch.cuda.synchronize()` |
| **Peak GPU memory** | `torch.cuda.max_memory_allocated()` after `torch.cuda.reset_peak_memory_stats()` |
| **GPU utilisation** | `nvidia-smi` sampling, or a `torch.profiler` trace showing busy and idle time |
| **Quality** | Accuracy, loss, or another task metric on a **held-out** set |

### 5.4 Profiling or Diagnostic Evidence

Use the evidence that fits your question. Not every project needs every kind:

- GPU profiler traces (`torch.profiler`, Nsight Systems, Nsight Compute)
- Peak-memory breakdowns
- `DataLoader` timelines
- A roofline analysis of one operator
- Activation or weight analysis
- Failure-category analysis
- Regression tests across model behaviours

### 5.5 Critical Analysis

Discuss:

- What improved, and what did not
- Whether your evidence supports your explanation
- Trade-offs between speed, memory, quality and complexity
- Possible measurement errors and threats to validity
- Whether your findings are likely to generalise (another model, size, or GPU)
- What you would investigate next

Do not assume that quantisation gives a proportional speedup, that kernel fusion removes every bottleneck, or that a targeted model edit improves general capability. Test it.

---

## 6. Proposal (Week 10, one page)

Submit one page (PDF or Markdown in your repository) with:

```text
Title:
Members:
Track (A-E):
Research question (one sentence):
System: model / dataset / workload, and why it matters
Hardware: e.g. Colab T4 16 GB, plus anything else you will use
Baseline: what you will run first, and how you will measure it
Interventions (at least 2): what you will change, one variable at a time
Metrics: time, memory, utilisation, quality
Compute plan: estimated GPU-hours; what you will cut if time runs short
Risks and fallback: what could block you, and your plan B
```

Before you submit, **run the baseline once on the T4** (or your hardware) so you know the model fits and the workload finishes in reasonable time. You will get feedback within a week.

---

## 7. Required Figures (4-6 total)

Every figure must have a caption that states a **conclusion**, not just a description.

| # | Figure | What it shows |
|---|--------|---------------|
| 1 | Baseline profile (trace, memory breakdown, or failure analysis) | Where the time, memory or failures are |
| 2 | Main result: metric vs. condition, with spread (error bars or box plots) | What each intervention changed |
| 3 | Trade-off plot (e.g. speed vs. quality, memory vs. throughput) | What you pay for each improvement |
| 4 | Scaling plot (vs. input size, batch size, sequence length or budget) | Where the improvement holds and where it breaks |
| 5-6 | *(optional)* Roofline placement, profiler comparison, regression table | Supporting evidence for your explanation |

> X "Figure 2: Latency for different batch sizes"  
> OK "Figure 2: Batch size 8 triples decode throughput (41 to 128 tokens/s), but p95 latency doubles; beyond 8 the T4 runs out of KV-cache memory."

---

## 8. Report Structure (6-8 pages, excluding references and appendices)

| Section | Pages | Content |
|---------|:-----:|---------|
| **1. Introduction** | 0.5-1 | Motivation and research question |
| **2. System and Workload** | 0.5-1 | Model, data, workload, hardware and software environment |
| **3. Baseline and Method** | 1-1.5 | Baseline, interventions, controlled-experiment design, measurement procedure |
| **4. Results** | 2-3 | Main measurements with figures and tables |
| **5. Profiling and Diagnosis** | 0.5-1 | Evidence explaining *why* the results look the way they do |
| **6. Discussion and Limitations** | 1 | Trade-offs, threats to validity, generalisation |
| **7. Related Work** | 0.5 | How your question relates to existing work and tools |
| **8. Conclusion and Future Work** | 0.5 | What you found; what you would do next |
| **Appendix** | -- | Configuration files, extra tables, raw results, agent logs (Track D) |

---

## 9. Presentation (Weeks 14-15, 15 min + 5 min Q&A)

Tell an **evidence story**:

1. **The question:** "We wanted to know whether X makes Y faster / smaller / better on a T4."
2. **The baseline:** "Here is what we measured first, and where the time or memory went."
3. **The experiments:** "We changed A, then B. Here is what each did."
4. **The explanation:** "This happened because of [bandwidth / memory / launch overhead / looping / ...]. Here is the evidence."
5. **The limits:** "This holds for ...; we do not know about ..."

End with one slide: **"What surprised us"** and **"What we would do with more time."**

---

## 10. Grading

### Report and Code (20 points)

| Criterion | Weight | Excellent |
|-----------|:------:|-----------|
| **Research question and experimental design** | 20% | Clear, answerable question; baseline and at least 2 controlled interventions |
| **Technical implementation** | 25% | Correct, working, reproducible code; correctness checked against the baseline |
| **Measurement, profiling and evaluation** | 30% | Fair benchmarks, required measurements, appropriate profiling or diagnostic evidence |
| **Interpretation and critical analysis** | 15% | Explanations backed by evidence; trade-offs and threats to validity discussed honestly |
| **Report quality and related work** | 10% | Well structured, conclusion-first figure captions, relevant related work |

### Presentation (10 points)

| Criterion | Weight | Excellent |
|-----------|:------:|-----------|
| Explanation of figures | 30% | Every figure explained with insight |
| Code explanation | 25% | Can explain any code line by line |
| Technical depth and Q&A | 25% | Handles questions confidently; connects results to course concepts |
| Demo and storytelling | 20% | Smooth demo, clear narrative |

### Bonus (up to +3 points)

- Repeat your main comparison on a second GPU type and explain the differences
- A roofline analysis of your key operator, with measured and peak numbers
- A reusable benchmark script others could run on their own model

### Penalties

| Penalty | Reason |
|---------|--------|
| -15 pts | Cannot explain a figure or code block during the presentation |
| -10 pts | Unfair benchmarks (conditions on different hardware, no warm-up, single runs, no GPU synchronisation) |
| -10 pts | Reported numbers that no submitted script or log reproduces |
| -5 pts | Missing `requirements.txt`, `README.md` or code |

---

## 11. Practical Tips

**Time GPU code correctly** (Week 6, Week 8):

```python
import torch

def gpu_time_ms(fn, warmup=3, runs=10):
    for _ in range(warmup):                    # JIT, autotuning, caches
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(runs):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()               # wait for the GPU before reading the timer
        times.append(start.elapsed_time(end))
    times.sort()
    return times[len(times) // 2], times[0], times[-1]   # median, min, max

torch.cuda.reset_peak_memory_stats()
median_ms, min_ms, max_ms = gpu_time_ms(lambda: model(x))
peak_gb = torch.cuda.max_memory_allocated() / 1e9
```

**Sample GPU utilisation** while your job runs (one line per second, saved to a file):

```bash
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv -l 1 > gpu_log.csv &
```

**Colab:**

- Free GPU time is limited and sessions can disconnect. Save checkpoints, logs and results to Google Drive as you go.
- Develop on small inputs (or CPU), then run the full measurements in one GPU session.
- Record `torch.__version__`, `torch.version.cuda` and `nvidia-smi` output at the start of every results notebook.
- Pin library versions in `requirements.txt`; Colab updates its images.

**Scope:** start small. One model, one workload, two or three interventions, measured well, is a better project than ten interventions measured badly.

---

## 12. FAQ

**Q: Can we use our own or a lab GPU instead of Colab?**  
A: Yes. Report its specifications, and run every condition you compare on the same GPU. Your code should still run on a T4 at a smaller scale, so others can reproduce it.

**Q: Can our project come from our thesis research?**  
A: Yes, through Track E, as long as it has a baseline, at least two controlled interventions, and the required measurements. Check with the instructor first.

**Q: Our optimisation made things slower. Is that a failed project?**  
A: No. Measure it carefully and explain *why* it is slower. That is a valid, graded result.

**Q: Do we have to beat a library (cuBLAS, PyTorch, a serving framework)?**  
A: No. You have to compare against it fairly and explain the difference.

**Q: In Track D, can the agent be a chat interface instead of an API?**  
A: Yes, if you log the full conversation and run every piece of code it proposes yourself, within the same fixed budget.

**Q: Can we work in a group of 3 or alone?**  
A: Groups of 2 by default. Contact the instructor if this does not work for you.

**Q: How much compute should we plan for?**  
A: Plan your main experiments to fit in a few Colab sessions (a few GPU-hours each). If your design needs much more, shrink it or discuss it with the instructor before the proposal.

---

## 13. Timeline

| Week | Milestone |
|------|-----------|
| Week 8 | Brief released. Form groups, choose a track |
| Week 9 | Baseline running on the T4; draft proposal |
| **Week 10** | **Proposal due** (one page). Feedback within a week |
| Week 11 | Interventions implemented; first measurements |
| **Week 12** | **Check-in** (5 min per group): baseline, first results, plan for the remaining runs |
| Week 13 | Controlled comparisons finished; results frozen; report written |
| **Week 14** | **Report and code due** (before the session). Presentations, part 1 |
| Week 15 | Presentations, part 2 |

---

## 14. Submission Checklist

**Proposal (Week 10):**
- [ ] One page, all fields of Section 6 filled in
- [ ] Baseline run once on the target hardware

**Report:**
- [ ] PDF, 6-8 pages excluding references and appendices
- [ ] Research question, system, workload, hardware and software versions
- [ ] Baseline and at least 2 interventions, measured the same way
- [ ] Wall-clock time, peak memory, GPU utilisation and a quality metric for every condition
- [ ] Profiling or diagnostic evidence
- [ ] 4-6 figures with conclusion-first captions
- [ ] Limitations, threats to validity and related work

**Code:**
- [ ] GitHub repository link
- [ ] `requirements.txt` and `README.md` with instructions to reproduce
- [ ] All figures generated by code
- [ ] Raw results (CSV or JSON) and logs, including agent transcripts for Track D

**Presentation:**
- [ ] 15-minute slides ending with "What surprised us" / "What we would do with more time"
- [ ] Live demo tested, plus a recorded backup
- [ ] Every figure and code block rehearsed

---

## 15. References and Starting Points

**Track D:**
- R. Takehi, J. Teoh, A. Zhang, O. Khattab. [*Sparks of Weight-Space Programming: LLMs can natively read & write neural network weights*](https://rikiyat.github.io/blog/weight-space-programming/). MIT CSAIL blog post, 2 November 2026.
- Complete experiment logs and agent transcripts: [github.com/RikiyaT/lm_understand-logs](https://github.com/RikiyaT/lm_understand-logs)
- The edited model: [huggingface.co/RikiyaT/Qwen3.5-0.8B-weight-space-programming](https://huggingface.co/RikiyaT/Qwen3.5-0.8B-weight-space-programming)
- K. Meng et al. *Locating and Editing Factual Associations in GPT* (ROME), NeurIPS 2022. [arXiv:2202.05262](https://arxiv.org/abs/2202.05262)
- A. M. Turner et al. *Steering Language Models With Activation Engineering*. [arXiv:2308.10248](https://arxiv.org/abs/2308.10248)

**Tracks A and B:**
- P. Micikevicius et al. *Mixed Precision Training*, ICLR 2018. [arXiv:1710.03740](https://arxiv.org/abs/1710.03740)
- T. Chen et al. *Training Deep Nets with Sublinear Memory Cost* (gradient checkpointing). [arXiv:1604.06174](https://arxiv.org/abs/1604.06174)
- E. J. Hu et al. *LoRA: Low-Rank Adaptation of Large Language Models*. [arXiv:2106.09685](https://arxiv.org/abs/2106.09685)
- T. Dettmers et al. *QLoRA: Efficient Finetuning of Quantized LLMs*. [arXiv:2305.14314](https://arxiv.org/abs/2305.14314)
- S. Rajbhandari et al. *ZeRO: Memory Optimizations Toward Training Trillion Parameter Models*, SC 2020. [arXiv:1910.02054](https://arxiv.org/abs/1910.02054)
- W. Kwon et al. *Efficient Memory Management for LLM Serving with PagedAttention* (vLLM). [arXiv:2309.06180](https://arxiv.org/abs/2309.06180)
- Y. Leviathan et al. *Fast Inference from Transformers via Speculative Decoding*, ICML 2023. [arXiv:2211.17192](https://arxiv.org/abs/2211.17192)

**Track C:**
- S. Williams, A. Waterman, D. Patterson. *Roofline: An Insightful Visual Performance Model for Multicore Architectures*, CACM 2009.
- [Triton tutorials](https://triton-lang.org/main/getting-started/tutorials/index.html) (fused softmax, layer norm)
- [PyTorch profiler recipe](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)
- Course materials: Week 8 slides (`lectures/08cuda-kernel-numba-cuda.md`) and `lectures/week08/`

---

*Measure first, change one thing at a time, and let the evidence tell the story.*

**YZM 513 -- Istanbul Medeniyet Universitesi -- Fall 2026**
