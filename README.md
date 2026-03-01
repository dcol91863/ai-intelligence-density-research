# Intelligence Density

Research notes on improving AI system efficiency by file size, memory footprint, and deployable capability per parameter byte.

This repo uses the term `intelligence density` as shorthand for:

- more capability per stored weight
- more useful behavior per byte on disk
- more deployable quality per unit of memory and compute

That framing is close to a recent public discussion around increasing "intelligence density" in AI systems: making models more capable without scaling storage and runtime costs linearly. In practice, that mostly means better compression, better parameter use, and better conditional computation.

## Executive Summary

If the goal is better model quality per file size, the current highest-leverage algorithmic directions are:

1. Distillation into smaller students
2. 4-bit and mixed-precision quantization
3. Codebook / additive quantization for more aggressive compression
4. Low-rank factorization and weight sharing when retraining is allowed
5. Sparse or routed computation when deployment cares about active rather than total parameters
6. KV-cache compression for long-context inference systems

The practical rule:

- safest compression win: 4-bit quantization
- best quality per byte: distillation plus 4-bit quantization
- smallest deployable checkpoints: codebook / additive quantization

## Why File Size Still Matters

File size is not just a storage problem.

It affects:

- model download and replication time
- edge and mobile deployment viability
- RAM / VRAM residency
- multi-model serving density
- checkpoint portability
- inference cost when memory bandwidth dominates

For many real deployments, "more intelligence per GB" is a better optimization target than raw benchmark score.

## Current Algorithmic Directions

### 1. Quantization

Quantization remains the most effective direct method for shrinking model files.

- FP16/BF16 to INT8 is usually about a 2x checkpoint reduction
- FP16/BF16 to 4-bit is usually about a 4x checkpoint reduction
- modern schemes preserve quality much better than naive uniform quantization

Best current direction:

- 4-bit or mixed W4A8-style quantization with outlier-aware scaling and error compensation

Why it works:

- weights are highly redundant
- not all channels require the same precision
- most quality loss comes from a small subset of sensitive parameters

Use when:

- you want immediate size reduction without changing architecture

### 2. Distillation

Distillation is the strongest way to improve capability per byte, not just bytes per model.

Instead of compressing the same large model harder, train a smaller student to imitate a stronger teacher.

Why it matters:

- smaller models often retain more useful task behavior than equivalently sized raw pruned models
- compression after distillation usually performs better than compression alone

Use when:

- retraining is possible
- you want the best quality-to-size tradeoff

### 3. Codebook / Additive Quantization

For extreme compression, codebook methods currently outperform simpler low-bit approaches on size efficiency.

Main idea:

- represent weights through learned codebooks and compact indices instead of direct scalar storage

Benefits:

- can push toward roughly 2-bit effective storage
- often higher quality than naive 2-bit scalar quantization

Tradeoff:

- more complex kernels and tooling
- less convenient than plain INT8 / INT4 serving

Use when:

- file size is the dominant constraint

### 4. Low-Rank Factorization and Weight Sharing

If retraining is allowed, many transformer and vision model layers can be reparameterized more compactly.

Typical methods:

- matrix factorization
- shared blocks
- grouped parameter tying

Benefits:

- structural reduction rather than just numerical compression
- can combine cleanly with quantization

Use when:

- you control training or finetuning
- you want architecture-level size reduction

### 5. Sparse and Conditional Computation

Sparse models and mixture-of-experts improve active efficiency more than total checkpoint size.

That means:

- total weights may still be large
- only part of the model activates per token or example

Why it still matters for intelligence density:

- capacity per active byte and per active FLOP can improve substantially

Use when:

- serving cost matters more than raw checkpoint size

### 6. KV-Cache Compression

This does not shrink the checkpoint much, but it can dominate real memory use in long-context systems.

Important for:

- chat systems
- agent loops
- retrieval-heavy inference
- API serving

Use when:

- runtime memory and throughput are bottlenecks

## Recommended Strategy By System Type

## 1. LLMs

Best near-term recipe:

- distill to a smaller dense student when possible
- quantize to 4-bit
- use KV-cache compression for long-context serving
- consider codebook quantization if download size is critical

Recommended priority:

1. distillation
2. 4-bit quantization
3. KV-cache compression
4. additive / codebook quantization

## 2. Vision Models

Best near-term recipe:

- low-rank compression or structured reparameterization
- INT8 or 4-bit quantization
- channel pruning only when validated carefully

Why:

- vision backbones often tolerate structural compression well
- activation layout and hardware support matter more than in some LLM setups

Recommended priority:

1. low-rank factorization
2. INT8 / INT4 quantization
3. selective structured pruning

## 3. On-Device Models

Best near-term recipe:

- aggressive quantization first
- distill into a smaller architecture
- prefer architectures designed for memory bandwidth constraints

Why:

- storage, RAM, and thermal limits dominate
- serving simplicity matters

Recommended priority:

1. 4-bit or mixed low-bit quantization
2. distillation
3. architecture redesign for bandwidth efficiency
4. codebook quantization if tooling supports it

## 4. API-Scale Inference Systems

Best near-term recipe:

- optimize for serving density, not just checkpoint size
- use quantization plus KV-cache compression plus batching-aware kernels
- use sparse routing only if infra is mature enough

Why:

- operational memory often costs more than static disk storage
- active memory footprint controls throughput

Recommended priority:

1. 4-bit / 8-bit quantization
2. KV-cache compression
3. distillation for tiered model offerings
4. sparse or MoE routing where infra justifies it

## What Is Less Promising

These can still help, but they are usually weaker first moves:

- naive unstructured magnitude pruning
- compressing without any teacher recovery
- chasing ever-lower bitwidth without hardware support

Compression that looks good on parameter count alone often underperforms on real quality-per-byte.

## A Practical Research Agenda

If you want to push intelligence density forward today, the most promising combined agenda is:

1. architecture search for compact students
2. teacher-student distillation
3. sensitivity-aware 4-bit or codebook quantization
4. runtime-aware memory compression, especially KV cache
5. evaluation on capability per GB rather than capability alone

## References

See [SOURCES.md](./SOURCES.md).
