# Executive Summary

## Intelligence Density

This repo argues that the next major practical improvements in AI should come from increasing `intelligence density`, not from scaling parameter count alone.

In this context, intelligence density means:

- more capability per model byte
- more capability per GB of active memory
- more capability per unit of active compute
- more deployable value per unit of infrastructure cost

## Core Claim

The highest-leverage path to better AI systems is to improve:

`quality / storage`

`quality / memory`

`quality / compute`

rather than optimizing only for absolute benchmark quality.

## Best Current Algorithmic Levers

1. Distillation
- Best for improving quality per byte
- Strongest when retraining is available

2. Quantization
- Best direct way to shrink checkpoints
- 4-bit is the practical default in many deployments

3. Codebook / Additive Quantization
- Best for extreme compression
- Strong when file size is the primary constraint

4. Low-Rank Factorization
- Best structural compression method
- Useful when you control training or finetuning

5. Sparse / Conditional Computation
- Best for improving active compute efficiency
- More useful for serving density than pure checkpoint shrinkage

6. KV-Cache Compression
- Best runtime memory lever for long-context systems
- Essential for API-scale inference efficiency

## Best Stack By System Type

### LLMs

1. distillation
2. 4-bit quantization
3. KV-cache compression
4. codebook quantization when distribution size matters

### Vision Models

1. low-rank or structured compression
2. INT8 / INT4 quantization
3. selective structured pruning

### On-Device Models

1. aggressive quantization
2. distillation
3. architecture redesign for bandwidth and thermal limits
4. codebook quantization if tooling supports it

### API-Scale Systems

1. 4-bit / 8-bit quantization
2. KV-cache compression
3. distillation for product tiering
4. sparse routing where infrastructure supports it

## Strategic Implication

If intelligence density improves faster than model scale grows, AI systems become:

- cheaper to deploy
- easier to replicate
- more viable on edge hardware
- more profitable to serve
- less dependent on brute-force infrastructure growth

## Bottom Line

The best practical AI systems will not necessarily be the largest ones.

They will be the systems that convert storage, memory, and active compute into useful behavior most efficiently.

For the full argument and mathematical analysis, see [WHITEPAPER.md](./WHITEPAPER.md).
