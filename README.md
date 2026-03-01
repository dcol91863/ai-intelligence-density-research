# Intelligence Density

A public research repo on improving AI capability per byte, per unit of memory, and per unit of active compute.

## Primary Documents

- [WHITEPAPER.md](./WHITEPAPER.md): full whitepaper with mathematical analysis
- [SOURCES.md](./SOURCES.md): references for the claims and framing

## Scope

This repo argues that AI progress should be evaluated not only by absolute benchmark quality, but also by `intelligence density`:

- capability per model byte
- capability per GB of active memory
- capability per active FLOP
- deployable quality per unit of infrastructure cost

The whitepaper covers:

1. LLMs
2. vision models
3. on-device models
4. API-scale inference systems

It also includes a mathematical analysis of how the main algorithmic levers improve these density metrics:

- distillation
- quantization
- codebook / additive quantization
- low-rank factorization
- sparse / conditional computation
- KV-cache compression

## Related Context

- [Super AI Safety](https://www.superaisafety.com)

## Repository Goal

The goal is to make the case that the next wave of practical AI improvement should come from higher intelligence density, not from scale alone.
