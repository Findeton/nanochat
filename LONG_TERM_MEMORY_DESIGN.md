# Long-Term Memory for NanoChat

## Purpose

NanoChat already has strong short-term working memory through ordinary causal attention over the current context window.

The long-term memory problem is narrower and more concrete:

\[
p_\theta(y \mid x, M) \approx p_\theta(y \mid x, H)
\]

where:

- `x` is the current query
- `H` is relevant past context that is no longer in the prompt
- `M` is persistent memory written from earlier interactions
- `y` is the answer sequence

The goal is not "add memory somehow". The goal is to make persistent memory behave like missing context without turning the model into a pile of special-case branches.

This document describes:

- what we learned from the previous architectures and training runs
- what the current architectural direction is
- what changed in the implementation
- how the training harness should be used on a real GPU run

## Current Verdict

Three conclusions are solid now.

1. Persistent latent memory is real.

Earlier `r18`-style runs already showed delayed narrow factual recall without querying an external retrieval source. So the project does not need to fall back to a database-like solution just to get memory at all.

2. Better losses alone do not fix a bad memory interface.

Several recipes improved local token losses while behavioral recall stayed at zero. That means loss quality matters, but architecture still determines whether memory can enter generation in a useful way.

3. The right architectural family is unified associative attention, but the previous instantiations were still too diffuse.

The current direction remains:

- persistent memory should be latent runtime state, not baked into weights
- context-window tokens and persistent memory should be read through the same kind of associative mechanism
- memory should help selectively, not act as a global fuzzy side-channel

What failed recently was not the high-level idea. What failed was a specific implementation that kept memory too dense, too broad, and too easy to route harmfully.

## What We Learned Empirically

### Early Success Matters

The initial success condition was important:

- two-session recall worked in a narrow setting
- specific facts could be recalled from persistent latent memory
- this happened without an external retrieval service

That result constrains the design space. It suggests the project should stay close to learned latent memory rather than jumping to explicit symbolic storage by default.

### Loss-Only Iterations Were Necessary but Not Sufficient

Weighted CE, fact-margin losses, and related recipes were useful because they exposed the right failure mode:

- memory sometimes improved local fact-token probabilities
- but generation still failed

So the problem was not simply "optimize harder". It was "the model is receiving memory in the wrong form".

### Plan/Payload and Dual-Copy Experiments Isolated the Bottlenecks

Those lines taught two important things:

- the opening answer token and the first fact token are disproportionately important
- later fact subtokens can improve even while the overall answer remains wrong

This means memory can carry useful signal while still failing to behave like missing context at the sequence level.

### The Recent Memory-Token Architecture Exposed a Different Failure

The most recent sparse-memory-token line fixed two old problems:

- memory gates were no longer effectively shut
- banks were no longer trivially rank-1 collapse

But the saved checkpoint still failed behaviorally:

- focused exact/core-fact eval remained `0.0 / 0.0`
- generation degenerated into repetitive punctuation / template corruption
- `memory_utility_margin` stayed negative

That is a stronger diagnosis than before.

The problem is no longer "memory is absent". It is:

- memory is active
- memory is somewhat diverse
- memory is still net harmful on the positions that matter

### Why the Previous Training Schedule Was Wrong

The previous phased schedule started with a trunk-only warmup on a narrow memory dataset.

That was a mistake.

It let hundreds of millions of base-model parameters drift before the memory path had proven useful. The practical result was:

- base behavior degraded
- memory still did not become net helpful
- later joint phases became unstable or regressive

So the next training harness should preserve the base model aggressively and force the memory path to earn its role first.

## First-Principles Design

Transformer attention and modern Hopfield-style retrieval have the same core associative form:

\[
\mathrm{Attn}(Q, K, V) = \mathrm{softmax}(QK^\top)V
\]

The main difference between context attention and persistent memory is therefore not the read operator. It is the write/update rule.

That suggests a clean architecture:

- live context is a fast transient associative store
- persistent memory is a slower latent associative store
- both are read through the same attention-style mechanism

The model should not need a separate workspace branch, copy branch, or output-only rescue path to use memory.

## Current Architecture

The live architecture now uses **persistent latent memory tokens** per layer.

Those are not prompt tokens and not model weights. They are runtime latent vectors that live outside the fixed trunk parameters and are saved/restored as session state.

At a high level:

1. Earlier context is run through the trunk.
2. A shared controller computes write salience and projects token states into a latent memory-key space.
3. Write-time pooling compresses those states into a fixed set of memory tokens.
4. A sparse slot budget keeps only a small active subset.
5. During recall, a summary query selects a small subset of memory tokens.
6. The current token state attends over those retrieved memory tokens inside attention.

This keeps the core principle:

- memory is read through attention
- memory differs mainly in how it is written

## Why Sparsity Is Required

The previous dense version kept essentially every slot alive.

That was a problem even after diversity pressure improved duplication:

- all slots stayed active
- effective rank remained low
- memory became diffuse instead of selective

For this architecture family, sparsity is not an optional optimization trick. It is part of the inductive bias.

We want:

- a small number of salient write positions
- a small number of active memory slots
- a small retrieved subset at recall time

so that memory competes like relevant missing context, not like a global blur.

## Current Model Changes

The current implementation keeps the unified associative-attention direction and makes it stricter.

### Kept

- shared episodic controller across layers
- per-layer persistent memory state
- retrieval through attention K/V injection
- checkpoint loading from older NanoChat checkpoints

### Removed

- compatibility-era copy/workspace/logit branches
- dead config flags for old episodic modes
- trainer losses tied to removed branches
- trainer support for old value/span/workspace auxiliary objectives

### Added / Tightened

- hard write-token budget via `episodic_max_write_tokens`
- sparse slot activation budgets:
  - `episodic_summary_budget`
  - `episodic_anchor_budget`
- cleaner runtime detection of whether live memory is present
- checkpoint-config cleanup that drops unknown legacy config keys

The aim is to keep only the surfaces that belong to the actual architecture we want to train.

## Training Objective

The training harness now focuses on the objectives that still make architectural sense.

### 1. Base supervised loss

The model is still trained to answer correctly with memory enabled.

### 2. Weighted answer CE

Answer positions are not equally important.

The harness upweights:

- fact-bearing positions
- lightly supervises template/control positions
- gives a moderate weight to the remaining answer tokens

This keeps the task aligned with factual recall instead of diffuse token matching.

### 3. Fact-span margin

For explicitly annotated fact spans, the correct fact tokens are pushed above hard negatives.

This keeps the model focused on exact fact discrimination where the data can support it.

### 4. Guardrail KL

Non-fact answer positions are regularized against the model’s own no-memory baseline.

This is the main guardrail:

- memory should help where needed
- memory should not rewrite the surrounding answer structure gratuitously

### 5. Memory utility loss

Fact positions must improve when memory is enabled relative to the same model with memory disabled.

This is critical. Without it, the model can satisfy the task by drifting the trunk while leaving memory nonessential.

### 6. Slot diversity penalty

Memory writes are penalized when the live bank collapses into highly similar slots.

This is not the main learning signal, but it supports the sparse write/read bias.

## Training Phases

The previous `trunk-only -> memory-only -> joint` schedule was wrong.

The current harness is organized around three different phases:

### Phase 1: Memory Only

- train only memory controller + memory gate parameters
- freeze the base trunk
- force the memory path to become useful without letting the base model absorb the task

This should be the default first phase for new GPU runs.

### Phase 2: Interface Joint

- unfreeze memory parameters plus a small interface set:
  - `lm_head`
  - top transformer layers
- keep the rest of the trunk frozen

This lets the model adapt the readout/interface without destabilizing the whole base model.

### Phase 3: Full Joint Release

- optional
- only after utility margin is near zero or positive
- lowest learning rate

This is a release phase, not the place where the memory mechanism should first become useful.

## Recommended GPU Training Strategy

For the next serious run, use the current harness with a memory-first bias.

Recommended qualitative order:

1. Start from the trained NanoChat checkpoint.
2. Run a substantial **Phase 1 memory-only** stage.
3. Watch:
   - memory utility margin
   - fact-span margin
   - guardrail KL
   - active slot count
4. Only start Phase 2 if utility margin is improving toward zero.
5. Only start Phase 3 if Phase 2 is stable and memory is no longer net harmful.

If memory utility remains strongly negative, do not push into broader joint training. That means the architecture/read-write selectivity still needs work.

## What to Watch During Training

The important metrics now are:

- `memory_utility_margin`
  - if this stays negative, memory is still hurting the fact positions
- `guardrail_kl`
  - if this blows up, memory is distorting the rest of the answer
- `active_slots`
  - if this stays too high, writes are still too diffuse
- `anchor_margin`
  - useful coarse indicator for the first important fact position
- `gate_avg`
  - confirms whether the memory path is open at all

These matter more than raw loss alone.

## Open Risks

The current architecture is cleaner, but not yet proven.

Known open risks:

- write sparsity may still be insufficient
- summary-query retrieval may still be too coarse for some facts
- the interface phase may need fewer or different trunk parameters than the current top-layer choice
- some tasks may need more explicit distinction between summary memory and higher-fidelity anchor memory

Those are real risks, but they are now concentrated in the live architecture instead of being spread across compatibility-era branches.

## Status

The repo is now set up for the next real experiment:

- the design is aligned with unified sparse memory tokens
- dead memory features have been removed from the live model/trainer path
- checkpoint loading still works from old NanoChat checkpoints
- the training harness is ready for phased GPU training with guardrails

The next serious question is no longer "which of five legacy memory branches should we tune?"

It is:

> can sparse, selective latent memory tokens become net useful under a memory-first, guardrailed training schedule without degrading the base model?

That is the right next experiment.
