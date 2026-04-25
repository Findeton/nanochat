# Long-Term Memory for NanoChat

## Purpose

NanoChat already has short-term working memory through ordinary causal attention over the current context window. The long-term memory goal is narrower and more specific:

\[
p_\theta(y \mid x, M) \approx p_\theta(y \mid x, H)
\]

where `x` is the current query, `H` is relevant past context that is no longer in the prompt, `M` is persistent runtime memory written from earlier interaction, and `y` is the answer.

The goal is not an external database, a symbolic key-value lookup, or a special-case copy branch. The goal is a generic transformer-style associative system where persistent memory behaves like missing context.

This document is the current design memo. It explains the architecture we have, the evidence that led us here, what is still wrong, and the next architecture we should move toward: symmetric attention-like read/write memory.

## Visual Overview

Normal transformer attention can only read from the current context window:

![Normal transformer attention](dev/diagrams/01_normal_transformer_attention.svg)

The current memory architecture extends the attention read by adding retrieved persistent memory K/V beside context K/V:

![Unified attention with memory](dev/diagrams/02_unified_attention_with_memory.svg)

The current write path is learned and sparse, but it is not yet the fully symmetric attention-write design:

![Memory write selection](dev/diagrams/03_memory_write_selection.svg)

Training tells the system what is useful to remember by rewarding exact delayed recall while guarding normal answer structure:

![Training alignment and guardrails](dev/diagrams/04_training_alignment_and_guardrails.svg)

## Executive Summary

The architecture direction is still right:

\[
\mathrm{Attn}(Q, [K_M; K_C], [V_M; V_C])
\]

where `C` is the current context and `M` is persistent memory. The read should remain unified attention over context and memory.

The current implementation is an important intermediate system:

- memory is latent torch tensor state, not Python dicts
- memory is not prompt text
- memory is not baked into model weights per session
- the read path is attention-like and integrated into the transformer block
- memory writes are sparse and learned
- the training harness now includes direct key-token objectives and rollout recovery

But the current implementation is not the final clean architecture. The write path still uses a separate learned salience head (`write_gate`) plus attention-like slot pooling. That was useful because it gave us control and observability, but it is less principled than the read path.

The next architectural refinement should make memory writes symmetric with reads:

\[
\text{read: context/query tokens attend to memory slots}
\]

\[
\text{write: memory slots attend to context tokens}
\]

That means the same associative-attention idea governs both directions. This is not implemented yet. It should be treated as the next design target, not as a description of the current code.

The caveat is real: pure attention-write can store too much unless it has scarcity. We still need top-k selection, budgets, slot competition, strength decay, no-op behavior, diversity pressure, and guardrail losses. Without those, memory becomes a blurry second context window instead of selective long-term memory.

## Design Target

The target is a generic associative-attention memory system:

- ordinary context tokens are one associative store
- persistent memory tokens are another associative store
- both are read through the same transformer attention law
- persistent memory differs in its update rule and lifetime
- the write rule should eventually be attention-like too

The read equation is:

\[
\mathrm{Attn}(Q, [K_M; K_C], [V_M; V_C])
\]

This stays. The question is not whether generalized read attention changes. It does not.

The deeper question is how `K_M` and `V_M` are produced. Today they are produced by a sparse learned writer with a scalar salience head. The cleaner future version should produce them by memory slots attending over context states.

## Current Architecture

The current implementation uses persistent latent memory tokens per transformer layer.

Each layer has a fixed-size memory bank with:

- latent memory tokens / values
- retrieval keys
- scalar strengths
- kind ids for summary vs anchor slots
- source-position metadata for diagnostics

The memory bank itself has no trainable parameters. Learned behavior lives in the shared episodic controller and in the per-block memory interface.

At a high level:

1. A previous turn is processed by the transformer.
2. Hidden states are projected into memory key/value space.
3. A learned write salience head scores candidate token states.
4. Learned summary and anchor slot queries pool selected states.
5. Slot competition and budgets keep only a sparse subset active.
6. At recall time, a stable recall query selects top-k memory slots.
7. The current token stream attends over those memory slots as extra K/V in the same attention operation used for context.

Memory retrieval scores must be on the same numeric scale as normal attention logits. The current implementation therefore scales dot-product scores in memory-key space by \(1 / \sqrt{d_M}\) and caps extreme similarities with `episodic_beta`. Without this, 256-dimensional memory-key dot products can become O(100) while normal attention logits are O(10), causing early layers to be hijacked by memory rather than smoothly combining memory and context.

The important point is that memory read is already integrated into attention. It is not a second decoder and not an explicit symbolic retrieval answer.

## Normal Attention vs Memory Attention

A normal transformer layer computes attention over the current context:

\[
Y = \mathrm{softmax}(Q_C K_C^\top)V_C
\]

If `Pebble-Cloud` is not in the context window, normal attention cannot retrieve it from runtime state. It may only guess from weights.

Our current memory read computes attention over both retrieved memory and context:

\[
Y = \mathrm{softmax}(Q_C [K_M;K_C]^\top)[V_M;V_C]
\]

Memory therefore acts like extra latent context. It can vote in the same token-generation path as normal attention.

This part of the idea is clean and should stay.

## Recall Query

The recall query is the pre-filter that chooses which persistent memory slots are available to token-level attention.

Current implementation:

1. Build a recall mask over the stable prefix, usually up to the latest assistant start token.
2. Average the hidden states under that mask.
3. Normalize and project that average with `recall_query_proj`.
4. Compare the query to stored memory keys.
5. Retrieve top-k active slots, biased by memory strength and kind bias.

In simplified form:

\[
r = W_{\text{recall}}\ \mathrm{norm}(\mathrm{mean}(H_{\text{stable prefix}}))
\]

\[
S = \mathrm{TopK}(rK_M^\top + \log(\mathrm{strength}))
\]

Then the selected memory slots become extra K/V for the transformer block.

The stable-prefix mask is a practical fix for retrieval drift. Under teacher forcing, the generated answer prefix is gold; under free generation, the prefix may be wrong or repetitive. If retrieval depends too much on that evolving suffix, one early bad token can change what memory is retrieved. The stable prefix makes retrieval depend more on the question and less on the model's own mistakes.

This is useful, but still not final. A learned answer-level retrieval policy should eventually replace the hand-shaped recall mask.

## Current Write Rule

The current write rule is learned and sparse, but not fully symmetric with read attention.

For each candidate hidden state \(h_t\), the controller computes:

\[
s_t = W_{\text{write}}\ \mathrm{norm}(h_t)
\]

\[
\alpha_t = \sigma(s_t)
\]

The highest-salience token states survive a max-write-token filter. Then learned slot queries attend over those selected token states:

\[
a_{slot,token} = q_{slot}^\top W_K h_t
\]

The system combines slot-to-token attention with token-to-slot competition. Summary slots get a general salience boost; anchor slots get a stronger salience boost because they are meant to preserve sharper identifying details. Memory strength is derived from salience, slot competition, and the active budget mask.

In other words, the current writer decides what to remember using:

- learned scalar salience
- learned slot-query affinity
- token-to-slot competition
- summary/anchor budgets
- strength decay and overwrite dynamics
- delayed recall gradients from training

This is not a hand-coded rule like "store dog names." The model learns that facts like `Pebble-Cloud` matter because later recall losses reward storing representations that make the answer token win.

## Why The Current Write Rule Is Not Fully Satisfying

The current write path was chosen because the immediate problem was empirical: memory was either inactive, collapsed, or unable to move logits enough. A scalar write gate gave us a direct way to expose and train salience.

That was useful, but it left a conceptual asymmetry:

- read uses attention from token/query state to memory slots
- write uses a separate salience detector plus slot pooling

This is better than external symbolic memory, but it is not the cleanest associative architecture. The honest critique is that we solved the immediate bottleneck first and kept an internal engineering knob that now looks less elegant than it should.

The next version should remove that asymmetry.

## Next Architecture: Symmetric Attention-Like Read and Write

The cleaner design is:

\[
\text{Read: } Q_C \rightarrow K_M,V_M
\]

\[
\text{Write: } Q_M \rightarrow K_C,V_C
\]

Read means current context/query tokens attend to memory slots:

\[
\mathrm{read}(h_t) = \mathrm{Attn}(W_Q h_t, K_M, V_M)
\]

Write means persistent memory slots attend to current context states:

\[
\mathrm{write}(m_s) = \mathrm{Attn}(q_s, K_C, V_C)
\]

where \(q_s\) is a learned or state-conditioned query for memory slot \(s\).

In this design, salience is not a separate scalar head. It emerges from attention energy and confidence:

\[
e_{s,t} = q_s^\top k_t
\]

\[
a_{s,t} = \mathrm{sparsemax/topk/softmax}(e_{s,t})
\]

\[
\tilde{m}_s = \sum_t a_{s,t} v_t
\]

Slot strength can be derived from the confidence of the attention distribution:

- high max score
- large margin between best and second-best token
- low entropy
- strong agreement across layers or heads
- downstream utility during delayed recall

This would make memory a bidirectional associative system:

- context writes into memory by attention
- memory reads back into context by attention

That is closer to the transformer/Hopfield compatibility idea than the current separate salience head.

## Scarcity Is Still Required

Pure attention-write is not automatically good memory.

If every memory slot softly attends to every token, the system can store:

- syntactic glue
- frequent filler words
- diffuse summaries
- duplicated slots
- noisy global biases

That would make memory act like a blurry second context window, which is not what we want.

The symmetric write design still needs bottlenecks:

- top-k write tokens per slot
- top-k active slots per write event
- no-op / do-not-write option
- strength thresholds
- summary vs anchor budgets
- slot diversity pressure
- overwrite and decay rules
- memory-vs-no-memory utility loss
- non-fact guardrail KL
- exact key-token rank loss
- rollout validation against gibberish and repetition

The point is not "attention alone solves memory." The point is that attention should be the common association primitive, while scarcity and training objectives make it selective.

## How The System Learns What To Remember

The system does not know ahead of time that a dog name is important. It learns from delayed consequences.

The training episode says, implicitly:

1. This earlier span appeared: `My dog's name is Pebble-Cloud.`
2. Later, the model is asked: `What's my dog's name?`
3. The answer requires the exact tokens for `Pebble-Cloud`.
4. If memory-on improves those tokens, the memory pathway is rewarded.
5. If memory-on corrupts ordinary answer structure, guardrails penalize it.

Current training signals include:

- normal answer cross-entropy
- weighted answer CE with higher fact-token weight
- fact-span hard-negative margin
- direct key-token CE
- key-token top-1 rank margin
- memory-vs-no-memory key-token utility margin
- anchor/confuser margin
- memory-read lexical auxiliary
- write diversity pressure
- non-fact guardrail KL
- rollout recovery loss after short generated prefixes

The practical rule is:

> architecture creates scarcity; losses define usefulness.

That is the important mental model. The memory system is not told what facts are by rules. It is trained so that facts that later change an answer become valuable to store.

## Evidence From Earlier Iterations

### Early Latent Recall Was Real

Earlier `r18`-style experiments showed that narrow delayed recall can emerge from latent memory without external retrieval. That result still matters. It means the project should not default to Python dict memory or symbolic retrieval as the core architecture.

### Branchy Rescue Architectures Were Useful But Not Final

Plan/payload, workspace, and copy-style branches taught us that:

- the first fact token matters disproportionately
- later fact subtokens can improve even when the answer fails
- local token uplift is not enough if sequence-level generation drifts

But those branches were too ad hoc to be the final architecture. They were probes, not the destination.

### Sparse Memory Tokens Fixed A Real Failure

Sparse memory tokens improved earlier collapse modes:

- memory gates were no longer effectively shut
- memory banks were no longer trivially rank-1 collapsed
- memory became active enough to influence logits

That was real progress.

### The New Failure Mode Is Sequence-Level Use

The latest failures are more specific:

- memory can raise correct-token logprob under teacher forcing
- memory still often fails to make the correct token top-1
- free generation can degenerate into repetition or gibberish
- live sequential writes are harder than training-style one-shot memory builds

The system no longer looks like "dead memory." It looks like active but badly aligned memory.

## Current Implementation State

The current `nanochat-felix` code should be understood as a strong intermediate implementation, not the final symmetric design.

Implemented now:

- persistent latent memory tokens per layer
- shared episodic controller
- sparse summary and anchor slots
- functional online memory update
- replay-based memory construction for training
- completed-turn write policy for chat
- stable pre-answer recall mask
- memory K/V injection into transformer attention
- direct key-token CE loss
- key-token rank margin loss
- key-token memory-utility loss
- anchor/confuser loss
- lexical auxiliary on memory read
- rollout-aware recovery training
- diagnostic ablations for no-memory-state and no-memory-read

Not implemented yet:

- symmetric attention-only write rule
- removal of the standalone `write_gate`
- learned replacement for the recall-mask heuristic
- rollout-aware validation as the primary checkpoint selector
- stochastic low-temperature rollout curriculum after greedy rollout works
- proof that live chat generation stops degenerating

## First-Principles Diagnosis

There are four core issues.

### 1. Training And Inference Must Use The Same Memory Dynamics

Training used to approximate:

\[
M = B_\theta(H_{\text{full prior context}})
\]

Live chat actually uses:

\[
M_{t+1} = U_\theta(M_t, H_t)
\]

These are not equivalent. The current code now supports replay through the online update rule, which is the right direction. The symmetric writer should preserve that: the same write/update operator must be used in training and runtime.

### 2. Retrieval Must Not Drift With Bad Generated Prefixes

Teacher forcing gives the model a gold prefix. Free generation does not. A wrong early answer token can distort hidden state and therefore memory retrieval.

The current stable prefix recall query is a practical fix. Longer term, the model should learn answer-level retrieval that is stable for the duration of a response.

### 3. Correct Token Must Win, Not Merely Improve

Old memory utility could improve:

\[
\log p(y_t \mid x,M) - \log p(y_t \mid x,\varnothing)
\]

without satisfying:

\[
z(y_t) > \max_{j \ne y_t}z(j)
\]

That is why key-token rank margin is now part of training. The memory must make the exact datum win the token decision.

### 4. Memory Must Help Without Becoming A Global Bias

If memory changes every answer token, the model degenerates. The system needs memory to affect fact-bearing decisions while preserving normal answer structure.

That is why guardrails remain essential even in a cleaner symmetric attention-write design.

## Training Strategy

### Phase 1: Memory Path Only

Freeze the base trunk. Train the memory controller, writer, retriever, and memory interface. The goal is to make memory useful before the trunk adapts around it.

Primary metrics:

- key-token CE down
- key-token rank margin up toward positive
- key-token utility margin positive
- anchor margin improving
- guardrail KL controlled
- no explosion in active slots or repetition

### Phase 2: Interface Joint

Unfreeze a small interface set, such as top layers and answer-facing projections. Keep most of the trunk frozen. Start only if held-out live recall improves.

### Phase 3: Full Joint Release

Use a very low learning rate. This is a release phase, not where memory should first become useful.

## Rollout Training Plan

The training path should become increasingly aligned with live generation.

Stage 1 uses greedy or near-greedy rollout:

- `temperature = 0`
- `top_k = 1`
- generate a short prefix from the model itself
- train the gold continuation beginning at the fact token
- upweight missing-key and repetition failures

Stage 2 adds low-temperature stochastic rollout only after greedy generation is stable:

- temperature around `0.2 - 0.4`
- small top-k
- evaluate robustness around the decision boundary

High-temperature rollout should not be the starting point. It would add noise before the deterministic recall path is healthy.

## Evaluation Plan

Raw loss is not enough. Teacher-forced logprob is not enough.

Primary evaluation should include:

- live sequential-write recall accuracy
- short-horizon greedy generation accuracy
- missing-key rate
- repetition/gibberish rate
- first fact-token top-1 accuracy
- subsequent fact-token top-1 accuracy
- key-token rank and logprob
- memory-on vs no-memory key-token delta
- no-memory-state and no-memory-read ablations
- control-token accuracy and KL
- active slot count and retrieval entropy
- actual memory attention mass per layer/head/token
- max memory-vs-context attention score at the first generated token

The best checkpoint is not necessarily the one with the lowest training loss. Previous runs showed that held-out behavior could peak early and then worsen.

## Next Code Plan

### Keep The Current Read Equation

Do not change the generalized attention read:

\[
\mathrm{Attn}(Q, [K_M;K_C], [V_M;V_C])
\]

This is the foundation.

### Refactor Write Into Slot-To-Context Attention

Implement an experimental writer where each memory slot attends over context hidden states:

\[
A_{s,t} = \mathrm{sparse\_attention}(q_s, k_t)
\]

\[
\tilde{m}_s = \sum_t A_{s,t}v_t
\]

Derive strength from attention confidence rather than a separate scalar write gate.

### Preserve Scarcity

The first symmetric writer should still include:

- top-k token selection per slot
- top-k slot activation per event
- summary/anchor budgets
- slot diversity loss
- decay and overwrite rules
- optional no-write slot

### Compare Against The Current Writer

The current writer should stay only long enough to provide a fair ablation:

- current `write_gate` salience writer
- symmetric slot-to-context attention writer
- no-memory-state diagnostic
- no-memory-read diagnostic

If the symmetric writer wins or matches, remove the standalone salience path and its dead parameters.

### Upgrade Validation

Make rollout-aware validation part of the default watcher:

- evaluate checkpoint 500, 1000, 1500, ...
- save full key-token ranks and logprobs
- save failure examples with generated text
- compare full memory, no memory state, and no memory read

## What To Stop Doing

Do not:

- treat full-context teacher-forced gains as proof live chat is fixed
- optimize only gold-token uplift without rank/top-1 pressure
- keep adding branchy copy/workspace paths
- keep dead parameters for backwards compatibility
- assume attention-write alone solves memory without scarcity
- move to high-temperature rollout before greedy rollout works

## Status

Current status:

- architecture family: still correct
- generalized memory/context attention read: implemented
- sparse latent memory: implemented
- direct key-token training pressure: implemented
- rollout recovery training: implemented
- symmetric attention-write: not implemented yet
- live generation quality: not solved yet

The next design problem is now precise:

> Can NanoChat train a single generic attention-with-memory system whose read and write operations are both associative, whose memory remains scarce and selective, and whose live sequential recall behaves like missing context instead of latent noise?

That is the right target.
