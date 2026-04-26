# Long-Term Memory for NanoChat

## Purpose

NanoChat already has short-term working memory through ordinary causal attention over the current context window. The long-term memory goal is narrower:

\[
p_\theta(y \mid x, M) \approx p_\theta(y \mid x, H)
\]

where `x` is the current query, `H` is relevant past context that is no longer in the prompt, `M` is persistent runtime memory written from earlier interaction, and `y` is the answer.

The goal is not a Python dictionary, an external database, a symbolic key-value lookup, or a special-case copy branch. The goal is a generic transformer-style associative memory where persistent state behaves like missing context.

## Visual Overview

Normal transformer attention reads only from the current context window:

![Normal transformer attention](dev/diagrams/01_normal_transformer_attention.svg)

The current memory architecture extends attention by adding retrieved persistent memory K/V beside context K/V:

![Unified attention with memory](dev/diagrams/02_unified_attention_with_memory.svg)

The current write path is learned and sparse. It is not yet the cleaner symmetric attention-write design:

![Memory write selection](dev/diagrams/03_memory_write_selection.svg)

Training aligns the memory path with delayed recall while guarding normal answer structure:

![Training alignment and guardrails](dev/diagrams/04_training_alignment_and_guardrails.svg)

## Executive Summary

The core read architecture is still the right foundation:

\[
\mathrm{Attn}(Q, [K_M; K_C], [V_M; V_C])
\]

where `C` is current context and `M` is persistent memory. The model should not switch into a separate memory decoder. Memory should enter through the same attention law as context.

The latest runs changed our diagnosis. The main problem is no longer "memory is ignored" or "memory is only used in layer 0." Instrumentation shows memory is causal and distributed across layers. The current failure is more specific:

- The model often knows useful local fact tokens under teacher forcing.
- Live generation still chooses the wrong remembered branch on some examples.
- After producing the correct key token, it can continue wrongly, repeat, or drift into a longer lexical neighbor such as `Fig` -> `Figaro`.
- Current-vs-old corrections remain harder than simple identity recall.

That means more generic memory read/write is still desirable, but the immediate training gap is self-generated prefix robustness. Phase 3.7 implements that next patch locally: generated-context whole-span contrast plus post-key stop/continuation training. It does not add a copy path. It adds losses that make the existing decoder prefer the correct remembered span over wrong remembered branches under prefixes the model actually samples.

## Design Target

The long-term target is a generic associative-attention memory system:

- Context tokens are one associative store.
- Persistent memory slots are another associative store.
- The read path attends over both stores using the same attention law.
- Persistent memory differs by lifetime and update rule, not by being a separate symbolic API.
- The write path should eventually become attention-like too.

The read equation stays:

\[
\mathrm{Attn}(Q, [K_M;K_C], [V_M;V_C])
\]

The open design question is how `K_M` and `V_M` are produced and updated. Today they are produced by a sparse learned writer with salience, slot competition, and budgets. A cleaner future version should make memory slots attend back over context states, so read and write are two directions of the same associative mechanism.

## Current Architecture

The current implementation uses persistent latent torch tensor memory. It is not prompt text and it is not a Python dictionary.

Each layer has a fixed-size memory bank with:

- latent memory values
- retrieval keys
- scalar strengths
- kind ids for summary and anchor behavior
- source-position metadata for diagnostics

The memory bank itself is runtime state. Learned behavior lives in the episodic controller and per-block memory interface.

At a high level:

1. A previous turn is processed by the transformer.
2. Hidden states are projected into memory key/value space.
3. A learned salience/write path scores candidate token states.
4. Learned summary and anchor queries pool selected states into slots.
5. Slot competition, budgets, decay, and diversity pressure keep memory sparse.
6. At recall time, stable recall queries select top-k memory slots.
7. The current token stream attends over selected memory slots as extra K/V in normal attention.

Memory retrieval scores must be on the same numeric scale as normal attention logits. The current implementation scales memory-key dot products by \(1 / \sqrt{d_M}\) and caps extreme similarities with `episodic_beta`. This avoids memory logits overwhelming context attention.

## Read Path

Normal attention computes:

\[
Y = \mathrm{softmax}(Q_CK_C^\top)V_C
\]

If `Pebble-Cloud` is not in the context window, normal attention cannot retrieve it from runtime state.

Memory-augmented attention computes:

\[
Y = \mathrm{softmax}(Q_C [K_M;K_C]^\top)[V_M;V_C]
\]

Memory therefore acts like extra latent context. It can vote in the same path that produces next-token logits.

This part is clean and should stay.

## Recall Query

The recall query is the pre-filter that chooses which memory slots are available to token-level attention. It is created from current hidden states, then projected into memory-key space. The retrieved memory slots then become extra K/V for the attention block.

The system currently uses a stable-prefix recall mask. This matters because under teacher forcing the answer prefix is always gold, while under live generation the answer prefix may contain mistakes. If retrieval follows every generated suffix too aggressively, one bad token can change the memory being retrieved. Stable-prefix recall makes retrieval depend more on the user question and less on the model's own sampled mistakes.

## Write Path

The current writer is learned but not fully symmetric with the read path.

Current write rule:

\[
\text{salience}_t = f_\theta(h_t)
\]

\[
\tilde{m}_s = \sum_t \mathrm{sparse\_weights}_{s,t} v_t
\]

where token states compete to be written into summary or anchor slots.

This is not ad hoc in the sense of using symbolic rules or hand-coded facts. It is trained tensor machinery. But it is less elegant than the read equation because salience is a separate scalar decision, not the same energy function used by attention.

The future symmetric writer should look more like:

\[
A_{s,t} = \mathrm{sparse\_attention}(q_s, k_t)
\]

\[
\tilde{m}_s = \sum_t A_{s,t} v_t
\]

where memory slots attend to context tokens to decide what to absorb.

Pure attention-write is not automatically better. Without scarcity, it will store too much. We still need budgets, no-write behavior, decay, slot competition, and diversity pressure. The goal is not "write everything by attention." The goal is "write selectively using an attention-shaped mechanism."

## How Salience Is Determined

Today, salience is learned from training pressure, not hand-coded rules. Tokens become worth writing when writing them helps delayed recall and does not damage guardrails.

The training signal says, in effect:

- If a fact token is needed later, memory should make it easier to predict.
- If no memory is present, the model should not hallucinate arbitrary old facts.
- If multiple facts conflict, current/correct facts should beat stale facts.
- If a token is ordinary conversational filler, writing it should not help enough to spend scarce slots.

Mathematically, salience is induced by gradients from recall losses, utility margins, anchor losses, write diversity, and guardrail losses. The scalar write gate is only the current parameterization of that learned decision.

## Training Harness

The current training harness is phased because the base model already knows language, but the memory interface starts untrained.

### Phase 1: Memory Path Only

Freeze the base trunk. Train memory controller, writer, retriever, and memory interface.

Primary goals:

- make memory causally useful
- avoid destroying base language behavior
- make key-token CE and rank margins improve
- keep active slots sparse and stable

### Phase 2: Interface Joint

Unfreeze a limited interface set. Keep most of the trunk frozen. Train the model to combine memory with normal answer structure.

This phase introduced answer-start pressure, habit-confuser margins, and deference objectives. It helped memory become useful but did not fully solve live generation.

### Phase 2.5: Deference and Span Robustness

Add stronger branch/deference objectives:

- current fact over old fact
- correct remembered branch over no-memory habit
- answer start and key-token margins
- post-key continuation losses

This improved margins and reduced some obvious wrong defaults. It still left failures where local token probabilities looked good, but live generation selected the wrong branch or drifted after the key.

### Phase 3: Generated-Prefix Recovery

Train on prefixes sampled from the model itself. This moves training closer to inference because the model must recover after its own imperfect early tokens.

This is the right direction, but early versions mostly taught recovery after generic bad prefixes. They did not directly contrast whole wrong remembered branches against the correct remembered span.

### Phase 3.5 and 3.6: Branch Recovery Attempts

These phases added generated-context branch recovery and dynamic no-memory confuser detection.

Findings:

- Memory was causal: no-memory-state and no-memory-read ablations collapsed.
- Memory was not only layer 0. A representative diagnostic showed memory mass around layer 2, layer 6, layer 10, and layer 4 rather than only the first block.
- Some token probabilities improved. Phase 3.6 step 250 raised geometric key probability from about `0.1472` to `0.1629` compared with the prior branch run, and top-1 key-token accuracy moved from about `61.5%` to `62.8%`.
- End-to-end live exact recall remained poor at `2/15 = 13.3%`.
- Stage 2 structured examples improved more than mixed/current-vs-old examples.

The central lesson: local token training is insufficient. The model can know a token locally and still choose the wrong remembered branch as a sequence.

### Phase 3.7: Generated-Context Span Contrast

Phase 3.7 is the current local patch.

It trains on generated answer prefixes, but the loss is span-level rather than single-token only:

\[
\Delta = \log p(\text{correct remembered span} \mid \text{generated prefix}) -
\log \sum_i p(\text{wrong branch}_i \mid \text{generated prefix}_i)
\]

The loss pushes:

\[
\Delta > \text{margin}
\]

It also adds post-key stop/continuation pressure:

- after the correct key appears, prefer the correct next token
- penalize repeating the key span
- penalize drifting into known lexical continuations such as `Figaro` when the fact is `Fig`

This is a dynamic training algorithm in the practical sense: it samples from the current model, finds the mistakes that actually occur, and turns those into contrastive training events. It is not dynamic runtime weight update, and it is not a copy mechanism.

Implemented metrics include:

- `phase37_branch_ce`
- `phase37_branch_loss`
- `phase37_branch_margin`
- `phase37_stop_loss`
- `phase37_stop_margin`
- `phase37_applied`
- `phase37_no_memory_default`
- `phase37_wrong_span`
- `phase37_partial_wrong`
- `phase37_repeated`

## Current Failure Model

The current failures split into three buckets.

### 1. Branch Selection

The model retrieves useful memory but chooses the wrong remembered branch. Example pattern:

```text
expected current: Clover
predicted stale: Miso
```

This is not solved by copying one token. It requires scoring the correct remembered span above competing remembered spans under the same live prefix.

### 2. Post-Key Continuation

The model emits the correct key but continues wrongly:

```text
expected: Fig
generated: Figaro
```

or:

```text
expected: Biscuit
generated: Biscuit Biscuit ...
```

This means the first key token is not enough. The system needs a stop/continuation loss after the remembered fact.

### 3. Teacher-Forcing Mismatch

Teacher forcing sees the gold prefix. Live generation sees the model's own prefix. If training never conditions on generated prefixes, the model can look good in token-probability probes but fail in actual chat.

Phase 3.7 directly targets this mismatch.

## What We Know From Instrumentation

Important findings so far:

- The memory path is causal: disabling memory state or memory read damages recall.
- The read path is not concentrated only in the first layer after the multi-layer memory fix.
- Active slots remain sparse, typically around the intended budget rather than all slots saturating.
- Guardrail KL rises when memory objectives get aggressive, so language preservation remains a real constraint.
- Key-token rank and probability are necessary diagnostics but not sufficient.
- Live exact recall is the primary metric.

Useful checkpoint diagnostics:

- full memory vs no-memory-state vs no-memory-read
- token probabilities for first key token and continuation pieces
- generated text examples, not just CE
- memory attention mass per layer
- key-token rank margin
- branch/span contrast margin
- post-key repeat margin
- current-vs-old correction accuracy

## Why Not A Copy Branch

A direct copy path would make some toy examples look better, but it is not the architecture target.

A copy branch says:

```text
retrieve span -> bypass decoder -> emit span
```

The architecture target says:

```text
retrieve memory as latent K/V -> attend with context -> decoder assigns token probabilities
```

Phase 3.7 stays in the second category. It changes the loss, not the inference architecture. The model must still generate through its normal logits.

## Symmetric Attention-Like Write Direction

The cleaner architecture remains:

\[
\text{read: context/query tokens attend to memory slots}
\]

\[
\text{write: memory slots attend to context tokens}
\]

This would reduce the mismatch between "what got written" and "what gets queried later." It is especially relevant for binding failures, because the same associative geometry would shape both writing and reading.

But it is not implemented yet, and it should not be treated as magic. A symmetric writer must preserve scarcity:

- top-k token selection
- top-k slot activation
- no-write slot or threshold
- slot diversity
- strength decay
- overwrite rules
- summary/anchor budgets
- guardrail losses

The current recommendation is not to jump directly to a full writer rewrite before validating phase 3.7. If phase 3.7 improves live exact recall but binding failures remain, the symmetric writer becomes the next architecture patch.

## Current Implementation Status

Implemented in `nanochat-felix`:

- unified memory/context attention read
- sparse latent memory slots
- scaled memory logits
- stable-prefix recall
- multi-layer memory use
- memory utility and anchor losses
- key-token CE and rank margins
- answer-start and habit-confuser objectives
- rollout/generation-aware recovery
- phase 3.7 generated-context span contrast
- phase 3.7 post-key stop/continuation loss
- CPU/MPS-compatible diagnostics and smoke tests

Not implemented yet:

- symmetric attention-write replacing scalar salience
- learned no-write slot for the symmetric writer
- full online generated-chat training loop with long multi-turn sampled conversations
- runtime weight updates
- a separate copy/pointer decoder, intentionally

## Recommended Next Training Step

Run phase 3.7 from the best current checkpoint rather than blindly continuing older phases.

The first run should be short enough to catch regressions:

- 250 steps: check branch margin, post-key stop margin, live exact recall
- 500 steps: check if gains persist or overfit
- stop early if live exact recall does not move while guardrail KL climbs

Expected healthy signs:

- `phase37_branch_margin` moves positive
- `phase37_stop_margin` improves
- `phase37_repeated` drops
- generated examples stop repeating key spans
- current-vs-old examples improve, not just simple identity facts
- memory ablations still show that memory is the cause of the improvement

Unhealthy signs:

- key-token CE improves but live exact recall stays flat
- guardrail KL climbs without end-to-end gains
- phase37 events mostly become no-ops
- model learns templates but still picks stale facts
- memory attention collapses back into one layer

## Evaluation Standard

The best checkpoint is the one with the best live behavior, not the lowest training loss.

Required evaluation:

- live sequential write/recall exact accuracy
- simple identity recall
- structured field recall
- mixed current-vs-old correction recall
- first key-token probability and rank
- full fact-span probability
- generated answer text
- repetition and gibberish rate
- memory-on vs no-memory-state vs no-memory-read
- memory attention mass by layer
- guardrail KL and normal chat sanity

## What To Stop Doing

Do not:

- treat teacher-forced token gains as proof chat recall is solved
- add a copy/pointer branch just to win the dog-name demo
- keep dead parameters for backwards compatibility
- train high-temperature rollouts before low-temperature/greedy recovery is stable
- optimize only the first arbitrary token and ignore post-key continuation
- ignore stale-vs-current correction examples

## Current Status

The architecture family still looks right. Memory is real, causal, and distributed across layers. The live problem is now sharper: the decoder must make the correct remembered branch stable under its own generated prefixes, then stop or continue sanely after the remembered key.

Phase 3.7 is the current local implementation of that idea. If it moves live exact recall, the next step is to consolidate it and then revisit the symmetric attention-write design. If it does not, the evidence points toward an architecture-level write/read alignment problem rather than simply "train longer."
