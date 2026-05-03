# Long-Term Memory for NanoChat

## Purpose

NanoChat already has short-term working memory through ordinary causal attention over the current context window. The long-term memory objective is to make missing past context available again without turning memory into a Python dictionary, retrieval database, or special-purpose copy path.

The target behavior is:

\[
p_\theta(y \mid x, M) \approx p_\theta(y \mid x, H)
\]

where `x` is the current query, `H` is relevant past context that is no longer in the prompt, `M` is persistent runtime memory written from earlier interaction, and `y` is the answer.

The guiding principle is simple: persistent memory should behave like additional attention context, with different lifetime and write rules. It should not be a separate symbolic API.

## Current Bottom Line

The symmetric read/write architecture is the right foundation, but the latest
evidence says the remaining bottleneck is not "can the model read memory?" It
is "can the decoder reliably treat memory as an arbitrary copy source during
live generation?"

The unified attention read path is active and useful:

\[
\mathrm{Attn}(Q, [K_M;K_C], [V_M;V_C])
\]

where `C` is the current context and `M` is persistent memory. We have strong evidence that this memory path is active, causal, and no longer concentrated only in the first layer.

What we achieved:

- Memory is represented as latent torch tensors, not prompt text or Python dicts.
- Memory K/V is injected into transformer attention, so the decoder remains a normal transformer decoder.
- The symmetric attention-like write path is implemented: memory slots query context-token keys and route through an explicit no-write option.
- Ablations show memory causally improves key-token probabilities and exact recall.
- Layer probes show memory usage across layers, especially middle and later layers.
- The training harness now measures token probabilities, no-memory collapse, memory mass by layer, live deterministic recall, sampled recall, branch errors, repetition, and default/habit failures.
- The V4 long run showed stable improvement in supervised/token/probe metrics over thousands of steps: lower answer CE, better anchor/key-token margins, better span-hard-token margins, and positive probe binding margins.
- High-cardinality name and normal-chat data improved the closed-list problem; the model is no longer just learning a few dog names.

What we did not achieve yet:

- Robust live exact recall across sessions.
- Stable current-vs-old correction, for example `Clover` should override older `Miso`.
- Reliable post-key stopping and continuation, for example `Fig` should not become `Figaro`.
- Reliable arbitrary-string recall, for example `Buh` should not become `Bist` and `Taloobrook` should not become a fluent but unrelated name.
- Positive whole-trajectory preference margins. Teacher-forced and probe metrics improve faster than generated-answer arbitration.
- A Titans-style test-time learned neural memory.
- A MemoryLLM-style self-updatable memory pool evaluation pipeline at scale.

The most honest diagnosis is: memory read/write now works well enough to expose
the real failure. The model often has the right fact available, but the frozen
or lightly trained trunk does not yet have a strong enough generic policy for:

```text
retrieve relevant memory span -> emit the exact span -> stop or continue naturally
```

This matches the useful version of the "attention mostly recognizes what the
weights already understand" hypothesis. Memory can supply new data, but the
weights must already know the operation for using arbitrary data. V5 therefore
trains that operation directly, with full-model adaptation for a short phase and
strong guardrails so ordinary chat behavior does not get bulldozed.

## V5 Plan: Arbitrary Copy With Real Guardrails

V5 keeps the symmetric architecture and changes the training problem. Instead
of mostly asking for named entities from a large but still human-looking pool,
it adds truly arbitrary values:

- character-random spans of length 1 to 15 from `[a-zA-Z0-9 -_.]`
- token-random spans bucketed by token length
- varied memory types beyond pet names: project codes, branches, secrets, notes, devices, contacts, dates, preferences, regions, incident ids, aliases, and short exact strings
- no-memory guardrails where the correct answer is "I do not have that saved"
- irrelevant-memory guardrails where memory is present but should not affect the answer
- real original-distribution chat data through SmolTalk/KL guardrails, not only synthetic guardrail rows

The training run is one 100k-step script with three phases:

| Phase | Steps | Trainable weights | Purpose |
| --- | ---: | --- | --- |
| A | 30k | memory + interface | Learn arbitrary memory binding and exact-copy pressure without destabilizing the trunk. |
| B | 10k | full model at very low LR | Teach the trunk/LM head the generic arbitrary-copy operation that frozen weights may not already support. |
| C | 60k | memory + interface | Polish trajectory preference, branchpoint recovery, no-memory deference, and normal-chat preservation after the full-model adaptation. |

Phase B deliberately uses a smaller device batch. Phase A and C use the largest
safe batch for the 24GB GPU. This avoids repeating the OOM failure while still
using the GPU aggressively.

## Visual Overview

Normal transformer attention reads only from the current context window:

![Normal transformer attention](dev/diagrams/01_normal_transformer_attention.svg)

The current memory architecture extends attention by adding retrieved persistent memory K/V beside context K/V:

![Unified attention with memory](dev/diagrams/02_unified_attention_with_memory.svg)

The current write path is now learned, sparse, and attention-shaped. The diagram is still a simplification of the selection bottleneck:

![Memory write selection](dev/diagrams/03_memory_write_selection.svg)

Training aligns delayed recall, branch choice, and guardrails:

![Training alignment and guardrails](dev/diagrams/04_training_alignment_and_guardrails.svg)

## Architecture We Have

The current implementation uses persistent latent memory banks inside the model. Each layer has fixed-size memory state with:

- memory values
- retrieval keys
- scalar strengths
- kind ids for anchor and summary behavior
- source-position metadata for diagnostics

At read time, each block receives memory K/V in addition to normal context K/V. Retrieval is sparse: the model selects top memory slots instead of attending densely over every slot. This keeps compute bounded and gives us inspectable active-slot behavior.

At write time, memory slots query context tokens. The writer uses learned salience, a learned surprise-like priority, recency bias, a no-write route, slot competition, diversity pressure, decay, top-k limits, and write budgets.

The read path is already generic:

```text
Q_context = project_queries(hidden_states)
K_all = concat(K_memory, K_context)
V_all = concat(V_memory, V_context)
hidden_states = attention(Q_context, K_all, V_all)
```

The write path is now symmetric in shape:

```text
Q_memory = learned_slot_queries
K_context = write_key_proj(hidden_states)
V_context = write_value_proj(hidden_states)
write_scores = Q_memory @ K_context.T + salience + surprise + recency
write_weights = sparse_softmax([write_scores; no_write])
memory = decay_and_replace(memory, write_weights @ V_context)
```

The remaining issue is not whether write is tensor-native or attention-like. It is whether the new writer trains into the right binding geometry.

## What The Experiments Proved

### Memory Is Causal

The latest useful local checkpoint family is:

```text
d12-memory-phase38-default-branch-lr3-from500 @ step 250
```

With memory enabled, teacher-forced key-token statistics were much better than with memory ablated:

| Slice | Memory key top-1 | No-memory key top-1 | Memory mean rank | No-memory mean rank |
| --- | ---: | ---: | ---: | ---: |
| Stage 1 identity | 0.742 | 0.355 | 26.7 | 266.8 |
| Stage 2 structured | 0.554 | 0.351 | 8.2 | 185.0 |
| Stage 3 mixed | 0.659 | 0.295 | 7.7 | 497.5 |

This means the memory state is doing real work. If we remove memory state or memory read, the correct key probability collapses.

### Memory Is Not Layer-0-Only Anymore

Earlier versions effectively used memory in the first block and then lost it. That was fixed by passing memory K/V into every block and by instrumenting memory attention mass per layer.

Representative layer mass at the first generated token now looks distributed:

| Layer | Memory mass at last token | Mean memory mass | Gate |
| --- | ---: | ---: | ---: |
| L2 | 0.485 | 0.429 | 0.307 |
| L6 | 0.259 | 0.157 | 0.517 |
| L10 | 0.165 | 0.083 | 0.417 |
| L4 | 0.149 | 0.080 | 0.329 |
| L3 | 0.088 | 0.234 | 0.584 |

Layer 0 is no longer the story. The memory signal is being used throughout the trunk.

### Exact Live Recall Is Still Weak

The same checkpoint family gives roughly:

| Evaluation | Exact | Core exact |
| --- | ---: | ---: |
| Deterministic, overall | 0.133 | 0.150 |
| Sampled, overall | 0.117 | 0.133 |
| Stage 1 curriculum sampled core | 0.139 | 0.139 |

This is not good enough. It is above no-memory collapse, but it is not a usable memory system.

### Failure Modes Are Now Specific

The useful failures are not random gibberish anymore. They are structured:

- Branch error: the model recalls a plausible but wrong remembered value.
- Old/new conflict: an older fact wins over the current corrected fact.
- Post-key continuation error: the first key token is right, then the word drifts.
- Template habit: the decoder falls into a memorized answer shell instead of using the specific memory.
- Repetition: the answer repeats the key or phrase after a correct start.
- Endpoint drift: a run can improve mid-way and regress by the saved endpoint.

Examples we saw:

- `Biscuit` can be locally top-1, but live output repeats it.
- `Pepper` can lose to `Juniper` even when memory is present.
- `Fig` can become `Figaro`; the first token is correct but the continuation is wrong.
- `Clover` can lose to older `Miso`; correction/currentness is still weak.

The goblin in the walls is not "no memory." It is "memory is present but not decisive enough during rollout."

## Training Path So Far

### Phase 1: Memory-Only Alignment

Goal: make the memory path useful while freezing most of the trunk.

What improved:

- key-token CE dropped substantially
- memory utility margins became positive
- anchor margins crossed positive after enough steps
- no-memory ablations showed memory was causal

What failed:

- live generation still looped or produced template gibberish in early checkpoints
- teacher-forced gains did not transfer cleanly to live recall

### Phase 2: Interface Joint Training

Goal: let memory and the decoder interface co-adapt without fully destabilizing the base model.

Added losses:

- answer-start CE
- habit-confuser margin
- stronger key-token rank and utility margins
- rollout-aware penalties

What improved:

- answer starts became more sane
- key-token ranks improved
- habit/default answers became more measurable

What failed:

- exact recall remained low
- branch selection and post-key continuation were still brittle

### Phase 2.5: Span And Post-Key Pressure

Goal: stop rewarding only the first key token and start shaping the entire answer span.

Added losses:

- span contrast margin
- span utility margin
- post-key CE
- post-key repeat margin

What improved:

- post-key CE became very low in teacher-forced settings
- span margins improved
- repetition became measurable and sometimes reduced

What failed:

- live decoding could still pick the wrong branch before the span loss helped
- some examples had the correct key top-1 under teacher forcing but failed under generation

### Phase 3: Generated-Prefix Recovery

Goal: train on prefixes the model actually produces, not only gold prefixes.

The idea was: after memorizing a fact, ask the recall question, sample a few answer tokens, detect a bad or fragile prefix, and train the model to recover toward the gold answer.

This was conceptually right, but early versions were too blunt. If the model sampled a bad path, the recovery loss sometimes pushed a hard correction without enough structure around branch choice, stop behavior, or no-memory defaults.

### Phase 3.5 To 3.7: Branch And Stop Losses

Goal: target the actual observed failure families.

Added losses:

- branch contrast
- stop margin
- no-memory default detection
- wrong-span detection
- partial-wrong detection
- repetition detection

What improved:

- diagnostics became much more informative
- several margins improved
- the model sometimes became very close to positive trajectory margins

What failed:

- stronger losses could rough up the distribution while improving a margin
- final checkpoints were often worse than the best mid-run pockets
- exact live recall stayed flat enough that we should not declare success

### Phase 3.8: Trajectory Preference And Default-Branch Pressure

Goal: prefer the whole good trajectory over the sampled bad/default trajectory.

Representative good sign:

```text
phase38_traj_margin approached zero from below
phase38_no_memory_default became measurable
phase38_wrong_span and phase38_partial_wrong exposed branch failures
```

This was a genuine conceptual improvement. It turned the problem from "make token X high under gold prefix" into "make the full answer path better than the bad sampled path."

But the run did not produce a decisive exact-recall breakthrough. The important lesson is not that trajectory preference is wrong. The lesson is that it needs a better curriculum and checkpoint selection policy.

### Default Curriculum Attempt

We tried a staged curriculum from the latest good checkpoint:

```text
d12-memory-phase38-default-branch-lr3-from500 @ 250
```

Stage 1 improved teacher-forced key metrics slightly:

- Stage 1 key top-1: 0.742 -> 0.774
- Stage 2 key top-1: 0.554 -> 0.581
- Stage 3 key top-1: 0.659 -> 0.682

But live exact recall stayed roughly flat. Mid-run logs had promising pockets, while saved endpoints regressed. This tells us to checkpoint more often and select by live evaluation windows, not by final training loss.

## What We Learned From Titans

Paper: [Titans: Learning to Memorize at Test Time](https://arxiv.org/html/2501.00663)

The most relevant Titans ideas are:

- Attention is short-term associative memory; long-term memory should be a separate but interoperable memory system.
- Long-term memory should be updated at test time, not only trained offline.
- Surprise is a useful write signal: events that violate model expectation deserve more memory.
- Forgetting should be adaptive and capacity-aware.
- Memory can be incorporated as context, as a gated branch, or as a layer; each choice trades off precision, efficiency, and coupling.
- Persistent task memory and contextual long-term memory are distinct concepts.

The direct implication for NanoChat is that our learned salience gate should not be the only notion of "what to remember." We should add surprise/residual-based write pressure:

\[
\mathrm{write\_priority}_t =
\alpha \cdot \mathrm{salience}_t +
\beta \cdot \mathrm{surprise}_t +
\gamma \cdot \mathrm{utility}_t -
\lambda \cdot \mathrm{redundancy}_t
\]

where surprise can be approximated by high CE, high key/value prediction error, or disagreement between memory and no-memory branches.

Titans also argues for evaluating memory as an online learner. For us, that means tests should include:

- write fact now, recall later
- overwrite fact now, recall latest later
- distractor facts between write and recall
- no-memory ablations
- memory-update ablations
- retention under many updates
- specificity: unrelated behavior should not degrade

What we should not copy blindly:

- Titans uses neural memory weights updated at test time. Our current memory is a latent slot bank, not a neural module whose weights are optimized online.
- Titans is primarily a long-context/sequence-model architecture paper. Our hardest failure is exact session fact binding under live chat rollout.

Still, Titans gives the right north star: memory write should be surprise-aware, online, capacity-limited, and integrated with attention rather than bolted on.

## What We Learned From MemoryLLM

Paper: [MEMORYLLM: Towards Self-Updatable Large Language Models](https://arxiv.org/abs/2402.04624)

MemoryLLM is especially relevant because it uses a fixed-size memory pool in the latent space of a transformer. Its memory pool is layerwise, self-updatable, and designed so old knowledge gradually phases out instead of growing without bound.

The relevant ideas are:

- Split static backbone parameters from self-updatable memory parameters.
- Store memory as hidden vectors inside transformer layers.
- During generation, allow normal hidden states to attend to memory tokens.
- During update, replace only a proportion of memory so capacity stays fixed.
- Evaluate not just recall, but efficacy, generalization, specificity, retention, and robustness after many updates.

This overlaps strongly with our direction. We already use a fixed-size latent memory bank and layerwise memory. The difference is that MemoryLLM treats the memory pool more like self-updatable parameters/tokens per layer, while our current design exposes retrieved K/V to attention and writes with a learned sparse controller.

MemoryLLM changes our test plan more than our immediate architecture:

- Efficacy: after one session writes a fact, does recall succeed?
- Generalization: can paraphrased questions retrieve the same fact?
- Specificity: do unrelated facts and normal chat behavior survive?
- Retention: does the fact survive after many unrelated writes?
- Conflict update: does a later correction override an earlier fact?
- Robustness: does repeated memory updating avoid distribution collapse?

The old dog-name test is necessary but not sufficient. It is one row in a bigger model-editing style evaluation.

## The Symmetric Read/Write Architecture

The clean architecture is not "add a copy head." The clean architecture is to make read and write two directions of the same associative mechanism. The first implementation of this now exists in `nanochat/gpt.py`.

Read:

\[
A_{\text{read}} = \mathrm{sparse\_softmax}(Q_C K_M^\top)
\]

\[
Y_C = A_{\text{read}} V_M
\]

Write:

\[
A_{\text{write}} =
\mathrm{sparse\_softmax}(Q_M K_C^\top + B_{\text{surprise}} + B_{\text{recency}} - B_{\text{redundancy}})
\]

\[
\Delta M = A_{\text{write}} V_C
\]

\[
M \leftarrow \mathrm{decay}(M) + \mathrm{gate} \odot \Delta M
\]

In words:

- context queries memory to read
- memory queries context to write
- both use attention-shaped competition
- write has scarcity terms so it does not store everything
- decay and replacement decide what old memory loses

This is the general version of "the attention structure determines what to read and write." It still has a bottleneck: top-k write budgets and an explicit no-write score. Pure attention-write without scarcity would write too much and become a compressed context dump. The important design constraint is that memory must be useful because it is scarce.

### Why Symmetry Should Help

The current writer can write one representation and the later reader can query a slightly different representation. That mismatch is a plausible reason for branch errors.

A symmetric writer should improve:

- binding: the slot that later answers `dog name?` is the slot that wrote from the dog-name context
- correction: new write attention can target and weaken old slots with overlapping key structure
- field separation: region, service, person, project, and pet-name slots can compete separately
- salience: surprise and retrieval utility can shape write attention directly

It will not automatically fix:

- decoder loops
- bad stop behavior
- insufficient SFT/chat ability in the base model
- all live rollout mismatch

So the symmetric architecture is necessary for elegance and likely helpful for binding, but it still needs training and rollout-aware validation.

## Next Architecture Plan

### Step 1: Keep The Read Path Stable

Do not rewrite the whole model first. Keep:

```text
Attn(Q, [K_memory; K_context], [V_memory; V_context])
```

This is the part that is already proven causal.

### Step 2: Replace The Writer With Attention-Like Slot Queries

Introduce per-layer memory-slot write queries:

```text
write_scores = q_memory_slots @ k_context_tokens.T
write_scores += surprise_bias + recency_bias - redundancy_bias
write_weights = sparse_softmax(write_scores, top_tokens, top_slots)
slot_updates = write_weights @ v_context_tokens
```

The writer should have no free-form Python rule for facts. It should be a tensor operation trained end-to-end.

### Step 3: Add Scarcity As A First-Class Constraint

Use:

- top-k context tokens per update
- top-k slots per token
- no-write slot
- diversity pressure
- age/strength decay
- overwrite penalty for unrelated memory
- anti-redundancy against existing slots

The no-write slot matters. Without it, every token must be stored somewhere, which is exactly how memory becomes sludge.

### Step 4: Add Surprise-Aware Write Bias

Borrowing from Titans, add a write-priority term from:

- next-token CE
- memory-vs-no-memory disagreement
- answer-span utility
- field/key novelty
- correction markers such as "actually", "not X, Y", "changed to"

This should be trained as a soft bias, not a hard rule.

### Step 5: Add Correction-Aware Slot Update

Correction examples need an operation closer to:

```text
old_slot = attend(memory, new_fact_key)
memory[old_slot].strength *= forget_gate
memory[new_or_old_slot] = write(new_fact_value)
```

In differentiable form, this is just attention over old memory plus a learned decay/update gate. It should not be a symbolic delete operation.

## Training Plan

### Stage A: Stabilize Current Architecture With Better Selection

Before architecture surgery, run short continuation experiments with:

- checkpoint every 25 to 50 steps
- live deterministic exact recall every checkpoint
- sampled exact recall at temperature 0.2 to 0.4
- no-memory ablation every checkpoint
- layer memory mass every checkpoint
- token probability tables for first key token and post-key continuation

Select by a score like:

\[
S =
3 \cdot \mathrm{live\_exact}
+ 2 \cdot \mathrm{sampled\_core}
+ \mathrm{key\_top1}
- \mathrm{no\_memory\_default}
- \mathrm{repeat\_rate}
- \mathrm{wrong\_span\_rate}
\]

Do not trust final-step training loss. We repeatedly saw mid-run pockets beat saved endpoints.

### Stage B: Train The Symmetric Writer

The model now reports:

```text
episodic_mode=symmetric_associative_attention
```

The old sparse write behavior is no longer the target path. Start from the original chat checkpoint and let missing memory parameters initialize fresh.

Minimum smoke tests:

- shape compatibility on CPU/MPS/CUDA
- no-memory mode still works
- memory write changes state after a write turn
- no-write route can leave memory mostly unchanged on filler text
- layerwise memory diagnostics still run

### Stage C: Train With Frozen Trunk First

Start from the original NanoChat checkpoint or the best stable memory checkpoint, depending on compatibility.

Freeze:

- token embeddings
- most trunk weights
- LM head unless needed for interface repair

Train:

- write queries
- memory key/value projections
- write/read gates
- low-rank interface adapters

Losses:

- answer CE
- key-token CE
- memory utility margin
- span contrast margin
- post-key CE
- no-memory default contrast
- correction/currentness margin
- guardrail KL

### Stage D: Add Rollout Training Only After Teacher-Forced Memory Is Stable

Rollout training should not be the first hammer. It is expensive and noisy.

Once teacher-forced and ablation metrics are good:

- sample at temperature 0.2 first
- detect first bad token or bad branch
- train recovery on the next 8 to 16 tokens
- gradually increase temperature to 0.4
- include "recover after one or two bad tokens" examples

This is the user's idea and it is correct: train on real chat failure prefixes, not only clean gold prefixes. It belongs after the memory mechanism reliably places the fact into the logits.

### Stage E: Add Titans-Inspired Surprise Write

Train surprise write in two forms:

- differentiable proxy during normal supervised batches
- online update simulation during write-then-recall tasks

Tests:

- surprising fact should be written more than filler
- repeated filler should not overwrite useful facts
- correction should update current fact more than duplicate old fact
- unrelated chat should not decay recent important memory too quickly

### Stage F: Add MemoryLLM-Inspired Self-Update Benchmarks

Create a benchmark family with:

- single-fact injection
- paraphrased recall
- multi-fact retention
- conflict update
- stale fact rejection
- unrelated specificity
- many-update robustness

Report:

- efficacy
- generalization
- specificity
- retention
- conflict accuracy
- live exact recall
- sampled exact recall
- memory ablation delta

## Evaluation Standard

A checkpoint is not good because one loss went down. It is good only if it passes all of:

- Live deterministic exact recall improves.
- Sampled recall at small temperature improves.
- No-memory ablation remains much worse.
- Correct key tokens have high probability under generated prefixes.
- Post-key continuation and stop behavior are sane.
- Old/new correction improves.
- Normal chat guardrails do not collapse.
- Memory use is distributed across useful layers, not an artifact of one block.

Minimum dog-name test:

```bash
PYTHONPATH=$PWD python scripts/chat_cli.py \
  -i sft -g <checkpoint_group> -s <step> \
  --session-id dog-demo --forget-session \
  -t 0 -k 1 --max-tokens 16 \
  -p "My dog's name is Pebble-Cloud."

PYTHONPATH=$PWD python scripts/chat_cli.py \
  -i sft -g <checkpoint_group> -s <step> \
  --session-id dog-demo \
  -t 0 -k 1 --max-tokens 16 \
  -p "What's my dog's name?"
```

But this is only a smoke test. The real evaluation is the multi-slice suite with token probabilities, layer probes, and no-memory ablations.

## What We Should Stop Doing

Stop treating a positive margin as success by itself. Margins can improve while the distribution gets rougher.

Stop saving only endpoint checkpoints. The model has shown mid-run pockets that are better than the final saved state.

Stop adding narrow copy logic. If a patch cannot be expressed as normal attention, normal logits, or a differentiable memory update, it is probably the wrong direction.

Stop relying on teacher forcing as the main signal. It is necessary but insufficient.

Stop ignoring correction/currentness. Memory that cannot update is just a polite archive with bad manners.

## Current Implementation Status

Implemented:

- sparse unified associative attention read
- symmetric attention-like write from memory slots to context tokens
- explicit no-write route for memory scarcity
- learned surprise-like write priority and recency bias
- fixed-size self-update with decay plus replacement
- latent persistent memory banks
- per-layer memory K/V injection
- memory write salience and slot competition
- active slot diagnostics
- memory/no-memory ablations
- key-token probability diagnostics
- rollout and trajectory diagnostics
- phase 1, 2, 2.5, 3, 3.5, 3.6, 3.7, and 3.8 training harnesses
- symmetric-memory GPU launcher: `dev/run_symmetric_memory_training.sh`
- V4 high-variety live-session, branchpoint, no-memory, irrelevant-memory, and normal-chat curriculum shards
- V5 arbitrary-character and arbitrary-token exact-copy curriculum shards
- phase-specific batch sizing so full-trunk adaptation can run at a smaller batch inside the same long training script
- diagrams for normal attention, unified memory attention, write selection, and training alignment

Not implemented:

- true Titans-style test-time neural-memory weight updates
- CE-derived surprise write scores; the current surprise signal is a learned proxy
- correction-aware differentiable overwrite
- MemoryLLM-style full self-update retention benchmark
- automatic best-window checkpoint selection
- event-level memory routing or a pointer distribution over retrieved memory tokens

## Recommended Next Step

The next serious step is V5 full-copy training from the best current symmetric
checkpoint, not another narrow dog-name patch:

1. Stop the current V4 continuation once the latest good checkpoint is safe.
2. Generate the V5 curriculum with arbitrary character strings, token-random spans, broader memory fields, no-memory rows, irrelevant-memory rows, and real SmolTalk guardrails.
3. Train a 100k-step A/B/C schedule:
   - A: memory/interface exact-copy stabilization
   - B: short low-LR full-model arbitrary-copy adaptation
   - C: memory/interface trajectory and guardrail polish
4. Keep checkpoint pruning on and evaluate checkpoints with live two-process recall, no-memory ablation, token probabilities, and layer memory usage.
5. If V5 still plateaus with negative trajectory margins, then change architecture rather than add more loss pressure:
   - event-level memory routing: retrieve top-k written events, then attend inside those events
   - explicit recency/currentness features for correction cases
   - a generic pointer distribution over retrieved memory tokens

This keeps the core promise intact: a generic transformer attention system with
memory, trained to copy arbitrary remembered data, not a hand-coded dog-name
lookup table wearing a trench coat.

## References

- [Titans: Learning to Memorize at Test Time](https://arxiv.org/html/2501.00663)
- [MEMORYLLM: Towards Self-Updatable Large Language Models](https://arxiv.org/abs/2402.04624)
