# Long-Term Memory for NanoChat

## Purpose

NanoChat already has strong short-term working memory through ordinary causal attention over the current context window.

The long-term memory problem is narrower:

\[
p_\theta(y \mid x, M) \approx p_\theta(y \mid x, H)
\]

where:

- `x` is the current query
- `H` is the relevant past context that is no longer in the prompt
- `M` is persistent memory written from earlier interactions
- `y` is the answer sequence

The goal is not to bolt on a retrieval branch or a special-case copy system. The goal is to make persistent memory behave as much like missing context as possible while staying inside a generic transformer-style associative framework.

This document is the current design memo for that goal. It summarizes the relevant evidence from earlier iterations, explains the present architecture and its failure modes, and lays out the next model and training changes.

## Executive Summary

The current verdict is:

1. The right architectural family is still **latent persistent memory read through transformer-style attention**.
2. The current sparse-memory-token implementation is **not yet the right instantiation** of that idea.
3. The biggest problem is no longer "memory is dead." The biggest problem is now a **train/inference mismatch plus an objective mismatch**:
   - training builds memory one way
   - live chat writes memory another way
   - the loss improves gold-token probability
   - but does not force the correct token to actually win generation
4. Training longer with the current setup is not the answer. The long GPU run improved training metrics but over-specialized and did not fix held-out free generation.

So the next step is not "throw away memory" and not "just keep training." The next step is to keep the unified attention-with-memory direction while changing the write dynamics, retrieval interface, and losses so the training problem matches the live recall problem.

The code in `nanochat-felix` now implements the first half of that plan:

- functional online memory updates shared by training and runtime
- replay-built memory state from actual turn sequences
- a stable pre-answer recall mask/query path
- explicit anchor/confuser supervision
- a lexical auxiliary on the memory read
- completed-turn session writes as the default live behavior
- rollout-aware recovery training for short generated prefixes

What is still missing is a full rollout-aware evaluation loop and longer training evidence. The trainer can now optimize against short autoregressive drift, but checkpoint selection still needs to treat live short-horizon generation as a first-class metric.

## Design Target

The target remains a generic associative-attention system:

- ordinary context-window tokens are one associative store
- persistent memory is another associative store
- both are read through the same attention-style mechanism
- the main difference between them is the write/update rule, not the read operator

This is the transformer/Hopfield compatibility point in practical form:

\[
\mathrm{Attn}(Q, K, V) = \mathrm{softmax}(QK^\top)V
\]

The read law is the same. The real design question is how persistent state is written, updated, selected, and supervised.

## What Earlier Iterations Taught Us

### Early Latent-Memory Recall Was Real

The earlier `r18`-style success matters. It showed that narrow delayed recall of specific facts could emerge from persistent latent memory without querying an external retrieval service.

That result still constrains the design space. It means:

- persistent latent memory is viable
- the project does not need to default to an external symbolic database
- the right next move should stay close to learned runtime memory rather than abandoning it

### Branchy Rescue Architectures Were Informative but Not the Destination

Plan/payload, workspace, and copy-style branches taught useful lessons:

- the first fact token matters disproportionately
- later fact subtokens can improve even when the overall answer still fails
- local token help is not enough if the sequence-level interface is wrong

But those architectures were too ad hoc. They helped isolate bottlenecks, not define the final memory system.

### Sparse Memory Tokens Fixed an Old Failure Mode

The more recent sparse-memory-token architecture fixed two real issues from previous versions:

- gates were no longer effectively shut
- memory banks were no longer trivially rank-1 collapsed

That was real progress. The memory path became active, sparse, and trainable.

### But the New Failure Mode Is Different

The latest runs show a stronger and more precise diagnosis:

- memory is active
- memory often improves the correct token's logprob under teacher forcing
- yet free generation still fails and often degenerates into gibberish or repetition

So the problem is not "memory is absent." The problem is:

- memory is written differently in training and live inference
- retrieval drifts when generated prefixes drift
- the loss rewards gold-token uplift more than actual argmax correctness
- the retrieved memory read is not lexicalized enough to drive clean token selection

## Current Architecture

The live architecture today uses persistent latent memory tokens per layer.

These are:

- not prompt tokens
- not model weights
- runtime latent vectors stored outside the fixed trunk
- saved and restored as session state

At a high level:

1. The trunk processes the current sequence.
2. A shared episodic controller scores write salience and projects hidden states into memory-space.
3. Write-time pooling compresses selected token states into a fixed latent bank.
4. Sparse budgets keep only a limited number of summary and anchor slots active.
5. At read time, a retrieval query selects a subset of memory slots.
6. The current token state attends over those retrieved memory slots through the same attention-style K/V mechanism used by the transformer.

This preserves the right high-level principle: memory is part of associative attention, not a side output head pretending to be memory.

## Current Implementation State

The current `nanochat-felix` implementation should be understood as a partially completed version of the next design, not as the old system.

### Implemented Now

The following changes are already live in code:

1. **Canonical online memory update rule**
   - the persistent memory module now exposes a functional state update
   - the same update logic is used for runtime writes and for training-time replay construction

2. **Replay-based memory building**
   - training no longer has to rely on a purely separate one-shot mental model of memory
   - prior context can be replayed as:
     - `user`
     - `turn`
     - `turn_recall`
     - `full_context`

3. **Stable pre-answer recall conditioning**
   - memory subset selection is now anchored to a prefix mask up to the current answer start
   - this is still heuristic, but it is much closer to the real retrieval problem than conditioning entirely on the evolving generated suffix

4. **Anchor/confuser loss**
   - the strongest wrong token at the anchor now matters directly in training

5. **Lexical auxiliary on the memory read**
   - the diagnostic memory-read logit path is now lightly supervised

6. **Improved live write default**
   - chat now defaults to completed-turn memory writes rather than user-only writes

7. **Rollout-aware recovery loss**
   - the trainer can generate a short answer prefix from the model's own logits
   - it then supervises the gold continuation under that generated prefix
   - the rollout path starts with greedy generation by default and supports later low-temperature sampling
   - losses are upweighted when the generated prefix misses an expected key token/span or falls into a short repetition loop

### Not Implemented Yet

The following parts of the plan are still missing:

1. **Rollout-aware validation as the primary selection criterion**
   - the code still needs a clean evaluation loop that treats short-horizon free generation as a first-class checkpoint selection signal

2. **Learned replacement for the current recall-mask heuristic**
   - the present stable prefix mask is a practical fix, not the final theory

3. **Evidence from a serious GPU run**
   - the local 500-step MPS run showed stable mechanics but no behavioral success yet
   - the next question is whether the new aligned trainer plus rollout recovery crosses over in a longer Phase 1 GPU run

## The Current Implementation Gap

The architecture family is still reasonable. The concrete implementation is not yet aligned with the actual task.

There are three different "memory modes" hiding inside the current system:

1. **Training full-context build**
   - memory is constructed in one shot from the whole prior context
2. **User-only one-shot build**
   - memory is built from a narrower user-only subset
3. **Live sequential write**
   - memory is written incrementally turn by turn during chat

These are not equivalent, and the experiments show that clearly.

## What The Newest Evidence Says

### 1. Teacher-Forced Full-Context Memory Helps

On the best held-out checkpoint so far (`step 3000`), using the training-aligned full-context memory build improved teacher-forced gold-token probabilities substantially.

On a 20-example probe:

- control/template positions: mean `dlogp` about `+1.30`
- fact positions: mean `dlogp` about `+4.27`
- fact-token rank gain was very large

So the memory signal is real. This is not a dead path.

### 2. But Correct-Token Uplift Is Not Translating Into Correct Generation

That same checkpoint still failed badly at top-1 token selection.

On the same held-out probe, top-1 correctness under full-context memory was:

- control positions: base about `53%`, memory about `3%`
- fact positions: base about `16%`, memory about `0%`

This is the central mismatch in one line:

> memory raises the correct token's probability, but still does not make the correct token win.

That is exactly why free generation still produces nonsense.

### 3. Live Sequential Writing Is Much Worse Than Training-Style Memory

The clearest failure is the difference between training-style memory construction and live chat writes.

On an 8-example comparison:

- full-context build improved both control and fact teacher-forced logprobs
- user-only one-shot memory was clearly worse
- live sequential user-turn writes were catastrophic, with both control and fact positions going strongly negative relative to no-memory

That means two separate mismatches exist:

1. **content mismatch**
   - user-only memory is weaker than richer prior context
2. **write-dynamics mismatch**
   - online incremental writing behaves much worse than one-shot memory construction

### 4. The Best Held-Out Checkpoint Was Early

The long GPU run improved training metrics dramatically through `step 10000`:

- lower loss
- high `memory_utility_margin`
- high `anchor_margin`
- lower apparent active slot count

But held-out token behavior was best around `step 3000`, then worsened by `4000` and `10000`.

So the current recipe over-optimizes the training objective without producing the best live recall behavior.

### 5. Interactive Chat Confirms the Same Diagnosis

The local two-session chat test with the `step 3000` checkpoint showed:

- session persistence worked
- memory was written and loaded correctly
- the model still generated gibberish

That is important because it removes a simpler excuse. The current problem is not broken session persistence. The current problem is that the model still uses memory badly under live autoregressive generation.

## First-Principles Diagnosis

There are four core issues.

### 1. Training and Inference Use Different Memory Dynamics

Training currently approximates something like:

\[
M = B_\theta(H_{\text{full prior context}})
\]

while live chat actually uses:

\[
M_{t+1} = U_\theta(M_t, H_t)
\]

where:

- \(B_\theta\) is a one-shot memory builder
- \(U_\theta\) is the incremental runtime writer

These are not the same operator.

If the goal is a generic transformer attention system with memory, then the same write/update rule must be used in training and inference. Right now it is not.

### 2. Retrieval Query Drift Makes Teacher Forcing Too Optimistic

The current retrieval subset is conditioned on evolving hidden state that includes generated prefix information.

Under teacher forcing, that prefix is gold.
Under free generation, one wrong early token changes the hidden state and retrieval shifts with it.

So the model is being trained and diagnosed under a much friendlier retrieval condition than it faces when it actually has to answer.

### 3. The Loss Optimizes Gold-Token Lift More Than Winning the Token Decision

The current utility objective is structurally close to:

\[
\log p_\theta(y_t \mid x, M) > \log p_\theta(y_t \mid x, \varnothing)
\]

But generation needs something closer to:

\[
z_t(y_t) > \max_{j \neq y_t} z_t(j)
\]

The current system can satisfy the first condition without satisfying the second.

That is exactly what the newest probes show:

- positive `dlogp`
- poor top-1 correctness
- gibberish free generation

### 4. The Retrieved Memory Read Is Not Lexical Enough

The memory path is useful enough to bias logits, but the read itself is still not a clean lexical fact object.

This is why the model often moves toward bizarre attractors such as:

- `stream`
- `function`
- `Democratic`
- `United`
- repetitive junk like `regard regard ...`

The retrieved latent is influencing the distribution, but not in a directly decodable way.

## Decision

The correct conclusion is:

- **do not abandon the attention-based latent memory direction**
- **do not keep training the current setup unchanged**
- **do change the write dynamics, retrieval interface, and objectives**

In other words:

- the architectural family is still promising
- the current instantiation is not yet right

## Next Architecture and Training Plan

### 1. Unify Training and Inference Around the Same Writer

This is the most important change.

This is now implemented in `nanochat-felix`.

The training harness no longer has to rely only on one-shot full-context memory build as the main memory path. It can train with replay through the same online write/update logic that live chat uses.

That means:

- replay prior turns through the real writer during training
- treat persistent memory state as an evolving recurrent external state
- use the same slot-update rule in both training and inference

If the current writer is too awkward to train through directly, then the writer itself should be refactored into a cleaner functional state-update operator and used everywhere.

### 2. Use a Stable Recall Query for Answer-Time Memory Selection

Memory subset selection should not be driven entirely by the evolving generated prefix.

Instead:

1. compute a stable recall query from the pre-answer state, especially the user question
2. select a memory subset once for the answer
3. let token-local attention operate over that fixed subset during generation

Conceptually:

\[
S = \mathrm{TopK}(q_{\text{query}} K_M^\top)
\]

then during decoding:

\[
r_t = \mathrm{Attn}(h_t, K_S, V_S)
\]

This keeps memory relevant to the question while reducing retrieval drift after one bad generated token.

This is also now implemented in a first practical form. The current code uses a stable pre-answer prefix mask to derive the recall query. That is still heuristic, but it is already better aligned than letting the generated answer suffix fully steer subset retrieval.

### 3. Turn `anchor_margin` Into a Real Loss

`anchor_margin` is currently one of the most informative diagnostics, but it is not the core training signal it should be.

It needs to become an explicit max-confuser loss:

\[
L_{\text{anchor}} = \max(0, m - (z_y - \max_{j \neq y} z_j))
\]

applied at:

- the first fact token
- and likely a few subsequent fact tokens

This is more aligned with generation than only pushing up the gold token against curated negatives.

This is now implemented for the early fact tokens. The current trainer still uses a simple fixed early-token window rather than a more adaptive sequence-level version, so this part is improved but not finished.

### 4. Keep Guardrails, but Make Them Answer-Trajectory Aware

The guardrail KL is still useful. It protects non-fact answer structure from gratuitous memory distortion.

It now has help from a short rollout-aware objective, but still needs a proper rollout-aware validation loop.

At minimum, the system should evaluate short autoregressive rollouts for the first few answer tokens. Even a very short horizon is more aligned with the actual failure mode than pure teacher forcing.

The current failure is sequence-level drift after early token mistakes. The training signal needs to see that.

The trainer now includes a first practical version of that signal: it can generate a short answer prefix, feed that generated prefix back in, and train on the gold continuation. The next missing piece is making that same rollout path a primary validation and checkpoint-selection signal.

### 5. Add a Small Lexical Decodability Auxiliary on the Memory Read

The top memory read already has a diagnostic logit path, but it is not being trained as a meaningful lexical object.

A low-weight auxiliary loss on the memory-read logits at fact positions should help make memory more directly decodable without reverting to a branchy explicit copy architecture.

This remains compatible with the generic attention-with-memory goal. It is not a special-case retrieval head; it is an auxiliary shaping signal.

This is now implemented in lightweight form.

### 6. Write Better Runtime Content by Default

The experiments say user-only live memory is worse than richer full-context memory.

So runtime memory writing should move closer to what actually helps:

- write completed turns rather than only the user prompt
- or write a reconsolidated turn summary plus anchor traces
- or both

The default chat write policy should not stay in the weakest regime if the training evidence already shows it is mismatched.

This is now implemented in the chat surfaces by making completed-turn writes the default and keeping user-only writes as an explicit debug option.

## Training Strategy

The previous trunk-heavy schedules were wrong.

The next serious GPU run should be organized like this.

### Phase 1: Memory and Write/Read Path Only

- freeze the base trunk
- train the memory controller, writer, retriever, and memory interface
- train on the actual online memory dynamics
- keep guardrails active

The memory path needs to become useful before the trunk is allowed to adapt around it.

This phase remains the correct default first training phase.

### Phase 2: Interface Joint

- unfreeze a small interface set
- keep most of the trunk frozen
- let the model refine how memory enters the answer trajectory

This phase should be conservative and should only begin if held-out live recall is improving.

### Phase 3: Optional Full Joint Release

- lowest learning rate
- only after live sequential recall and anchor behavior are clearly improving

This is a release phase, not the place where the memory mechanism should first start working.

## Rollout Training Plan

The trainer now has rollout-aware recovery loss. It should still be used as a curriculum, not turned into noisy sequence training immediately.

### Stage 1: Greedy or Near-Greedy Rollouts

Start with:

- `temperature = 0`
- or effectively `top_k = 1`

The point is to expose the model to its own most likely early mistakes, not to inject lots of random noise.

This stage should:

- generate the first few answer tokens from the model itself
- feed those generated prefixes back into the model
- apply losses to the subsequent positions
- become a key validation path for checkpoint selection
- upweight the loss if the generated prefix should contain the key datum but does not
- upweight the loss if the prefix falls into short repetition

### Stage 2: Low-Temperature Stochastic Rollouts

Only after the deterministic rollout path is working and no longer degenerates should training add small stochasticity.

That means:

- low temperature only
- roughly in the `0.2 - 0.4` range
- likely still with a small `top_k`

The purpose of this stage is not diversity for its own sake. The purpose is robustness around the model's real decision boundary once the deterministic path is already stable.

High-temperature rollout training should not be the starting point. It would inject too much noise and blur the actual recall failure we are trying to fix.

## What To Measure

The next training loop should treat these as primary metrics:

- held-out live sequential-write recall
- held-out short-horizon generation after memory write
- first fact-token top-1 accuracy
- next few fact-token top-1 accuracy
- control-token top-1 accuracy
- `anchor_margin`
- `memory_utility_margin`
- guardrail KL
- active slot count and retrieval entropy

Raw loss alone is not enough, and teacher-forced gold-token logprob alone is not enough either.

During the next major run, rollout metrics should be added in this order:

1. greedy short-horizon rollout accuracy
2. greedy short-horizon rollout degeneration rate
3. low-temperature short-horizon rollout accuracy
4. missing-key rate within the rollout horizon

The low-temperature version should only become important after the greedy version is already healthy.

## What To Stop Doing

The current evidence says we should not:

- keep training the current setup and hope gibberish disappears
- reintroduce old copy/workspace branch architectures as the main path
- treat full-context teacher-forced gains as proof that live chat is fixed
- optimize only curated hard-negative losses without real max-confuser supervision
- keep the live chat writer in a user-only incremental mode if that continues to be the weakest setting

## Concrete Repo Plan

The next code changes should center on these files.

### `nanochat/gpt.py`

- separate stable answer-level recall query from token-local retrieval
- stop letting evolving generated hidden state fully control memory subset selection
- expose the memory read path cleanly for auxiliary supervision

### `nanochat/episodic_memory.py`

- make the online writer the canonical state-update mechanism
- ensure the same update logic is used in training and inference
- keep sparsity and budget logic, but make it compatible with replayable training

### `scripts/chat_memory.py`

- train with replay through the actual online writer
- add explicit anchor/confuser loss
- add lexical auxiliary on the memory read
- add rollout-aware recovery loss
- add rollout-aware evaluation and early stopping
- stop optimizing for memory behaviors that only exist in one-shot build mode

At the time of writing:

- replay training is implemented
- anchor/confuser loss is implemented
- lexical auxiliary is implemented
- rollout-aware recovery training is implemented
- rollout-aware evaluation and early stopping are still missing

### `scripts/chat_cli.py`

- move the default write behavior toward completed-turn or reconsolidated memory writes
- keep session persistence simple and inspectable

## Status

The current sparse-memory-token architecture was an important intermediate step. It showed that:

- memory can be active
- sparsity matters
- the right architectural family is not dead

But it also showed that the current system still fails the real objective:

- coherent live recall without gibberish

The repository state is therefore:

- architecture direction: still correct
- runtime/training memory alignment: substantially improved
- token-level supervision: substantially improved
- rollout-aware recovery supervision: implemented in the trainer
- rollout-aware evaluation and checkpoint selection: still missing
- live generation quality: not solved yet

So the next experiment is now well defined.

The question is no longer:

> can NanoChat have latent persistent memory at all?

The question is:

> can NanoChat train a single generic attention-with-memory system whose online write dynamics, retrieval interface, and token-level objectives are aligned closely enough that live sequential recall behaves like missing context instead of latent noise?

That is the right next design problem.
