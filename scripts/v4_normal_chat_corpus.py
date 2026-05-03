"""
Curated ordinary-chat guardrail corpus for V4 memory training.

The memory curriculum needs broad non-memory language so the model does not
learn that every user turn should be answered as a recall task. This file keeps
that ordinary-chat material out of the memory generator itself and makes the
coverage auditable. Rows are deterministic, standalone prompt/answer pairs;
they are not memory-fact templates.
"""


def dedupe(rows):
    return list(dict.fromkeys(rows))


CONCEPT_ROWS = [
    ("batching", "Batching groups several items into one operation so hardware spends more time doing useful parallel work.", "It improves throughput when fixed overhead is large compared with per-item work.", "It can add latency if requests wait too long for a batch to fill."),
    ("learning-rate warmup", "Learning-rate warmup starts with small optimizer steps before using the target learning rate.", "It helps stabilize early training when activations and optimizer statistics are still settling.", "Too much warmup can waste steps after the model is already stable."),
    ("gradient clipping", "Gradient clipping limits unusually large gradients before the optimizer applies an update.", "It can prevent one unstable batch from pushing parameters too far.", "It does not fix a bad objective; it only bounds the update size."),
    ("validation loss", "Validation loss measures performance on held-out examples that were not used for updates.", "It is a useful signal for generalization and overfitting.", "A noisy validation set can make short-term changes look more meaningful than they are."),
    ("overfitting", "Overfitting means a model fits training examples better than it fits new examples.", "It matters because the model may memorize quirks instead of learning the intended pattern.", "More data, regularization, and early stopping can reduce it."),
    ("regularization", "Regularization adds constraints or noise that discourage brittle memorization.", "It helps the model prefer simpler patterns that generalize better.", "Too much regularization can underfit and hide useful signal."),
    ("dropout", "Dropout randomly disables some activations during training.", "It makes the network less dependent on any single path.", "It is usually disabled during evaluation and inference."),
    ("weight decay", "Weight decay gently penalizes large parameter values.", "It can make optimization prefer smaller, smoother solutions.", "The best value depends on model size, data, and optimizer details."),
    ("temperature sampling", "Temperature changes how sharply probabilities are used when sampling tokens.", "Lower temperature is more deterministic; higher temperature explores more alternatives.", "Very high temperature can make answers incoherent."),
    ("top-k sampling", "Top-k sampling restricts choices to the k most likely next tokens.", "It can reduce bizarre low-probability outputs while keeping some variety.", "If k is too small, the output can become repetitive or overly conservative."),
    ("beam search", "Beam search keeps several likely partial sequences while generating text.", "It can improve short structured outputs where likelihood tracks quality.", "For open-ended chat it can sound bland or over-optimized."),
    ("teacher forcing", "Teacher forcing trains a model on gold previous tokens instead of its own generated tokens.", "It makes next-token training efficient and stable.", "It can leave a gap between training and free-running generation."),
    ("scheduled sampling", "Scheduled sampling mixes model-generated tokens into training contexts.", "It can teach recovery from imperfect prefixes.", "If introduced too early, it can destabilize learning."),
    ("cross entropy", "Cross entropy penalizes the model when it assigns low probability to the target token.", "It is the standard objective for next-token prediction.", "It measures token likelihood, not every aspect of answer quality."),
    ("KL divergence", "KL divergence measures how one probability distribution differs from another.", "It is useful for keeping a finetuned model close to a reference distribution.", "It is directional, so swapping the two distributions changes the value."),
    ("margin loss", "A margin loss asks the correct option to beat a competitor by some minimum gap.", "It is useful when ranking relative choices matters more than raw probability alone.", "A margin that is too large can rough up the whole distribution."),
    ("attention", "Attention lets a token combine information from other tokens by scoring their relevance.", "It gives the model a flexible way to retrieve context-dependent information.", "Attention over long contexts can be expensive."),
    ("self-attention", "Self-attention lets tokens in the same sequence attend to one another.", "It is the main mechanism behind transformer context mixing.", "Without positional information, it does not know token order."),
    ("positional encoding", "Positional encoding gives the model information about token order.", "It lets attention distinguish the same word in different places.", "Different schemes have different extrapolation behavior."),
    ("KV cache", "A KV cache stores previous attention keys and values during generation.", "It avoids recomputing the whole prefix for every generated token.", "It becomes more complex when extra persistent memory tokens are involved."),
    ("embedding", "An embedding maps a discrete item into a continuous vector.", "It lets models compare and combine symbolic inputs numerically.", "Similar embeddings are useful only if training shaped them that way."),
    ("normalization", "Normalization rescales activations to make optimization easier.", "It can improve stability and reduce sensitivity to scale.", "The exact placement can strongly affect transformer behavior."),
    ("residual connection", "A residual connection adds a layer's input back to its output.", "It helps gradients flow through deep networks.", "Residual paths can also preserve unwanted behavior if losses do not counter it."),
    ("activation function", "An activation function adds nonlinearity to a neural network.", "Without nonlinearities, stacked layers collapse toward a linear transformation.", "Different activations trade off speed, smoothness, and saturation."),
    ("optimizer state", "Optimizer state stores extra statistics such as momentum or variance estimates.", "It helps optimizers choose better update directions and scales.", "Losing optimizer state can change training dynamics after a restart."),
    ("gradient accumulation", "Gradient accumulation sums gradients over several smaller batches before stepping.", "It simulates a larger batch when memory is limited.", "It is slower than a true parallel batch if hardware has room."),
    ("mixed precision", "Mixed precision uses lower-precision numbers for speed while preserving key computations.", "It can greatly improve GPU throughput and memory use.", "Some operations still need care to avoid overflow or underflow."),
    ("checkpointing", "Checkpointing saves model state so training can resume or be inspected later.", "It protects long runs from interruptions.", "Saving too often can waste disk space and time."),
    ("early stopping", "Early stopping ends training when validation quality stops improving.", "It can prevent overfitting and save compute.", "It depends on a validation signal that matches the real goal."),
    ("ablation", "An ablation removes or changes one component to measure its contribution.", "It helps separate real improvements from noise.", "A bad ablation setup can answer the wrong question."),
    ("control group", "A control group is the baseline condition used for comparison.", "It helps show whether a change actually caused an effect.", "The control must be comparable to the experimental condition."),
    ("statistical noise", "Statistical noise is random variation that obscures the underlying signal.", "It matters because short runs can look better or worse by chance.", "Repeated measurements reduce the chance of chasing noise."),
    ("confidence interval", "A confidence interval gives a range of plausible values for an estimate.", "It communicates uncertainty better than a single number.", "It does not guarantee the true value is inside for one specific experiment."),
    ("latency", "Latency is the time it takes for one operation or request to finish.", "It matters for interactive user experience.", "Average latency can hide slow tail cases."),
    ("throughput", "Throughput is how much work a system completes per unit of time.", "It matters when serving many requests or processing large batches.", "High throughput does not always mean each request is fast."),
    ("tail latency", "Tail latency measures slow outlier requests, such as the 95th or 99th percentile.", "It matters because users often feel the slow cases most.", "It can be caused by queues, retries, cold starts, or noisy neighbors."),
    ("backpressure", "Backpressure slows incoming work when downstream systems are overloaded.", "It prevents queues from growing without bound.", "If applied poorly, it can move the bottleneck instead of fixing it."),
    ("queue", "A queue stores work until a consumer can process it.", "It decouples producers from consumers that run at different speeds.", "A growing queue is often a symptom of insufficient processing capacity."),
    ("load balancer", "A load balancer distributes requests across healthy service instances.", "It improves availability and spreads work.", "It still needs good health checks and capacity planning."),
    ("health check", "A health check reports whether a service instance is alive or ready.", "It helps orchestration systems route traffic safely.", "A shallow health check can miss broken dependencies."),
    ("canary deployment", "A canary deployment sends a small amount of traffic to a new version first.", "It reduces blast radius if the new version has a bug.", "It needs monitoring that catches the failure mode quickly."),
    ("blue-green deployment", "Blue-green deployment switches traffic between two complete environments.", "It can make rollback quick and clean.", "It requires enough infrastructure to run both environments."),
    ("rollback", "A rollback returns a system to a known-good version.", "It is essential when a release causes production problems.", "Database changes can make rollback harder if they are not planned carefully."),
    ("feature flag", "A feature flag lets a team enable or disable behavior without redeploying.", "It supports gradual rollout and fast mitigation.", "Old flags should be cleaned up or they become confusing."),
    ("idempotency", "Idempotency means repeating an operation has the same effect as doing it once.", "It makes retries safer after partial failures.", "It usually requires careful request identifiers or state checks."),
    ("retry", "A retry repeats an operation after a failure.", "It can handle transient errors without user intervention.", "Retries need limits, backoff, and timeouts to avoid amplifying outages."),
    ("timeout", "A timeout stops waiting after a configured amount of time.", "It keeps failures bounded and prevents threads from hanging forever.", "Too short a timeout can fail healthy slow requests."),
    ("circuit breaker", "A circuit breaker temporarily stops calls to a failing dependency.", "It protects the rest of the system from repeated slow failures.", "It needs recovery logic so the dependency can be tried again later."),
    ("rate limit", "A rate limit caps how many requests can happen in a time window.", "It protects services from overload and abuse.", "A strict limit can frustrate legitimate bursts if not designed carefully."),
    ("cache", "A cache stores reusable results closer to where they are needed.", "It can reduce latency and load on expensive systems.", "It must handle freshness and invalidation carefully."),
    ("cache invalidation", "Cache invalidation removes or refreshes cached data when it may be stale.", "It keeps fast reads from returning old answers.", "It is hard because data can change through many paths."),
    ("database index", "A database index is a lookup structure that speeds up reads.", "It helps queries avoid scanning every row.", "Indexes use storage and can slow writes."),
    ("transaction", "A transaction groups operations so they succeed or fail together.", "It protects data consistency across related changes.", "Long transactions can block other work."),
    ("deadlock", "A deadlock happens when tasks wait on each other in a cycle.", "It stops progress until something is interrupted.", "Consistent lock ordering can reduce the risk."),
    ("race condition", "A race condition happens when timing changes the result of operations.", "It appears when shared state is accessed without enough coordination.", "Tests may miss it because timing bugs are intermittent."),
    ("schema migration", "A schema migration changes the structure of stored data.", "It lets software evolve while keeping data usable.", "Safe migrations often need backward-compatible intermediate steps."),
    ("eventual consistency", "Eventual consistency means replicas may differ briefly but converge later.", "It improves availability in distributed systems.", "Users may see stale data for a short time."),
    ("strong consistency", "Strong consistency means reads reflect the latest committed write.", "It simplifies reasoning for users and developers.", "It can cost latency or availability in distributed systems."),
    ("replication", "Replication copies data across multiple locations or nodes.", "It improves availability and read capacity.", "It introduces lag and consistency tradeoffs."),
    ("sharding", "Sharding splits data across multiple partitions.", "It helps scale storage and traffic.", "It makes cross-shard queries and rebalancing more complex."),
    ("observability", "Observability is the ability to infer system behavior from logs, metrics, and traces.", "It helps teams diagnose problems they did not predict exactly.", "It requires useful instrumentation, not just more data."),
    ("metric cardinality", "Metric cardinality is the number of unique label combinations in metrics.", "High cardinality can make metrics expensive and hard to query.", "Unbounded labels like user IDs are usually risky."),
    ("trace", "A trace follows one request through multiple services.", "It helps find where time was spent or where an error appeared.", "Sampling can miss rare requests."),
    ("log level", "A log level describes severity, such as debug, info, warning, or error.", "It helps control noise and focus during incidents.", "Too many error logs can make real problems harder to see."),
    ("structured logging", "Structured logging records fields in a machine-readable shape.", "It makes search, filtering, and alerting easier.", "It requires consistent field names to be useful."),
    ("authentication", "Authentication verifies who a user or service is.", "It answers the question of identity.", "It does not by itself decide what the identity may access."),
    ("authorization", "Authorization decides what an authenticated identity is allowed to do.", "It protects resources from inappropriate access.", "Rules should be checked near the protected operation."),
    ("least privilege", "Least privilege gives a user or service only the access it needs.", "It reduces damage if credentials are misused.", "Permissions need regular review as systems change."),
    ("two-factor authentication", "Two-factor authentication asks for a second proof in addition to a password.", "It makes stolen passwords less useful.", "Recovery flows still need careful protection."),
    ("phishing", "Phishing tricks people into revealing secrets or granting access.", "It often imitates a trusted sender or website.", "Clear verification habits reduce the risk."),
    ("encryption", "Encryption transforms readable data into protected data using a key.", "It protects confidentiality when data is stored or transmitted.", "Key management is as important as the algorithm."),
    ("hashing", "Hashing maps data to a fixed-size digest.", "It is useful for checksums, lookup, and password verification when used correctly.", "A hash is not encryption because it is not meant to be reversed."),
    ("salt", "A salt is random data added before hashing a password.", "It prevents identical passwords from producing identical stored hashes.", "It must be unique enough to defeat precomputed tables."),
    ("input validation", "Input validation checks that data has the expected shape and constraints.", "It prevents bad data from reaching assumptions deeper in the system.", "Validation should happen at trust boundaries."),
    ("SQL injection", "SQL injection happens when untrusted input becomes part of a query as code.", "Parameterized queries prevent user text from changing query structure.", "Escaping alone is easy to get wrong."),
    ("XSS", "Cross-site scripting lets attacker-controlled script run in a user's browser.", "It can steal data or act as the user.", "Output escaping and content security policies help reduce the risk."),
    ("CSRF", "Cross-site request forgery tricks a browser into sending an authenticated request.", "It abuses existing cookies or sessions.", "CSRF tokens and same-site cookies help defend against it."),
    ("unit test", "A unit test checks a small piece of behavior in isolation.", "It makes failures easier to locate.", "Excessive mocking can hide integration problems."),
    ("integration test", "An integration test checks whether several components work together.", "It catches wiring and contract issues.", "It is often slower and harder to debug than a unit test."),
    ("flaky test", "A flaky test passes or fails without a meaningful code change.", "It wastes attention and weakens trust in the test suite.", "Flakiness often comes from timing, shared state, or external dependencies."),
    ("test fixture", "A test fixture provides known setup data for a test.", "It makes tests repeatable and easier to read.", "Large fixtures can obscure what a test actually needs."),
    ("mock", "A mock replaces a real dependency with controlled behavior in a test.", "It isolates the unit under test.", "Too many mocks can make tests mirror implementation details."),
    ("contract test", "A contract test checks that two systems agree on an interface.", "It catches breaking changes before deployment.", "It still needs representative examples."),
    ("code review", "Code review is a human check for correctness, clarity, and maintainability.", "It catches issues tests may miss and spreads context.", "It works best when comments focus on behavior and tradeoffs."),
    ("technical debt", "Technical debt is the future cost created by a shortcut or outdated design.", "It can be worth taking deliberately to move faster.", "Untracked debt compounds and slows future work."),
    ("refactor", "A refactor changes code structure without changing intended behavior.", "It can make future changes safer and easier.", "Tests are important because behavior should stay the same."),
    ("API", "An API is an interface that lets software components communicate.", "It defines what callers can ask for and what responses to expect.", "A confusing API spreads complexity to every caller."),
    ("semantic versioning", "Semantic versioning uses major, minor, and patch numbers to communicate compatibility.", "It helps users judge upgrade risk.", "It only works if maintainers follow the compatibility promise."),
    ("dependency pinning", "Dependency pinning fixes package versions.", "It makes builds more reproducible.", "Pinned versions still need updates for bug and security fixes."),
    ("environment variable", "An environment variable is a named value provided by the runtime environment.", "It lets configuration change without editing code.", "Secrets in environment variables still need careful handling."),
    ("command-line flag", "A command-line flag changes how a program runs.", "It makes behavior configurable for scripts and operators.", "Too many flags can make behavior hard to understand."),
    ("JSON", "JSON is a text format for structured data with objects, arrays, strings, numbers, booleans, and null.", "It is easy for both humans and programs to read.", "It does not support comments in the standard format."),
    ("YAML", "YAML is a human-friendly structured data format.", "It is common for configuration files.", "Whitespace and implicit typing can create surprises."),
    ("CSV", "CSV stores tabular data as comma-separated text.", "It is simple and widely supported.", "Escaping commas and newlines can be awkward."),
    ("Markdown", "Markdown is a lightweight format for writing structured text.", "It is useful for README files, notes, and documentation.", "Different renderers support slightly different extensions."),
    ("recursion", "Recursion solves a problem by having a function call itself on smaller inputs.", "It is useful for tree-like structures and divide-and-conquer problems.", "It needs a base case or it will not stop."),
    ("dynamic programming", "Dynamic programming saves solutions to repeated subproblems.", "It can turn exponential recomputation into a much cheaper process.", "It requires identifying the right state representation."),
    ("Big O", "Big O describes how an algorithm's cost grows with input size.", "It helps compare scalability independent of machine speed.", "Constants still matter for small inputs."),
    ("binary search", "Binary search repeatedly halves a sorted search space.", "It is much faster than scanning for large sorted lists.", "It only works when the data is ordered and random access is cheap."),
    ("hash table", "A hash table maps keys to values using a hash function.", "It gives fast average lookup and insertion.", "Bad hashing or too many collisions can hurt performance."),
    ("stack", "A stack is a last-in-first-out collection.", "It is useful for undo history, parsing, and depth-first traversal.", "It is different from a queue, which serves oldest items first."),
    ("queue data structure", "A queue is a first-in-first-out collection.", "It is useful when work should be handled in arrival order.", "Priority queues intentionally break plain arrival order."),
    ("graph", "A graph represents nodes connected by edges.", "It models networks, dependencies, routes, and relationships.", "Large graphs can require specialized algorithms."),
    ("tree", "A tree is a graph with a hierarchy and no cycles.", "It is useful for file systems, parse structures, and indexes.", "Unbalanced trees can become inefficient."),
    ("binary tree", "A binary tree is a tree where each node has at most two children.", "It is a common structure for search and expression parsing.", "Performance depends on how balanced it is."),
    ("sorting", "Sorting puts items into an ordered sequence.", "It makes searching, grouping, and presentation easier.", "The best algorithm depends on size, stability, and data shape."),
    ("stable sort", "A stable sort keeps equal items in their original relative order.", "It matters when sorting by multiple keys in stages.", "Not every sorting algorithm is stable by default."),
]


CHECKLIST_ROWS = [
    ("debug a flaky test", "Reproduce it repeatedly, remove timing assumptions, isolate shared state, inspect logs, and replace sleeps with explicit waits."),
    ("review a pull request", "Check correctness, edge cases, tests, naming, failure behavior, and whether the change fits the surrounding design."),
    ("prepare a rollback", "Confirm the target version, database compatibility, config changes, monitoring checks, and the exact rollback command."),
    ("investigate high CPU", "Find the busiest process, inspect hot threads, compare recent deploys, and check whether traffic or retries changed."),
    ("investigate high memory", "Check process growth, heap or object counts, recent changes, cache sizes, and whether memory returns after load drops."),
    ("debug a 500 error", "Find the request ID, inspect server logs, identify the failing dependency, reproduce with the same input, and add a regression test."),
    ("debug a slow query", "Look at the query plan, filters, indexes, row counts, locks, and whether the query shape changed recently."),
    ("clean up a README", "Put the quick start first, remove stale commands, group related sections, add examples, and keep troubleshooting concise."),
    ("write release notes", "Summarize user-visible changes, mention migrations or risks, list fixes, and include any action required from users."),
    ("triage an incident", "Stabilize impact, assign roles, gather facts, form hypotheses, communicate status, and record follow-up work."),
    ("plan a migration", "Inventory dependencies, design a reversible path, test with production-like data, monitor rollout, and keep rollback ready."),
    ("prepare a design review", "State the problem, constraints, alternatives, recommended approach, risks, and the questions needing feedback."),
    ("debug CI failure", "Compare local and CI environments, dependency versions, file paths, permissions, timing, and cached artifacts."),
    ("reduce log noise", "Identify repeated low-value messages, lower their level, add sampling, and preserve logs needed for diagnosis."),
    ("write an experiment plan", "Define the hypothesis, baseline, metric, change, duration, guardrails, and decision rule before running it."),
    ("evaluate a model checkpoint", "Compare validation metrics, inspect qualitative failures, test regressions, and verify the target behavior improved."),
    ("prepare for an on-call handoff", "Share current status, active alerts, recent changes, known risks, dashboards, and next actions."),
    ("make a meeting useful", "Write the goal, share context beforehand, keep decisions visible, and end with owners for next steps."),
    ("estimate a task", "Break it into parts, identify unknowns, compare similar work, include review time, and state assumptions."),
    ("choose between two libraries", "Compare maintenance, API fit, performance, license, ecosystem, migration cost, and how easily you can exit."),
    ("write a bug report", "Include expected behavior, actual behavior, reproduction steps, environment, logs, and suspected impact."),
    ("write a support reply", "Acknowledge the issue, restate the useful facts, give the next step, and set expectations for follow-up."),
    ("prepare a database migration", "Check schema compatibility, backfill strategy, dual-write needs, rollback safety, and monitoring."),
    ("test a retry policy", "Simulate transient failures, permanent failures, timeouts, rate limits, and dependency recovery."),
    ("review a security-sensitive change", "Check authentication, authorization, input validation, logging of secrets, auditability, and failure defaults."),
    ("improve an API error", "Name the failed operation, include the relevant identifier, explain the cause if known, and suggest a next step."),
    ("debug a deployment", "Compare artifact versions, environment variables, secrets, health checks, logs, and recent infrastructure changes."),
    ("organize a messy task list", "Separate urgent from important work, remove duplicates, group related items, and choose the next concrete action."),
    ("write a short project update", "State what changed, what is blocked, what comes next, and where help is needed."),
    ("prepare a demo", "Pick the story, reset data, test the happy path, prepare fallback screenshots, and keep the script short."),
    ("debug a cache issue", "Check cache keys, TTLs, invalidation paths, serialization, and whether stale data is being served."),
    ("debug a permissions issue", "Confirm identity, roles, resource ownership, policy evaluation order, and the exact denied action."),
    ("write a test plan", "Cover the happy path, edge cases, failure modes, data boundaries, and rollback or cleanup behavior."),
    ("improve a dashboard", "Remove noisy panels, highlight user impact, add relevant percentiles, and link to runbooks."),
    ("debug a memory leak", "Track memory over time, capture heap snapshots, inspect retained references, and test whether cleanup releases memory."),
    ("review observability for a service", "Check logs, metrics, traces, dashboards, alerts, and whether they answer common failure questions."),
    ("prepare a production change", "Confirm the rollout plan, owners, monitoring, rollback, support impact, and communication channel."),
    ("investigate a queue backlog", "Check producer rate, consumer rate, failures, retries, slow dependencies, and queue age."),
    ("write a concise commit message", "Use an imperative summary that names the behavior changed, then add details only if they clarify risk."),
    ("debug a tokenization issue", "Inspect raw text, token boundaries, special tokens, normalization, and whether train and inference formatting match."),
]


WRITING_ROWS = [
    ("Make this more concise: I am writing to let you know that the deployment has completed successfully.", "The deployment completed successfully."),
    ("Make this less abrupt: send me the logs now.", "Could you send me the logs when you have a moment?"),
    ("Make this clearer: the thing fails when the data thing is empty.", "The request fails when the dataset is empty."),
    ("Make this more collaborative: fix this before merging.", "Could we address this before merging? I think it will prevent a follow-up issue."),
    ("Write a calm one-sentence reply saying I will check the logs.", "I will check the logs and follow up with the most relevant findings."),
    ("Write a friendly two-sentence reply saying I need another hour.", "I am still working through the last details and need about another hour. I will send an update as soon as I have something solid."),
    ("Write a short note saying the deploy is paused.", "The deploy is paused while we investigate the issue and confirm the safest next step."),
    ("Write a polite note asking for reproduction steps.", "Could you share the exact steps you used to reproduce this? That will help me narrow it down quickly."),
    ("Write a brief apology for missing a notification.", "Sorry I missed the notification earlier. I am catching up now and will follow through."),
    ("Write a concise code review comment about unclear naming.", "Could we rename this to make the responsibility clearer at the call site?"),
    ("Write a neutral sentence for disagreeing respectfully.", "I see the reasoning, but I think there is another tradeoff we should consider."),
    ("Write a short risk statement for a migration.", "The main risk is that the migration touches production data, so we should test rollback before release."),
    ("Rewrite this as a status update: still debugging but close.", "I am still debugging the issue, but I have narrowed it down and expect to have a clearer update soon."),
    ("Rewrite this to sound more confident: I think maybe the cache is broken.", "The evidence points to the cache path, so I am checking keys, invalidation, and stale reads next."),
    ("Write a one-line reminder to review error handling.", "Reminder: review the error handling path before merging."),
    ("Write a concise message asking for a second reviewer.", "Could someone else take a quick second look at this before we merge?"),
    ("Make this more specific: it is slow sometimes.", "The endpoint is slow during peak traffic, especially when requests include large filters."),
    ("Make this less defensive: that was not my fault.", "I do not think this came from my change, but I will help trace where it started."),
    ("Write a warm reply to someone who is blocked.", "I can pair with you on it and help narrow down the blocker."),
    ("Write a short handoff note for an unresolved bug.", "The bug is still open; I reproduced it with the staging config and the next step is checking the request parser."),
]


EVERYDAY_ROWS = [
    ("How do I make oatmeal creamier?", "Use a little more liquid, cook it slowly, stir often, and finish with milk or yogurt after the oats soften."),
    ("How can I make coffee less bitter?", "Use slightly cooler water, grind coarser, shorten the brew time, or choose a lighter roast."),
    ("How do I keep rice from getting mushy?", "Rinse it, use the right water ratio, keep the lid closed, and let it rest after cooking."),
    ("Give me a simple recipe for scrambled eggs.", "Whisk eggs with a pinch of salt, cook slowly in a buttered pan, and stir gently until just set."),
    ("Give me a quick plan for tidying a desk.", "Remove trash, group loose items, wipe the surface, keep daily tools nearby, and put the rest away."),
    ("What is one tip for staying focused during a long task?", "Pick a small next action and set a short timer so the work feels easier to start."),
    ("How can I make a grocery list faster?", "Plan around meals, check staples first, group items by store section, and keep the list visible while shopping."),
    ("How do I pack lightly for a short trip?", "Choose versatile clothes, repeat shoes, pack small toiletries, and leave room for only what you know you will use."),
    ("What is a good way to start cleaning a room?", "Start with trash and laundry because they create visible progress quickly."),
    ("How can I remember to drink more water?", "Keep water where you already spend time and pair drinking with habits like meals or breaks."),
    ("Give me a simple stretching reminder.", "Take a minute to stand up, roll your shoulders, and stretch your neck gently."),
    ("How do I make a small room feel less cluttered?", "Clear surfaces, use closed storage, keep a consistent color palette, and remove items you rarely use."),
    ("What is a practical way to save a little money each month?", "Automate a small transfer after payday and treat it like a regular bill."),
    ("How can I make mornings less rushed?", "Prepare clothes, keys, and breakfast items the night before, then keep the first task simple."),
    ("Give me a quick way to plan tomorrow.", "Write the top three tasks, choose the first action for each, and leave space for interruptions."),
    ("How do I make a habit easier to keep?", "Attach it to an existing routine and make the first version small enough that skipping feels unnecessary."),
    ("What is a simple way to handle a crowded inbox?", "Archive obvious noise, answer quick items, and turn larger messages into tasks with owners and dates."),
    ("How can I improve a short walk?", "Leave your phone in your pocket, notice landmarks, and choose a route with a little sunlight or greenery."),
    ("Give me a tiny checklist for hosting dinner.", "Confirm the time, plan one simple main dish, check dietary needs, clear the table, and prep drinks early."),
    ("How do I cool soup safely before storing it?", "Divide it into shallow containers, let steam escape briefly, and refrigerate it once it is no longer very hot."),
]


SCIENCE_ROWS = [
    ("Explain photosynthesis in one sentence.", "Photosynthesis is how plants use sunlight, water, and carbon dioxide to make sugar and release oxygen."),
    ("What is evaporation?", "Evaporation is when liquid molecules gain enough energy to become gas."),
    ("Explain gravity simply.", "Gravity is the attraction between objects with mass, and it pulls things toward each other."),
    ("What is a hypothesis?", "A hypothesis is a testable explanation for something you observe."),
    ("What is the difference between weather and climate?", "Weather describes short-term conditions, while climate describes long-term patterns in a region."),
    ("Why does sleep matter for learning?", "Sleep helps the brain consolidate memories and recover attention for the next day."),
    ("What is an ecosystem?", "An ecosystem is a community of organisms interacting with each other and their physical environment."),
    ("Explain inertia simply.", "Inertia is the tendency of an object to keep doing what it is already doing unless a force changes it."),
    ("What is a molecule?", "A molecule is a group of atoms bonded together."),
    ("What is a cell in biology?", "A cell is the basic living unit that makes up organisms."),
    ("Explain natural selection briefly.", "Natural selection is the process where traits that help survival or reproduction become more common over generations."),
    ("What is a fossil?", "A fossil is preserved evidence of ancient life, such as bones, shells, footprints, or impressions."),
    ("Why does metal feel colder than wood at the same temperature?", "Metal conducts heat away from your skin faster, so it feels colder even when both objects are the same temperature."),
    ("What is pH?", "pH is a scale that describes how acidic or basic a solution is."),
    ("Why does salt melt ice?", "Salt lowers water's freezing point, so ice can melt at temperatures where pure water would stay frozen."),
    ("What is an orbit?", "An orbit is the curved path one object follows around another because of gravity and motion."),
    ("What is a black hole?", "A black hole is a region where gravity is so strong that nothing, not even light, can escape past its boundary."),
    ("Explain plate tectonics simply.", "Plate tectonics describes how Earth's outer shell is divided into moving plates that shape continents, earthquakes, and volcanoes."),
    ("What is a vaccine?", "A vaccine trains the immune system to recognize a pathogen without requiring the full disease."),
    ("What is herd immunity?", "Herd immunity happens when enough people are immune that a disease has trouble spreading through a community."),
]


MATH_ROWS = [
    ("What is 17 plus 28?", "17 plus 28 is 45."),
    ("What is 9 times 7?", "9 times 7 is 63."),
    ("What is 144 divided by 12?", "144 divided by 12 is 12."),
    ("What is 15 percent of 80?", "15 percent of 80 is 12."),
    ("If a book costs 12 dollars and is 25 percent off, what is the sale price?", "The discount is 3 dollars, so the sale price is 9 dollars."),
    ("If I walk 3 miles each day for 5 days, how far do I walk?", "You walk 15 miles."),
    ("What is the average of 4, 8, and 12?", "The average is 8."),
    ("What is the perimeter of a rectangle with sides 4 and 9?", "The perimeter is 26 because 2 times 4 plus 2 times 9 equals 26."),
    ("What is the area of a rectangle that is 6 by 7?", "The area is 42 square units."),
    ("What is 2 to the power of 5?", "2 to the power of 5 is 32."),
    ("What is one third of 60?", "One third of 60 is 20."),
    ("If something doubles from 14, what is the new value?", "The new value is 28."),
    ("What is 3.5 plus 2.25?", "3.5 plus 2.25 is 5.75."),
    ("What is 100 minus 37?", "100 minus 37 is 63."),
    ("What is 11 squared?", "11 squared is 121."),
    ("If a train travels 60 miles in 2 hours, what is its average speed?", "Its average speed is 30 miles per hour."),
    ("What is the next number in 2, 4, 8, 16?", "The next number is 32."),
    ("What is 7 percent as a decimal?", "7 percent as a decimal is 0.07."),
    ("What is the median of 3, 9, and 12?", "The median is 9."),
    ("If I have 18 items and split them into 3 equal groups, how many are in each group?", "Each group has 6 items."),
]


def concept_variants():
    rows = []
    for name, plain, practical, caveat in CONCEPT_ROWS:
        rows.extend(
            [
                (f"Explain {name} in plain language.", plain),
                (f"What does {name} mean in practice?", plain),
                (f"Why does {name} matter?", practical),
                (f"Give one caveat about {name}.", caveat),
                (f"Summarize {name} in one sentence.", plain),
            ]
        )
    return rows


def checklist_variants():
    rows = []
    for task, answer in CHECKLIST_ROWS:
        rows.extend(
            [
                (f"Give me a practical checklist to {task}.", answer),
                (f"What should I check first when I need to {task}?", answer.split(",")[0] + "."),
                (f"Make a concise plan to {task}.", answer),
                (f"For a teammate, summarize how to {task}.", answer),
            ]
        )
    return rows


def writing_variants():
    rows = []
    for question, answer in WRITING_ROWS:
        rows.extend(
            [
                (question, answer),
                (f"Please help with wording: {question}", answer),
                (f"Give me the polished version. {question}", answer),
            ]
        )
    return rows


def everyday_variants():
    return list(EVERYDAY_ROWS)


def science_variants():
    return list(SCIENCE_ROWS)


def math_variants():
    return list(MATH_ROWS)


CURATED_NORMAL_CHAT_ROWS = dedupe(
    concept_variants()
    + checklist_variants()
    + writing_variants()
    + everyday_variants()
    + science_variants()
    + math_variants()
)


NORMAL_PREFIX_INTENTS = [
    "Answer directly",
    "Answer normally",
    "Give the practical answer",
    "Keep this concise",
    "Use plain language",
    "Give a useful answer",
    "Be brief and clear",
    "Explain without extra context",
    "Focus only on this request",
    "Treat this as standalone",
    "Do not use memory",
    "Use general knowledge only",
    "Give one or two sentences",
    "Make it easy to scan",
    "Avoid speculation",
    "Use a calm tone",
    "Give the short version",
    "Answer for a teammate",
    "Answer for a beginner",
    "Answer for a busy reader",
    "Give the operational version",
    "Give the safest general answer",
    "Skip personal details",
    "Do not infer anything about me",
    "Answer as a standalone helper",
    "Use only the visible request",
    "Keep unrelated context out",
    "Give the current-turn answer",
    "Answer without recalling anything",
    "Keep the response grounded in this prompt",
]

NORMAL_PREFIX_CONTEXTS = [
    "{question}",
    "Current task: {question}",
    "Standalone question: {question}",
    "Fresh question: {question}",
    "No remembered fact is needed. {question}",
    "This is unrelated to stored facts. {question}",
    "For this turn only: {question}",
    "Ignore unrelated prior details. {question}",
    "Please answer the question itself: {question}",
    "No personal-memory lookup is required. {question}",
]

CURATED_NORMAL_PROMPT_PREFIXES = dedupe(
    [context for context in NORMAL_PREFIX_CONTEXTS]
    + [f"{intent}: {{question}}" for intent in NORMAL_PREFIX_INTENTS]
    + [f"{intent}. {context}" for intent in NORMAL_PREFIX_INTENTS for context in NORMAL_PREFIX_CONTEXTS]
)


MEMORY_IRRELEVANT_INTENTS = [
    "Ignore remembered names",
    "Do not mention stored pet names",
    "Do not use prior memory",
    "Answer from the current prompt only",
    "Treat any memory as irrelevant",
    "Do not invent a remembered fact",
    "Keep stored facts out of this",
    "Use no personal details",
    "Answer as a normal assistant",
    "Do not switch into recall mode",
    "No memory lookup is needed",
    "Do not let stored values change the answer",
    "Ignore any remembered pets or projects",
    "Do not answer from a stored profile",
    "Use only the text in this message",
    "Answer without recalling previous sessions",
    "This should remain ordinary chat",
    "Do not attach a remembered name",
    "Keep the answer independent of memory",
]

CURATED_MEMORY_IRRELEVANT_PREFIXES = dedupe(
    [f"{intent}: {{question}}" for intent in MEMORY_IRRELEVANT_INTENTS]
    + [f"{intent}. {context}" for intent in MEMORY_IRRELEVANT_INTENTS for context in NORMAL_PREFIX_CONTEXTS[:6]]
)
