"""
Prepare balanced curriculum shards for episodic-memory training.

The old K/V campaign is intentionally large, but it over-represents narrow
recall templates. This script writes smaller, balanced shards that separate:

- memory facts: facts that should be retrieved strongly
- bindings/currentness: current-vs-old and same-field confusers
- structured fields: repo/region/branch/tooling bundles and single-field asks
- language guardrails: ordinary chats where memory must not hijack generation

The output is meant to be mixed with, or used instead of, the legacy campaign
files. Rows use the same CustomJSON format as the rest of nanochat.
"""

import argparse
import json
import random
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.generate_kv_campaign_corpus import (
    ACKS,
    BRANCHES,
    CHANNELS,
    CITIES,
    CLUSTERS,
    CUSTOMER_NAMES,
    DATABASES,
    DATES,
    EDITORS,
    FORMATS,
    PET_NAMES,
    PLAN_NAMES,
    PYTHON_VERSIONS,
    REGIONS,
    REPOS,
    SHELLS,
    build_fact_group,
    person,
    pick,
    pick_other,
    sample_negatives,
)
from scripts.v4_normal_chat_corpus import (
    CURATED_MEMORY_IRRELEVANT_PREFIXES,
    CURATED_NORMAL_CHAT_ROWS,
    CURATED_NORMAL_PROMPT_PREFIXES,
)


DEFAULT_OUTPUT_DIR = ROOT_DIR / "data" / "kv_campaign"
CURRICULUM_EVAL_MODE = False


def dedupe(values):
    return list(dict.fromkeys(values))


def train_eval_pool(train_values, eval_values):
    return eval_values if CURRICULUM_EVAL_MODE else train_values


def pick_train_eval(rng, train_values, eval_values):
    return pick(rng, train_eval_pool(train_values, eval_values))


def compose_prompt_variants(openers, bodies, closers):
    rows = []
    for opener in openers:
        for body in bodies:
            for closer in closers:
                pieces = [opener.strip(), body.strip(), closer.strip()]
                rows.append(" ".join(piece for piece in pieces if piece))
    return dedupe(rows)


def generated_branch_pool():
    prefixes = ["feature", "fix", "bugfix", "hotfix", "release", "chore", "refactor", "experiment", "spike", "safe"]
    topics = [
        "auth", "billing", "search", "memory", "checkout", "profile", "observability",
        "api", "cache", "scheduler", "notifications", "import", "export", "dashboard",
        "rate-limit", "session", "terraform", "keycloak", "docs", "metrics", "rollout",
        "migration", "token-refresh", "tenant-sync", "audit", "webhook", "worker",
    ]
    suffixes = ["", "-v2", "-cleanup", "-retry", "-rollback", "-phase2", "-gpu", "-mps", "-hardening", "-guardrail"]
    releases = [f"release/2026-{month:02d}" for month in range(1, 13)]
    generated = [f"{prefix}/{topic}{suffix}" for prefix in prefixes for topic in topics for suffix in suffixes]
    return releases + generated


def generated_repo_pool():
    domains = [
        "billing", "identity", "search", "chat", "client", "analytics", "gitops",
        "auth", "cohort", "release", "feature", "tenant", "memory", "metrics",
        "notifications", "payments", "profile", "support", "admin", "audit",
    ]
    components = [
        "api", "console", "worker", "gateway", "router", "sync", "deployer",
        "service", "planner", "orchestrator", "flags", "importer", "registry",
        "dashboard", "scheduler", "webhook", "jobs", "adapter", "proxy", "agent",
    ]
    return [f"{domain}-{component}" for domain in domains for component in components]


def generated_customer_pool():
    left = [
        "Acorn", "Northstar", "Blue Mesa", "Lumen", "Cinder", "Harbor", "Juniper",
        "Atlas", "Cedar", "River", "Summit", "Beacon", "Silverline", "Prairie",
        "Orchard", "Nimbus", "Solstice", "Meridian", "Aurora", "Copper",
    ]
    right = [
        "Health", "Labs", "Retail", "Transit", "Schools", "Solar", "Foods",
        "Finance", "Studios", "Systems", "Robotics", "Logistics", "Energy",
        "Cloud", "Works", "Media", "Bio", "Learning", "Travel", "Homes",
    ]
    return [f"{a} {b}" for a in left for b in right]


def generated_database_pool():
    engines = [
        "Postgres", "PostgreSQL", "SQLite", "Redis", "MySQL", "MariaDB",
        "MongoDB", "DynamoDB", "BigQuery", "Snowflake", "ClickHouse",
        "Elasticsearch", "OpenSearch", "Cassandra", "Neo4j", "DuckDB",
        "Spanner", "CockroachDB", "Firestore", "Supabase",
    ]
    roles = [
        "primary", "replica", "analytics", "events", "sessions", "billing",
        "profiles", "audit", "warehouse", "cache", "search", "ledger",
        "metrics", "logs", "imports", "exports", "queue", "inventory",
    ]
    envs = ["dev", "stage", "staging", "prod", "prod-use1", "prod-euw1", "canary", "dr"]
    generated = []
    for role in roles:
        for engine in engines:
            clean = engine.lower().replace("postgresql", "postgres").replace(" ", "-")
            generated.append(f"{role}-{clean}")
            for env in envs:
                generated.append(f"{env}-{role}-{clean}")
    return engines + generated


def generated_plan_pool():
    tiers = [
        "Free", "Starter", "Basic", "Standard", "Plus", "Pro", "Team",
        "Business", "Enterprise", "Growth", "Scale", "Premium", "Ultimate",
        "Trial", "Developer", "Community", "Advanced", "Agency", "Education",
        "Premier",
    ]
    audiences = [
        "Personal", "Startup", "Team", "Business", "Enterprise", "Agency",
        "Education", "Nonprofit", "Developer", "Creator", "Pro", "Scale",
        "Global", "Regional", "Partner", "Sandbox",
    ]
    cadences = ["Monthly", "Annual", "Trial", "Pilot", "Legacy", "2026", "Plus", "Priority"]
    generated = []
    for audience in audiences:
        for tier in tiers:
            generated.append(f"{audience} {tier}")
            for cadence in cadences:
                generated.append(f"{audience} {tier} {cadence}")
    return tiers + generated


# V4 uses this file directly, but several builders still consume constants
# imported from the older K/V generator. Expand them here so V4 does not inherit
# tiny closed-list domains such as 4 plans or 6 branches.
CITIES = dedupe(
    CITIES
    + [
        "Amsterdam", "Athens", "Barcelona", "Berlin", "Boston", "Brussels",
        "Buenos Aires", "Chicago", "Copenhagen", "Dublin", "Edinburgh",
        "Helsinki", "Hong Kong", "Istanbul", "Jakarta", "Madrid", "Manila",
        "Melbourne", "Mexico City", "Milan", "Montreal", "Munich", "Nairobi",
        "New York", "Paris", "Prague", "Reykjavik", "Rome", "Santiago",
        "Singapore", "Stockholm", "Sydney", "Tokyo", "Toronto", "Warsaw",
        "Zurich",
    ]
)
REPOS = dedupe(REPOS + generated_repo_pool())
BRANCHES = dedupe(BRANCHES + generated_branch_pool())
CUSTOMER_NAMES = dedupe(CUSTOMER_NAMES + generated_customer_pool())
DATABASES = dedupe(DATABASES + generated_database_pool())
PLAN_NAMES = dedupe(PLAN_NAMES + generated_plan_pool())
CHANNELS = dedupe(CHANNELS + [f"#{topic}-{suffix}" for topic in ["auth", "billing", "memory", "rollout", "search", "client", "ops", "data", "incident", "platform"] for suffix in ["room", "triage", "watch", "support", "alerts", "review"]])
REGIONS = dedupe(REGIONS + [f"{env}-{geo}{num}" for env in ["dev", "qa", "staging", "prod1", "prod2", "lab"] for geo in ["use", "usw", "euw", "euc", "aps", "sae"] for num in [1, 2]])
CLUSTERS = dedupe(CLUSTERS + [f"{left}-{right}" for left in ["atlas", "cedar", "ember", "fjord", "harbor", "ion", "lumen", "nimbus", "orion", "summit"] for right in ["blue", "green", "red", "gold", "silver", "east", "west", "core"]])
DATES = dedupe(DATES + [f"{month} {day}" for month in ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"] for day in [3, 7, 12, 18, 21, 27]])
SHELLS = dedupe(SHELLS + ["nushell", "xonsh", "PowerShell", "dash"])
EDITORS = dedupe(EDITORS + ["Zed", "Cursor", "JetBrains IDE", "Vim", "TextMate", "Kate"])
FORMATS = dedupe(FORMATS + ["one concise paragraph", "numbered steps", "JSON when useful", "a short checklist", "plain prose", "two short bullets"])
PYTHON_VERSIONS = dedupe(PYTHON_VERSIONS + ["3.9", "3.13", "3.14"])


IDENTITY_FACT_PROMPTS = [
    "Please remember this for later: my {kind}'s name is {name}.",
    "For future chats, store this: my {kind} is named {name}.",
    "Keep this in memory: the name of my {kind} is {name}.",
    "Long-term note: my {kind}'s name is {name}.",
]

IDENTITY_QUESTIONS = [
    "What's my {kind}'s name?",
    "Tell me my {kind}'s name.",
    "Which {kind} name did I tell you?",
    "What name should you remember for my {kind}?",
    "Remind me of my {kind}'s name.",
]

IDENTITY_ANSWERS = [
    "{name}.",
    "It's {name}.",
    "Your {kind}'s name is {name}.",
    "You told me your {kind}'s name is {name}.",
    "The {kind} name you gave me is {name}.",
]

IDENTITY_CLEAN_CONTINUATIONS = [
    "Your {kind}'s name is {name}. I'll use that name when you ask.",
    "You told me {name}. I will keep the answer short.",
    "The answer is {name}.",
]

CURRENT_QUESTIONS = [
    "What is the current {field}?",
    "After the correction, what should you remember as the current {field}?",
    "Which {field} is current now?",
]

OLD_QUESTIONS = [
    "What was the old {field} before the correction?",
    "Which {field} did I replace?",
    "What value is no longer current for {field}?",
]

CURRENT_OLD_QUESTIONS = [
    "What is current now, and what was the old value?",
    "Tell me the current {field} and the old {field}.",
    "After the update, what changed?",
]

PROJECT_FIELD_QUESTIONS = [
    ("repo", "Which repo should we use now?"),
    ("region", "Which region is current?"),
    ("branch", "What branch should we use?"),
    ("python", "Which Python version did I ask you to remember?"),
    ("cluster", "Which cluster goes with this setup?"),
]

STRUCTURED_ALL_QUESTIONS = [
    "What repo, region, branch, cluster, and Python version should we use?",
    "Summarize the project setup I asked you to remember.",
    "Give me the remembered project config in one sentence.",
]

EXTRA_PET_NAMES = [
    "Fig", "Figaro", "Poppy-Blue", "Peppercorn", "Pebble", "Pebble-Brook",
    "Cloverleaf", "Miso-Soup", "Biscotti", "Juniper-Bell",
]

NAME_PREFIXES = [
    "al", "ar", "ash", "bel", "bex", "brin", "cal", "ced", "cor", "dax",
    "dun", "el", "fen", "fiz", "gal", "har", "iv", "jor", "kal", "kir",
    "lem", "lor", "mar", "mer", "nel", "nor", "or", "pel", "quin", "rin",
    "saf", "sel", "tal", "tor", "ul", "val", "wen", "yor", "zen", "ziv",
]

NAME_MIDDLES = [
    "a", "e", "i", "o", "u", "ai", "ea", "io", "oo", "ou",
    "ba", "bo", "ca", "di", "fa", "ki", "la", "lo", "mi", "na",
    "no", "pa", "ra", "ri", "sa", "si", "ta", "ti", "va", "vi",
]

NAME_SUFFIXES = [
    "bee", "bell", "berry", "bloom", "brook", "cloud", "cove", "dawn",
    "drift", "fern", "field", "fizz", "frost", "glow", "grove", "harbor",
    "hush", "leaf", "light", "loop", "marsh", "mist", "moon", "moss",
    "nook", "patch", "pearl", "puff", "reed", "ridge", "river", "root",
    "sage", "seed", "shine", "skye", "spark", "sprout", "stone", "tail",
    "thorn", "vale", "whisk", "willow", "wing", "wisp",
]


def title_piece(piece):
    return piece[:1].upper() + piece[1:]


def make_generated_pet_names(limit=80000):
    names = []
    seen = set()

    def add(name):
        if not name or name in seen:
            return
        seen.add(name)
        names.append(name)

    for prefix in NAME_PREFIXES:
        for middle in NAME_MIDDLES:
            for suffix in NAME_SUFFIXES:
                add(title_piece(prefix + middle + suffix))
                if len(names) >= limit:
                    return names

    base = names[:2000]
    for left in base[:600]:
        for right in base[600:900]:
            add(f"{left}-{right}")
            if len(names) >= limit:
                return names
    return names


GENERATED_PET_NAMES = make_generated_pet_names()
PET_NAME_POOL = list(dict.fromkeys(PET_NAMES + EXTRA_PET_NAMES + GENERATED_PET_NAMES))


SHORT_ODD_NAMES = [
    "Buh", "Qo", "Qob", "Vex", "Zog", "Zup", "Ixi", "Ubo", "Riq", "Nuv",
    "Kex", "Jip", "Taz", "Wug", "Mox", "Fup", "Lir", "Biv", "Dop", "Gah",
    "Pim", "Suv", "Yem", "Zix", "Av", "Oq", "Ib", "Ux", "Boq", "Tiv",
]

V4_ONSETS = [
    "b", "br", "c", "cr", "d", "dr", "f", "fl", "g", "gl", "h", "j",
    "k", "kl", "l", "m", "n", "p", "pl", "q", "qu", "r", "s", "sk",
    "t", "tr", "v", "w", "x", "y", "z", "zh",
]
V4_VOWELS = ["a", "e", "i", "o", "u", "ae", "ai", "ea", "io", "oa", "oo", "ou", "ua"]
V4_CODAS = [
    "b", "ch", "d", "f", "g", "k", "l", "m", "n", "p", "r", "s", "sh",
    "t", "th", "v", "x", "z", "bix", "brook", "cloud", "dle", "dock",
    "fenn", "glen", "hush", "kin", "lark", "marsh", "nook", "paz",
    "quill", "rook", "sprig", "tock", "vane", "wisp", "yarn", "zle",
]
V4_CONNECTORS = ["", "", "", "l", "m", "n", "r", "v", "z"]


def make_v4_pseudoword_names(limit=260000):
    names = []
    seen = set()

    def add(name):
        if not name or name in seen:
            return
        seen.add(name)
        names.append(name)

    for name in SHORT_ODD_NAMES:
        add(name)

    for onset in V4_ONSETS:
        for vowel in V4_VOWELS:
            for coda in V4_CODAS:
                add(title_piece(onset + vowel + coda))
                if len(names) >= limit:
                    return names

    # Two-root compounds are where BPE often gets fuzzy; this is the bucket
    # that produced failures like Taloobrook -> Tavarraig.
    roots = names[: min(8000, len(names))]
    for left in roots:
        left_raw = left.lower()
        for onset in V4_ONSETS:
            for vowel in V4_VOWELS[:8]:
                for coda in V4_CODAS[:20]:
                    add(title_piece(left_raw + onset + vowel + coda))
                    if len(names) >= limit:
                        return names

    base = names[: min(4000, len(names))]
    for left in base[:1200]:
        for right in base[1200:1800]:
            add(f"{left}-{right}")
            if len(names) >= limit:
                return names
    return names


V4_PET_NAMES = make_v4_pseudoword_names()
V4_NAME_POOL = list(dict.fromkeys(SHORT_ODD_NAMES + PET_NAME_POOL + V4_PET_NAMES))
_TOKENIZER = None
_V4_NAME_BUCKETS = None


def get_curriculum_tokenizer():
    global _TOKENIZER
    if _TOKENIZER is None:
        try:
            from nanochat.tokenizer import get_tokenizer

            _TOKENIZER = get_tokenizer()
        except Exception:
            _TOKENIZER = False
    return _TOKENIZER if _TOKENIZER is not False else None


def name_token_len(name):
    tokenizer = get_curriculum_tokenizer()
    if tokenizer is None:
        # Cheap fallback: keep the buckets useful even in tokenizer-free tests.
        return max(1, min(8, (len(name.replace("-", "")) + 2) // 3))
    return max(1, len(tokenizer.encode(" " + name)))


def v4_name_buckets():
    global _V4_NAME_BUCKETS
    if _V4_NAME_BUCKETS is not None:
        return _V4_NAME_BUCKETS
    buckets = {
        "short_odd": list(SHORT_ODD_NAMES),
        "one_token": [],
        "two_token": [],
        "three_token": [],
        "long_token": [],
        "hyphen": [],
        "all": V4_NAME_POOL,
    }
    for name in V4_NAME_POOL:
        token_len = name_token_len(name)
        if "-" in name:
            buckets["hyphen"].append(name)
        if token_len <= 1:
            buckets["one_token"].append(name)
        elif token_len == 2:
            buckets["two_token"].append(name)
        elif token_len == 3:
            buckets["three_token"].append(name)
        else:
            buckets["long_token"].append(name)
    for key, values in list(buckets.items()):
        if not values:
            buckets[key] = V4_NAME_POOL
    _V4_NAME_BUCKETS = buckets
    return buckets


def pick_v4_name(rng, bucket=None):
    buckets = v4_name_buckets()
    if bucket is None:
        bucket = pick(
            rng,
            [
                "short_odd",
                "short_odd",
                "one_token",
                "two_token",
                "two_token",
                "three_token",
                "long_token",
                "hyphen",
            ],
        )
    return pick(rng, buckets.get(bucket, buckets["all"]))


def sample_v4_negatives(rng, current, k=10):
    buckets = v4_name_buckets()
    current_token_len = name_token_len(current)
    negatives = []
    seen = {current}
    pool = buckets["all"]
    attempts = 0
    while len(negatives) < k and attempts < k * 96:
        attempts += 1
        candidate = pick(rng, pool)
        if candidate in seen:
            continue
        if (
            candidate[:1] == current[:1]
            or abs(name_token_len(candidate) - current_token_len) <= 1
            or ("-" in candidate) == ("-" in current)
        ):
            seen.add(candidate)
            negatives.append(candidate)
    while len(negatives) < k:
        candidate = pick(rng, pool)
        if candidate in seen:
            continue
        seen.add(candidate)
        negatives.append(candidate)
    return negatives


def pick_other(rng, values, current):
    if values is PET_NAME_POOL or len(values) > 1000:
        for _ in range(32):
            candidate = rng.choice(values)
            if candidate != current:
                return candidate
        return next(value for value in values if value != current)
    candidates = [value for value in values if value != current]
    return rng.choice(candidates)


def sample_negatives(rng, values, current, k=4):
    if values is PET_NAME_POOL or len(values) > 1000:
        negatives = []
        seen = {current}
        attempts = 0
        while len(negatives) < min(k, len(values) - 1) and attempts < k * 64:
            attempts += 1
            candidate = rng.choice(values)
            if candidate in seen:
                continue
            seen.add(candidate)
            negatives.append(candidate)
        if len(negatives) < min(k, len(values) - 1):
            for candidate in values:
                if candidate in seen:
                    continue
                negatives.append(candidate)
                if len(negatives) >= k:
                    break
        return negatives
    candidates = [value for value in values if value != current]
    rng.shuffle(candidates)
    return candidates[: min(k, len(candidates))]

HARD_NAME_PAIRS = [
    ("Pebble-Cloud", ["Pebble", "Pebble-Brook", "Poppy", "Poppy-Blue", "Pepper"]),
    ("Poppy-Blue", ["Poppy", "Pebble-Cloud", "Pepper", "Pixel"]),
    ("Pepper", ["Peppercorn", "Pebble-Cloud", "Poppy", "Pixel"]),
    ("Fig", ["Figaro", "Fable", "Fjord", "Fable"]),  # first-token branch pressure
    ("Clover", ["Cloverleaf", "Comet", "Cinder", "Miso"]),
    ("Miso", ["Miso-Soup", "Mochi", "Maple", "Clover"]),
    ("Biscuit", ["Biscotti", "Cinder", "Miso", "Juniper"]),
    ("Juniper", ["Juniper-Bell", "Juno", "Pepper", "Clover"]),
]

ANSWER_REALIZATION_QUESTIONS = [
    "Answer in a complete sentence: what is my {kind}'s name?",
    "Just answer naturally. What is my {kind}'s name?",
    "Give a short friendly answer: what name did I give for my {kind}?",
    "Use the name in a sentence about a quick errand.",
    "Write a one-line reminder that includes my {kind}'s name.",
    "Answer with only the remembered name.",
]
ANSWER_REALIZATION_QUESTIONS = dedupe(
    ANSWER_REALIZATION_QUESTIONS
    + compose_prompt_variants(
        [
            "",
            "Answer like a normal assistant.",
            "Use the stored detail in a natural sentence.",
            "Keep it conversational.",
            "Make the reply sound like chat, not a database lookup.",
            "Give a warm but short answer.",
            "Please phrase this naturally.",
            "Respond in one sentence.",
        ],
        [
            "What is my {kind}'s name?",
            "Which name should you use for my {kind}?",
            "What name did I give you for my {kind}?",
            "Remind me what my {kind} is called.",
            "Use my {kind}'s name in a small reminder.",
            "Write a tiny note that includes my {kind}'s name.",
            "Say the remembered name and nothing unrelated.",
            "Mention the name I asked you to remember for my {kind}.",
        ],
        [
            "",
            "No extra trivia.",
            "Do not turn it into a story.",
            "Keep the spelling intact.",
            "Use only the remembered value.",
        ],
    )
)

ANSWER_REALIZATION_ANSWERS = [
    "{name}.",
    "It's {name}.",
    "Your {kind}'s name is {name}.",
    "You told me your {kind}'s name is {name}.",
    "For the errand, remember to bring water for {name}.",
    "Reminder: your {kind}, {name}, is the one I should remember.",
]

LIVE_SESSION_FACT_PROMPTS = [
    "My {kind}'s name is {name}.",
    "For later: my {kind}'s name is {name}.",
    "Please remember this: my {kind}'s name is {name}.",
    "My {kind} is named {name}.",
    "The name of my {kind} is {name}.",
]

LIVE_SESSION_RECALL_QUESTIONS = [
    "What is my {kind}'s name?",
    "What's my {kind}'s name?",
    "Just answer naturally. What is my {kind}'s name?",
    "Answer naturally: what is my {kind}'s name?",
    "Tell me my {kind}'s name.",
    "Remind me of my {kind}'s name.",
    "Answer with only the remembered name: what is my {kind}'s name?",
]
LIVE_SESSION_RECALL_QUESTIONS_TRAIN = dedupe(
    LIVE_SESSION_RECALL_QUESTIONS
    + compose_prompt_variants(
        [
            "",
            "Quick check:",
            "For this chat,",
            "Please answer from the detail I gave you:",
            "Can you remind me:",
            "Use the remembered name:",
            "Answer naturally:",
            "Keep it short:",
            "In plain language,",
            "Just to verify,",
            "I forgot what I told you earlier:",
            "Based on what I said before,",
        ],
        [
            "what is my {kind}'s name?",
            "what's my {kind}'s name?",
            "which name did I give for my {kind}?",
            "what did I say my {kind} is called?",
            "what name should you use for my {kind}?",
            "what is the remembered name for my {kind}?",
            "tell me my {kind}'s name.",
            "remind me of my {kind}'s name.",
            "say the name I gave you for my {kind}.",
            "what name belongs to my {kind} in memory?",
            "what should you call my {kind}?",
            "which exact name did I store for my {kind}?",
        ],
        [
            "",
            "Please keep it brief.",
            "Only answer the question.",
            "Use the spelling I gave.",
            "A short answer is fine.",
            "No explanation needed.",
            "One sentence is enough.",
        ],
    )
)
LIVE_SESSION_RECALL_QUESTIONS_EVAL = dedupe(
    compose_prompt_variants(
        [
            "",
            "Let's test memory:",
            "Without re-reading the original sentence,",
            "From the earlier note,",
            "What did you retain:",
            "For a direct answer,",
            "I need the remembered value:",
            "Please retrieve this from memory:",
        ],
        [
            "what name did I assign to my {kind}?",
            "what is the name attached to my {kind}?",
            "which name should you remember for my {kind}?",
            "what did I call my {kind} earlier?",
            "what is my {kind} named?",
            "give me the remembered {kind} name.",
            "what name did I ask you to keep for my {kind}?",
            "what is the stored answer for my {kind}'s name?",
        ],
        [
            "",
            "Keep the answer compact.",
            "Use the exact letters.",
            "Do not add unrelated context.",
            "Answer as chat, not JSON.",
        ],
    )
)

LIVE_SESSION_RECALL_ANSWERS = [
    "{name}.",
    "It's {name}.",
    "Your {kind}'s name is {name}.",
    "You told me your {kind}'s name is {name}.",
]
LIVE_SESSION_RECALL_ANSWERS_TRAIN = dedupe(
    LIVE_SESSION_RECALL_ANSWERS
    + [
        "The name you gave me is {name}.",
        "I have {name} for your {kind}.",
        "You asked me to remember {name}.",
        "For your {kind}, I have {name}.",
        "The remembered name is {name}.",
        "You said your {kind} is called {name}.",
        "{name} is the name I have stored.",
        "I remember it as {name}.",
    ]
)
LIVE_SESSION_RECALL_ANSWERS_EVAL = dedupe(
    [
        "{name}.",
        "It is {name}.",
        "I have the name as {name}.",
        "The stored name is {name}.",
        "You gave me {name}.",
        "Your {kind} is named {name}.",
    ]
)

PAIR_FIELD_QUESTIONS = [
    (("repo", "branch"), "Which repo and branch should we use?"),
    (("region", "cluster"), "Which region and cluster go together?"),
    (("repo", "python"), "Which repo and Python version did I ask you to remember?"),
    (("branch", "cluster"), "What branch and cluster are current?"),
]

COMMON_DOG_DEFAULTS = [
    "Buddy", "Luna", "Max", "Poppy", "Charlie", "Cooper", "Bailey", "Daisy",
    "Milo", "Bella", "Rocky", "Lucy", "Molly", "Teddy", "Coco", "Ruby",
    "Rosie", "Sadie", "Tucker", "Bear", "Duke", "Scout", "Riley", "Finn",
    "Murphy", "Winston", "Blue", "Rex", "Penny", "Nala", "Hazel", "Gus",
]

COMMON_CAT_DEFAULTS = [
    "Milo", "Luna", "Mochi", "Maple", "Oliver", "Leo", "Simba", "Nala",
    "Bella", "Loki", "Cleo", "Willow", "Jasper", "Chloe", "Socks", "Oreo",
    "Pumpkin", "Misty", "Shadow", "Tiger", "Binx", "Pepper", "Mittens",
    "Nova", "Felix", "Mango", "Salem", "Ash", "Pearl", "Miso", "Clover",
]

COMMON_BRANCH_DEFAULTS = [
    "main", "master", "develop", "dev", "staging", "production", "prod",
    "release", "release/2026-04", "release/2026-05", "hotfix", "next",
    "trunk", "integration", "qa", "canary", "feature/login", "feature/auth",
    "bugfix/session-timeout", "fix/keycloak-redirect", "chore/deps",
]

COMMON_DATABASE_DEFAULTS = [
    "Postgres", "PostgreSQL", "SQLite", "Redis", "MySQL", "MariaDB",
    "MongoDB", "DynamoDB", "BigQuery", "Snowflake", "ClickHouse",
    "Elasticsearch", "OpenSearch", "Cassandra", "Neo4j", "DuckDB",
    "Spanner", "CockroachDB", "Firestore", "Supabase",
]

COMMON_PLAN_DEFAULTS = [
    "Free", "Starter", "Basic", "Standard", "Plus", "Pro", "Team",
    "Business", "Enterprise", "Growth", "Scale", "Premium", "Ultimate",
    "Trial", "Developer", "Community", "Advanced", "Agency", "Education",
]


def make_branch_default_pool():
    prefixes = ["feature", "fix", "bugfix", "hotfix", "release", "chore", "refactor", "experiment"]
    topics = [
        "auth", "billing", "search", "memory", "checkout", "profile", "observability",
        "api", "cache", "scheduler", "notifications", "import", "export", "dashboard",
        "rate-limit", "session", "terraform", "keycloak", "docs", "metrics",
    ]
    suffixes = ["", "-v2", "-cleanup", "-retry", "-rollback", "-phase2", "-gpu", "-mps"]
    generated = [f"{prefix}/{topic}{suffix}" for prefix in prefixes for topic in topics for suffix in suffixes]
    releases = [f"release/2026-{month:02d}" for month in range(1, 13)]
    return list(dict.fromkeys(COMMON_BRANCH_DEFAULTS + releases + generated))


def make_database_default_pool():
    # Keep heuristic defaults broad but bounded. V4 already expands DATABASES
    # above for remembered facts; using that full pool here creates a huge
    # low-signal Cartesian product of plausible-but-unhelpful defaults.
    engines = dedupe(
        COMMON_DATABASE_DEFAULTS
        + [
            "Postgres", "PostgreSQL", "SQLite", "Redis", "MySQL", "MariaDB",
            "MongoDB", "DynamoDB", "BigQuery", "Snowflake", "ClickHouse",
            "Elasticsearch", "OpenSearch", "Cassandra", "Neo4j", "DuckDB",
            "Spanner", "CockroachDB", "Firestore", "Supabase",
        ]
    )
    roles = [
        "primary", "replica", "analytics", "events", "sessions", "billing",
        "profiles", "audit", "warehouse", "cache", "search", "ledger",
        "metrics", "logs", "imports", "exports", "queue", "inventory",
    ]
    envs = ["dev", "stage", "staging", "prod", "prod-use1", "prod-euw1", "canary", "dr"]
    generated = []
    for role in roles:
        for engine in engines:
            clean = engine.lower().replace("postgresql", "postgres").replace(" ", "-")
            generated.append(f"{role}-{clean}")
            for env in envs[:4]:
                generated.append(f"{env}-{role}-{clean}")
    return dedupe(COMMON_DATABASE_DEFAULTS + engines + generated)


def make_plan_default_pool():
    # Same logic as databases: facts can draw from the large PLAN_NAMES pool,
    # but heuristic wrong-answer defaults should be compact and high quality.
    tiers = dedupe(
        COMMON_PLAN_DEFAULTS
        + [
            "Free", "Starter", "Basic", "Standard", "Plus", "Pro", "Team",
            "Business", "Enterprise", "Growth", "Scale", "Premium", "Ultimate",
            "Trial", "Developer", "Community", "Advanced", "Agency", "Education",
            "Premier",
        ]
    )
    audiences = [
        "Personal", "Startup", "Team", "Business", "Enterprise", "Agency",
        "Education", "Nonprofit", "Developer", "Creator", "Pro", "Scale",
        "Global", "Regional", "Partner", "Sandbox",
    ]
    cadences = ["Monthly", "Annual", "Trial", "Pilot", "Legacy", "2026", "Plus", "Priority"]
    generated = []
    for audience in audiences:
        for tier in tiers:
            generated.append(f"{audience} {tier}")
            for cadence in cadences[:5]:
                generated.append(f"{audience} {tier} {cadence}")
    return dedupe(COMMON_PLAN_DEFAULTS + tiers + generated)


def make_heuristic_defaults():
    # These are plausible/habitual answers the model might emit from prior
    # language habits. They should be broad enough that override training does
    # not accidentally become a tiny closed-list game.
    dog_generated = [name for name in PET_NAME_POOL if name not in COMMON_DOG_DEFAULTS][:512]
    cat_generated = [name for name in PET_NAME_POOL if name not in COMMON_CAT_DEFAULTS][512:1024]
    branch_generated = make_branch_default_pool()
    database_generated = make_database_default_pool()
    plan_generated = make_plan_default_pool()
    return {
        "dog": list(dict.fromkeys(COMMON_DOG_DEFAULTS + dog_generated)),
        "cat": list(dict.fromkeys(COMMON_CAT_DEFAULTS + cat_generated)),
        "branch": branch_generated,
        "database": database_generated,
        "plan": plan_generated,
    }


HEURISTIC_DEFAULTS = make_heuristic_defaults()

ORDINARY_CHAT_ROWS = [
    (
        "Can you explain why batching makes GPU training faster?",
        "Batching keeps the GPU busy by processing many examples in parallel, so the fixed overhead of each operation is shared across more tokens.",
    ),
    (
        "Give me a short checklist for debugging a flaky test.",
        "Reproduce it, isolate shared state, check timing assumptions, inspect recent changes, and add logging around the failing boundary.",
    ),
    (
        "How do I make oatmeal creamier?",
        "Use a little more liquid, cook it slowly, stir often, and finish with milk or yogurt after the oats soften.",
    ),
    (
        "What is a concise way to explain gradient clipping?",
        "Gradient clipping limits unusually large updates so training stays stable instead of jumping too far in one step.",
    ),
    (
        "Write a calm one-sentence reply saying I will check the logs.",
        "I will check the logs and follow up with the most relevant findings.",
    ),
    (
        "In Python, how do I sort a list of dictionaries by a key?",
        "Use sorted(rows, key=lambda row: row['key']) and replace 'key' with the field you want.",
    ),
    (
        "Give me a two-sentence explanation of why tests should be deterministic.",
        "Deterministic tests fail for the same reason every time, which makes debugging much faster. They also reduce false alarms from timing or random inputs.",
    ),
    (
        "What is the difference between latency and throughput?",
        "Latency is how long one request takes. Throughput is how many requests the system can handle over a period of time.",
    ),
    (
        "Write a compact git commit message for fixing an auth timeout.",
        "Fix auth timeout handling in retry path",
    ),
    (
        "How should I think about a learning-rate warmup?",
        "Warmup starts with smaller updates so the model can settle before the optimizer uses the full learning rate.",
    ),
    (
        "Give me a calm reply to a teammate who is blocked.",
        "I can take a look with you and help narrow down where the blocker is coming from.",
    ),
    (
        "What is one practical way to reduce overfitting?",
        "Use a validation set and stop or regularize training when validation quality stops improving.",
    ),
    (
        "Explain DNS in one sentence.",
        "DNS maps human-readable domain names to the network addresses computers use to reach services.",
    ),
    (
        "Make this sentence more concise: I am writing to let you know that the deployment has completed successfully.",
        "The deployment completed successfully.",
    ),
]

V4_NORMAL_CORE_ROWS = [
    ("Explain why a smaller learning rate can help after a model starts oscillating.", "A smaller learning rate makes each update less aggressive, so the model can refine a solution without repeatedly overshooting it."),
    ("Give me a practical debugging plan for a program that works locally but fails in CI.", "Compare environment variables, dependency versions, file paths, timing assumptions, and permissions, then reproduce CI as closely as possible."),
    ("Write a friendly two-sentence reply saying I need another hour.", "I am still working through the last details and need about another hour. I will send an update as soon as I have something solid."),
    ("What is the difference between precision and recall?", "Precision measures how many selected items were correct, while recall measures how many correct items were successfully selected."),
    ("Summarize the tradeoff between caching and freshness.", "Caching can make systems much faster, but cached data can become stale unless you invalidate or refresh it carefully."),
    ("Give me a tiny Python example that filters even numbers.", "Use a list comprehension: evens = [n for n in numbers if n % 2 == 0]."),
    ("Explain Docker image layers in one sentence.", "Docker image layers are reusable filesystem snapshots that stack together to form the final container image."),
    ("Write a concise note asking someone to review a pull request.", "Could you review this pull request when you have a moment? I would especially appreciate a look at the edge cases."),
    ("What is one reason a validation loss can rise while training loss falls?", "That pattern often means the model is overfitting the training examples instead of learning behavior that generalizes."),
    ("Give me a calm first step for investigating high CPU usage.", "Start by identifying the busiest process and thread, then check recent changes and logs around when CPU usage increased."),
    ("How should I explain an API rate limit to a non-engineer?", "A rate limit is a traffic rule that caps how many requests an app can make in a period of time so the service stays reliable."),
    ("Make this more polite: send me the logs now.", "Could you please send me the logs when you have a chance?"),
    ("Give me a short checklist for preparing a deployment rollback.", "Confirm the target version, database compatibility, config changes, monitoring checks, and the exact command or procedure to roll back."),
    ("Explain why batching can increase throughput but not always reduce latency.", "Batching does more work per operation, which improves throughput, but waiting to form a batch can add delay for an individual request."),
    ("Write one sentence describing a careful experiment.", "A careful experiment changes one variable at a time and measures the result against a clear baseline."),
    ("Give me a simple analogy for regularization.", "Regularization is like adding a gentle constraint that keeps a model from memorizing every wrinkle in the training data."),
    ("What is the purpose of a health check endpoint?", "A health check endpoint lets infrastructure decide whether a service instance is alive and ready to receive traffic."),
    ("Make a tiny plan for cleaning up a messy README.", "Start with the quick-start path, remove outdated commands, group related sections, and add examples for the most common tasks."),
    ("What does idempotent mean in an API?", "An idempotent API operation can be repeated without changing the result beyond the first successful call."),
    ("Answer briefly: why do we use test fixtures?", "Test fixtures provide known setup data so tests are repeatable and easier to understand."),
    ("Explain backpressure in distributed systems.", "Backpressure is a way for overloaded parts of a system to slow incoming work before queues grow without control."),
    ("What is eventual consistency?", "Eventual consistency means replicas may disagree briefly, but converge to the same value if no new updates arrive."),
    ("Give me a simple explanation of a race condition.", "A race condition happens when the result depends on timing between operations that should have been coordinated."),
    ("What is a deadlock?", "A deadlock happens when tasks wait on each other in a cycle, so none of them can make progress."),
    ("Explain why retries need timeouts.", "Timeouts prevent retries from waiting forever, which keeps failure handling bounded and easier to reason about."),
    ("What is the difference between authentication and authorization?", "Authentication verifies who someone is, while authorization decides what that person is allowed to do."),
    ("Explain a database index in one sentence.", "A database index is a lookup structure that speeds up reads by avoiding a full scan of every row."),
    ("What is the tradeoff of adding an index?", "An index can speed up reads, but it uses storage and can make writes slower because the index must also be updated."),
    ("Explain a queue in plain language.", "A queue stores work in order so producers and consumers can operate at different speeds."),
    ("What does a load balancer do?", "A load balancer spreads requests across healthy service instances so no single instance has to handle everything."),
    ("Explain what a reverse proxy is.", "A reverse proxy receives client requests and forwards them to backend services, often adding routing, TLS, or caching."),
    ("Why are logs useful during an incident?", "Logs give a timeline of what the system did, which helps connect symptoms to a likely cause."),
    ("Give me a one-sentence explanation of observability.", "Observability is the ability to understand a system's internal state from outputs like logs, metrics, and traces."),
    ("What is a metric cardinality problem?", "A cardinality problem happens when too many unique label combinations make metrics expensive or hard to query."),
    ("Explain a feature flag in one sentence.", "A feature flag lets you turn behavior on or off without deploying new code."),
    ("What is a canary deployment?", "A canary deployment sends a small share of traffic to a new version first so problems are caught before full rollout."),
    ("Explain blue-green deployment.", "Blue-green deployment keeps two environments so traffic can switch quickly from the old version to the new one."),
    ("What is a rollback plan?", "A rollback plan is the prepared path for returning to a known-good version if a release causes problems."),
    ("Explain dependency pinning.", "Dependency pinning fixes package versions so builds are more repeatable and less likely to change unexpectedly."),
    ("What is semantic versioning?", "Semantic versioning uses major, minor, and patch numbers to communicate compatibility expectations."),
    ("Explain a memory leak.", "A memory leak happens when a program keeps references to data it no longer needs, so memory usage grows over time."),
    ("What is a null pointer error?", "A null pointer error happens when code tries to use a missing object as if it were real."),
    ("Give me a concise explanation of recursion.", "Recursion is when a function solves a problem by calling itself on a smaller version of the same problem."),
    ("What is dynamic programming?", "Dynamic programming solves repeated subproblems once and reuses their results instead of recomputing them."),
    ("Explain Big O notation in one sentence.", "Big O describes how an algorithm's cost grows as the input gets larger."),
    ("What is the difference between a stack and a queue?", "A stack is last-in-first-out, while a queue is first-in-first-out."),
    ("Give me a tiny SQL query that counts rows by status.", "Use SELECT status, COUNT(*) FROM table_name GROUP BY status; and replace table_name with your table."),
    ("How do I safely rename a database column?", "Add the new column, backfill it, write to both columns, migrate readers, then remove the old column after verification."),
    ("Explain why migrations should be reversible when possible.", "Reversible migrations make it safer to recover if a deployment has to be rolled back."),
    ("What is a unit test?", "A unit test checks a small piece of code in isolation so failures are easier to locate."),
    ("What is an integration test?", "An integration test checks whether multiple parts of a system work together correctly."),
    ("Explain a flaky test in one sentence.", "A flaky test sometimes passes and sometimes fails without a meaningful code change."),
    ("How can I reduce flaky tests?", "Remove timing assumptions, isolate shared state, use deterministic inputs, and wait on explicit conditions instead of sleeps."),
    ("Give me a concise code review comment about unclear naming.", "Could we rename this to make the responsibility clearer at the call site?"),
    ("Write a polite note asking for reproduction steps.", "Could you share the exact steps you used to reproduce this? That will help me narrow it down quickly."),
    ("Write a short message saying the deploy is paused.", "The deploy is paused while we investigate the issue and confirm the safest next step."),
    ("Make this sentence clearer: the thing fails when the data thing is empty.", "The request fails when the dataset is empty."),
    ("Rewrite this to sound less abrupt: fix this before merging.", "Could we address this before merging? I think it will prevent a follow-up issue."),
    ("Write a brief apology for missing a notification.", "Sorry I missed the notification earlier. I am catching up now and will follow through."),
    ("Give me a concise standup update for debugging.", "I am investigating the failing path, have narrowed it to the request boundary, and will share findings once I verify the logs."),
    ("Write a short project-risk sentence.", "The main risk is that the migration touches production data, so we should test rollback before release."),
    ("Summarize why documentation should include examples.", "Examples show how the documented behavior is used in practice, which makes the instructions easier to follow."),
    ("Explain why shorter functions can be easier to maintain.", "Shorter functions usually have fewer responsibilities, so they are easier to read, test, and change safely."),
    ("What is the benefit of naming intermediate variables?", "Intermediate variables can make complex logic easier to read by giving important steps clear names."),
    ("Explain why comments should explain why, not just what.", "The code often shows what happens, but comments are useful when they explain intent, tradeoffs, or surprising constraints."),
    ("What is one way to make errors more actionable?", "Include the failed operation, the relevant identifier, and the next step someone can take."),
    ("Give me a simple recipe for scrambled eggs.", "Whisk eggs with a pinch of salt, cook slowly in a buttered pan, and stir gently until just set."),
    ("How do I keep rice from getting mushy?", "Rinse the rice, use the right water ratio, keep the lid closed, and let it rest after cooking."),
    ("Give me a quick plan for tidying a desk.", "Remove trash, group loose items, wipe the surface, keep only daily tools nearby, and put the rest away."),
    ("Write a two-item grocery list for making pancakes.", "Flour and eggs."),
    ("How can I make coffee less bitter?", "Use slightly cooler water, grind coarser, shorten the brew time, or choose a lighter roast."),
    ("Give me one tip for staying focused during a long task.", "Pick a small next action and set a short timer so the work feels easier to start."),
    ("Explain compound interest simply.", "Compound interest means you earn interest on both the original amount and the interest already added."),
    ("What is opportunity cost?", "Opportunity cost is the value of the best alternative you give up when choosing one option."),
    ("Explain confirmation bias.", "Confirmation bias is the tendency to notice or favor information that supports what you already believe."),
    ("What is a useful way to compare two options?", "List the goal, constraints, benefits, risks, and the cost of reversing each choice."),
    ("Give me a neutral sentence for disagreeing respectfully.", "I see the reasoning, but I think there is another tradeoff we should consider."),
    ("Explain photosynthesis in one sentence.", "Photosynthesis is how plants use sunlight, water, and carbon dioxide to make sugar and release oxygen."),
    ("What is evaporation?", "Evaporation is when liquid molecules gain enough energy to become gas."),
    ("Explain gravity simply.", "Gravity is the attraction between objects with mass, and it pulls things toward each other."),
    ("What is a hypothesis?", "A hypothesis is a testable explanation for something you observe."),
    ("Explain why sleep matters for learning.", "Sleep helps the brain consolidate memories and recover attention for the next day."),
    ("What is the difference between weather and climate?", "Weather describes short-term conditions, while climate describes long-term patterns in a region."),
    ("Give me a one-sentence summary of encryption.", "Encryption transforms readable data into a protected form that only someone with the right key can decode."),
    ("What is two-factor authentication?", "Two-factor authentication requires a second proof of identity in addition to a password."),
    ("Explain phishing in simple terms.", "Phishing is a trick where someone pretends to be trustworthy to steal information or access."),
    ("Give me a safe password tip.", "Use a password manager to create long, unique passwords for each account."),
    ("What does least privilege mean?", "Least privilege means giving users or systems only the access they need to do their job."),
    ("Explain data validation.", "Data validation checks that input has the expected shape and values before it is trusted."),
    ("What is a schema?", "A schema describes the expected structure of data, such as fields, types, and relationships."),
    ("What is JSON?", "JSON is a text format for representing structured data with objects, arrays, strings, numbers, booleans, and null."),
    ("Explain an environment variable.", "An environment variable is a named value provided to a process by its surrounding runtime environment."),
    ("What is a command-line flag?", "A command-line flag is an option passed to a program to change how it runs."),
    ("Give me a tiny Bash example that lists Python files.", "Use ls *.py to list Python files in the current directory."),
    ("How do I check disk usage in a directory?", "Use du -sh path to see a summary of disk usage for that directory."),
]

# Keep the ordinary-chat base pool broad because older builders sample
# ORDINARY_CHAT_ROWS directly, not just through the V4 expanded guardrails.
ORDINARY_CHAT_ROWS = dedupe(ORDINARY_CHAT_ROWS + V4_NORMAL_CORE_ROWS + CURATED_NORMAL_CHAT_ROWS)

V4_NORMAL_PROMPT_PREFIXES = [
    "{question}",
    "Answer briefly: {question}",
    "Answer normally and directly: {question}",
    "This does not require memory. {question}",
    "Please keep this concise: {question}",
    "Give the useful answer directly. {question}",
    "In one or two sentences, answer this: {question}",
    "For a teammate: {question}",
    "For a beginner: {question}",
    "Without mentioning personal memory, answer this: {question}",
    "Treat this as a normal chat request: {question}",
    "No remembered facts are needed here. {question}",
    "Use general knowledge only: {question}",
    "Do not bring up stored details. {question}",
    "Just answer the question: {question}",
    "Please give a practical answer: {question}",
    "Give a compact answer: {question}",
    "Reply in plain language: {question}",
    "Give the shortest useful answer: {question}",
    "Answer as if this is unrelated to any prior memory: {question}",
    "Ignore any unrelated remembered facts and answer: {question}",
    "Do not infer anything about me; just answer: {question}",
    "Keep it factual and short: {question}",
    "Use a calm tone: {question}",
    "Give a small example if useful: {question}",
    "Make the answer easy to scan: {question}",
    "Answer without adding extra context: {question}",
    "Focus only on the current request: {question}",
    "Give the practical version: {question}",
    "Please answer the standalone question: {question}",
]
V4_NORMAL_PROMPT_PREFIXES = dedupe(V4_NORMAL_PROMPT_PREFIXES + CURATED_NORMAL_PROMPT_PREFIXES)

V4_MEMORY_IRRELEVANT_PREFIXES = [
    "Ignore unrelated remembered details and answer: {question}",
    "This is not about my pet or any stored code. {question}",
    "Without using personal memory, answer this: {question}",
    "Answer the general question only: {question}",
    "Do not mention any remembered names. {question}",
    "This is unrelated to anything I asked you to remember. {question}",
    "Please answer normally, not from memory: {question}",
    "General question, unrelated to stored facts: {question}",
    "Memory is irrelevant for this turn. {question}",
    "Use the current prompt, not stored personal details: {question}",
    "Do not let prior remembered facts change the answer. {question}",
    "Treat this as a fresh standalone request: {question}",
]
V4_MEMORY_IRRELEVANT_PREFIXES = dedupe(V4_MEMORY_IRRELEVANT_PREFIXES + CURATED_MEMORY_IRRELEVANT_PREFIXES)


def expand_v4_chat_rows(base_rows, prefixes):
    rows = []
    seen = set()
    for question, answer in base_rows:
        for prefix in prefixes:
            expanded = prefix.format(question=question)
            key = (expanded, answer)
            if key in seen:
                continue
            seen.add(key)
            rows.append(key)
    return rows


V4_NORMAL_CHAT_ROWS = expand_v4_chat_rows(ORDINARY_CHAT_ROWS, V4_NORMAL_PROMPT_PREFIXES)
V4_MEMORY_IRRELEVANT_ROWS = expand_v4_chat_rows(ORDINARY_CHAT_ROWS, V4_MEMORY_IRRELEVANT_PREFIXES)
V4_MEMORY_IRRELEVANT_QUESTIONS = [question for question, _ in V4_MEMORY_IRRELEVANT_ROWS]
V4_MEMORY_IRRELEVANT_ANSWERS = [answer for _, answer in V4_MEMORY_IRRELEVANT_ROWS]
V4_NORMAL_AND_ORDINARY_ROWS = V4_NORMAL_CHAT_ROWS + ORDINARY_CHAT_ROWS


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True))
            f.write("\n")


def pack(messages, family, fact_groups):
    return {
        "messages": messages,
        "memory_target": {
            "family": family,
            "fact_groups": fact_groups,
        },
    }


def guardrail_pack(messages, family="ordinary_chat_guardrail"):
    return {
        "messages": messages,
        "memory_target": {
            "mode": "guardrail_only",
            "family": family,
            "fact_groups": [],
        },
    }


def ack(rng):
    return {"role": "assistant", "content": pick(rng, ACKS)}


def validate(messages):
    assert len(messages) >= 2
    for i, message in enumerate(messages):
        expected = "user" if i % 2 == 0 else "assistant"
        assert message["role"] == expected, f"{i}: {message['role']} != {expected}"
        assert isinstance(message["content"], str)


def has_trainable_final_pair(row):
    messages = row["messages"]
    memory_target = row.get("memory_target")
    guardrail_only = (
        isinstance(memory_target, dict)
        and (
            memory_target.get("mode") == "guardrail_only"
            or memory_target.get("family") in {"ordinary_chat_guardrail", "smoltalk"}
        )
    )
    if len(messages) < 4 and not guardrail_only:
        return False
    min_split = 0 if guardrail_only else 1
    return any(
        messages[split]["role"] == "user" and messages[split + 1]["role"] == "assistant"
        for split in range(min_split, len(messages) - 1)
    )


def normal_context_turn(rng):
    question, answer = pick(rng, ORDINARY_CHAT_ROWS)
    return [{"role": "user", "content": question}, {"role": "assistant", "content": answer}]


def make_identity_varied(rng):
    kind = pick(rng, ["dog", "cat", "rabbit", "parrot"])
    name = pick(rng, PET_NAME_POOL)
    distractor_kind = pick_other(rng, ["dog", "cat", "rabbit", "parrot"], kind)
    distractor_name = pick_other(rng, PET_NAME_POOL, name)
    messages = [
        {"role": "user", "content": pick(rng, IDENTITY_FACT_PROMPTS).format(kind=kind, name=name)},
        ack(rng),
    ]
    if rng.random() < 0.45:
        messages.extend(
            [
                {
                    "role": "user",
                    "content": f"Also remember that my {distractor_kind}'s name is {distractor_name}.",
                },
                ack(rng),
            ]
        )
    answer_template = pick(rng, IDENTITY_ANSWERS + IDENTITY_CLEAN_CONTINUATIONS)
    messages.extend(
        [
            {"role": "user", "content": pick(rng, IDENTITY_QUESTIONS).format(kind=kind)},
            {"role": "assistant", "content": answer_template.format(kind=kind, name=name)},
        ]
    )
    validate(messages)
    negatives = [distractor_name] + sample_negatives(rng, PET_NAME_POOL, name, k=5)
    return pack(messages, "curriculum_identity_varied", [build_fact_group(name, negatives)])


def make_stop_after_fact(rng):
    kind = "dog"
    name = pick(rng, PET_NAME_POOL)
    question = pick(rng, [
        "Only answer with the name: what is my dog's name?",
        "Just the name, please: what dog name did I give you?",
        "Answer briefly. What is my dog's name?",
    ])
    answer = pick(rng, [name, f"{name}.", f"It's {name}."])
    messages = [
        {"role": "user", "content": pick(rng, IDENTITY_FACT_PROMPTS).format(kind=kind, name=name)},
        ack(rng),
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer},
    ]
    validate(messages)
    return pack(
        messages,
        "curriculum_stop_after_fact",
        [build_fact_group(name, sample_negatives(rng, PET_NAME_POOL, name, k=6))],
    )


def make_hard_prefix_identity(rng):
    kind = pick(rng, ["dog", "cat", "rabbit"])
    name, negatives = pick(rng, HARD_NAME_PAIRS)
    all_names = PET_NAME_POOL
    explicit_negative = pick(rng, negatives)
    prompt = pick(
        rng,
        [
            "Please remember this exactly: my {kind}'s name is {name}, not {negative}.",
            "For later, store the {kind} name as {name}. Do not confuse it with {negative}.",
            "Long-term note: my {kind}'s name is {name}; {negative} is the wrong name.",
        ],
    ).format(kind=kind, name=name, negative=explicit_negative)
    question = pick(rng, ANSWER_REALIZATION_QUESTIONS).format(kind=kind)
    answer = pick(rng, ANSWER_REALIZATION_ANSWERS).format(kind=kind, name=name)
    messages = [
        {"role": "user", "content": prompt},
        ack(rng),
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer},
    ]
    validate(messages)
    hard_negatives = list(dict.fromkeys(negatives + sample_negatives(rng, all_names, name, k=5)))
    return pack(messages, "curriculum_hard_prefix_identity", [build_fact_group(name, hard_negatives)])


def make_answer_realization(rng):
    kind = pick(rng, ["dog", "cat", "rabbit", "parrot"])
    name = pick(rng, PET_NAME_POOL)
    if rng.random() < 0.40:
        name, hard_negatives = pick(rng, HARD_NAME_PAIRS)
    else:
        hard_negatives = sample_negatives(rng, PET_NAME_POOL, name, k=6)

    messages = [
        {"role": "user", "content": pick(rng, IDENTITY_FACT_PROMPTS).format(kind=kind, name=name)},
        ack(rng),
    ]

    # Interleave an ordinary exchange before recall so the model learns that
    # memory can coexist with normal language instead of hijacking every turn.
    if rng.random() < 0.50:
        question, answer = pick(rng, ORDINARY_CHAT_ROWS)
        messages.extend([{"role": "user", "content": question}, {"role": "assistant", "content": answer}])

    question_template = pick(rng, ANSWER_REALIZATION_QUESTIONS)
    question = question_template.format(kind=kind)
    question_lower = question.lower()
    if "errand" in question_lower:
        answer = f"For the errand, remember to bring water for {name}."
    elif "reminder" in question_lower:
        answer = f"Reminder: your {kind}, {name}, is the one I should remember."
    elif "tiny note" in question_lower or "small note" in question_lower:
        answer = f"Note: {name} is your {kind}'s name."
    elif "sentence" in question_lower and "name" in question_lower:
        answer = f"Your {kind}'s name is {name}."
    elif "only the remembered name" in question_lower or "nothing unrelated" in question_lower:
        answer = name
    else:
        answer = pick(rng, IDENTITY_ANSWERS).format(kind=kind, name=name)
    messages.extend([{"role": "user", "content": question}, {"role": "assistant", "content": answer}])
    validate(messages)
    return pack(
        messages,
        "curriculum_answer_realization",
        [build_fact_group(name, list(dict.fromkeys(hard_negatives)))],
    )


def make_live_session_name_recall(rng):
    # This mirrors the deployed CLI workflow:
    # first process writes only the user fact, second process asks a fresh recall.
    kind = "dog" if rng.random() < 0.70 else pick(rng, ["cat", "rabbit", "parrot"])
    if rng.random() < 0.15:
        name, hard_negatives = pick(rng, HARD_NAME_PAIRS)
    else:
        name = pick(rng, PET_NAME_POOL)
        hard_negatives = sample_negatives(rng, PET_NAME_POOL, name, k=8)
    question = pick_train_eval(rng, LIVE_SESSION_RECALL_QUESTIONS_TRAIN, LIVE_SESSION_RECALL_QUESTIONS_EVAL).format(kind=kind)
    answer = pick_train_eval(rng, LIVE_SESSION_RECALL_ANSWERS_TRAIN, LIVE_SESSION_RECALL_ANSWERS_EVAL).format(kind=kind, name=name)
    messages = [
        {"role": "user", "content": pick(rng, LIVE_SESSION_FACT_PROMPTS).format(kind=kind, name=name)},
        ack(rng),
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer},
    ]
    validate(messages)
    return pack(
        messages,
        "curriculum_live_session_name_recall",
        [build_fact_group(name, list(dict.fromkeys(hard_negatives)))],
    )


V4_EXACT_FACT_PROMPTS = [
    "My {kind}'s name is {value}.",
    "Remember exactly: my {kind}'s name is {value}.",
    "For later, my {kind}'s name is {value}.",
    "Store this verbatim: my {kind}'s name is {value}.",
    "The exact name of my {kind} is {value}.",
    "Please remember this exact {kind} name: {value}.",
    "Small memory note: my {kind}'s name is {value}.",
    "If I ask later, the {kind}'s name is {value}.",
    "Do not normalize this spelling: my {kind}'s name is {value}.",
    "The current {kind} name, character for character, is {value}.",
    "Save the following as my {kind}'s name: {value}.",
    "My {kind} answers to {value}. Please remember the spelling.",
    "The name I want remembered for my {kind} is {value}.",
    "Memory update: my {kind}'s name should be stored as {value}.",
]

V4_EXACT_RECALL_QUESTIONS = [
    "What is my {kind}'s name?",
    "What's my {kind}'s name?",
    "Answer with only the remembered name: what is my {kind}'s name?",
    "Just the name: what is my {kind}'s name?",
    "Do you remember my {kind}'s name?",
    "Tell me the exact name I gave for my {kind}.",
    "What name did I tell you for my {kind}?",
    "Return my {kind}'s name exactly.",
    "Use the remembered spelling: what is my {kind}'s name?",
    "Can you give me my {kind}'s name from memory?",
    "What should you answer if I ask for my {kind}'s name?",
    "Which exact name is stored for my {kind}?",
    "Without adding anything extra, what is my {kind}'s name?",
    "I need the exact remembered {kind} name.",
    "What did I say my {kind} is called?",
    "Please answer naturally: what is my {kind}'s name?",
]

V4_EXACT_ANSWERS = [
    "{value}",
    "{value}.",
    "It's {value}.",
    "Your {kind}'s name is {value}.",
    "You told me your {kind}'s name is {value}.",
    "The name is {value}.",
    "I have {value} stored.",
    "You said it was {value}.",
    "Your remembered {kind} name is {value}.",
    "The exact name I have is {value}.",
]
V4_EXACT_FACT_PROMPTS_TRAIN = dedupe(
    V4_EXACT_FACT_PROMPTS
    + [
        "Please keep this for later: {value} is my {kind}'s name.",
        "Memory note for later: call my {kind} {value}.",
        "I want you to remember that my {kind} is called {value}.",
        "The name to associate with my {kind} is {value}.",
        "For future answers, my {kind}'s name is {value}.",
        "Store this as a personal detail: my {kind} is named {value}.",
        "When I ask about my {kind}, the name is {value}.",
        "Here is the name for my {kind}: {value}.",
        "Keep the {kind} name {value} available for later.",
        "Remember this session detail: my {kind} goes by {value}.",
        "I am telling you once: my {kind}'s name is {value}.",
        "Please save {value} as the name of my {kind}.",
        "The remembered answer for my {kind}'s name should be {value}.",
        "Add this to memory: {kind} name equals {value}.",
        "My {kind} is not the topic right now, but its name is {value}.",
        "For later recall, the name attached to my {kind} is {value}.",
    ]
)
V4_EXACT_FACT_PROMPTS_EVAL = [
    "Please retain this detail: my {kind} is named {value}.",
    "The name I use for my {kind} is {value}.",
    "Make a memory note that my {kind}'s name is {value}.",
    "For a later question, remember {value} as my {kind}'s name.",
    "The stored name for my {kind} should be {value}.",
    "I call my {kind} {value}; please remember it.",
    "Keep this exact association: {kind} name -> {value}.",
    "If I ask about my {kind}, retrieve the name {value}.",
]

V4_EXACT_RECALL_QUESTIONS_TRAIN = dedupe(
    V4_EXACT_RECALL_QUESTIONS
    + compose_prompt_variants(
        [
            "",
            "Quick memory check:",
            "Answer from the saved detail:",
            "Please retrieve this:",
            "For a normal chat answer,",
            "Use what I told you earlier:",
            "I need the stored value:",
            "Could you remind me,",
            "One short answer:",
            "Keep it grounded in memory:",
            "When I ask about the name,",
            "Please answer as if I just asked in chat:",
        ],
        [
            "what is my {kind}'s name?",
            "what name did I give for my {kind}?",
            "what did I call my {kind}?",
            "which name is attached to my {kind}?",
            "what should you call my {kind}?",
            "what is the remembered name for my {kind}?",
            "which name did I ask you to keep for my {kind}?",
            "what is my {kind} named?",
            "what name should appear in the answer for my {kind}?",
            "tell me the name I stored for my {kind}.",
            "give me the saved {kind} name.",
            "say the remembered {kind} name.",
        ],
        [
            "",
            "Use the exact spelling.",
            "Please keep the answer brief.",
            "No explanation is needed.",
            "Answer naturally.",
            "Do not add unrelated details.",
            "One sentence is enough.",
        ],
    )
)
V4_EXACT_RECALL_QUESTIONS_EVAL = dedupe(
    compose_prompt_variants(
        [
            "",
            "From the memory note,",
            "For this direct question,",
            "Please recall:",
            "Without inventing a new value,",
            "I want the saved answer:",
            "Based only on the stored detail,",
            "In ordinary chat form,",
        ],
        [
            "what name belongs to my {kind}?",
            "which name did I provide for my {kind}?",
            "what is the stored name for my {kind}?",
            "what did I say my {kind} was named?",
            "what should the answer be for my {kind}'s name?",
            "which exact {kind} name should you remember?",
            "what name did I ask you to associate with my {kind}?",
            "tell me the saved name for my {kind}.",
        ],
        [
            "",
            "Keep it exact.",
            "Short is fine.",
            "Use normal prose.",
            "Please do not include anything else.",
        ],
    )
)
V4_EXACT_ANSWERS_TRAIN = dedupe(
    V4_EXACT_ANSWERS
    + [
        "The saved name is {value}.",
        "I remember {value}.",
        "For your {kind}, I have {value}.",
        "The name you gave me is {value}.",
        "I have the {kind}'s name as {value}.",
        "{value} is the name I should use.",
        "You asked me to remember {value}.",
        "I have {value} stored for your {kind}.",
        "The remembered answer is {value}.",
        "You called your {kind} {value}.",
        "I should answer {value}.",
        "The stored {kind} name is {value}.",
    ]
)
V4_EXACT_ANSWERS_EVAL = [
    "{value}.",
    "It is {value}.",
    "The stored name is {value}.",
    "I have {value}.",
    "You gave me {value}.",
    "Your {kind} is named {value}.",
    "The answer I have is {value}.",
    "I remember the name as {value}.",
]


def pick_v4_exact_fact_prompt(rng):
    return pick_train_eval(rng, V4_EXACT_FACT_PROMPTS_TRAIN, V4_EXACT_FACT_PROMPTS_EVAL)


def pick_v4_exact_recall_question(rng):
    return pick_train_eval(rng, V4_EXACT_RECALL_QUESTIONS_TRAIN, V4_EXACT_RECALL_QUESTIONS_EVAL)


def pick_v4_exact_answer(rng):
    return pick_train_eval(rng, V4_EXACT_ANSWERS_TRAIN, V4_EXACT_ANSWERS_EVAL)

V4_FIELD_SPECS = [
    (
        "project codename",
        "The exact project codename is {value}.",
        "What project codename should you remember?",
        "The project codename is {value}.",
    ),
    (
        "project codename",
        "Remember this project codename exactly: {value}.",
        "Which project codename did I give you?",
        "{value}.",
    ),
    (
        "backup code",
        "Remember this backup code exactly: {value}.",
        "What backup code did I give you?",
        "{value}.",
    ),
    (
        "backup code",
        "For later, store the backup code as {value}.",
        "Return the backup code from memory.",
        "The backup code is {value}.",
    ),
    (
        "label",
        "Store this exact label for later: {value}.",
        "What exact label did I ask you to remember?",
        "The label is {value}.",
    ),
    (
        "label",
        "Remember this label without changing the spelling: {value}.",
        "Which label is stored?",
        "{value}.",
    ),
    (
        "preferred database",
        "My preferred database for this project is {value}.",
        "What preferred database should you remember?",
        "The preferred database is {value}.",
    ),
    (
        "current branch",
        "The current branch to remember is {value}.",
        "What current branch did I give you?",
        "The current branch is {value}.",
    ),
    (
        "support ticket",
        "Please remember this support ticket exactly: {value}.",
        "Which support ticket did I ask you to remember?",
        "The support ticket is {value}.",
    ),
    (
        "release marker",
        "Store {value} as the release marker for this rollout.",
        "What release marker should you recall?",
        "The release marker is {value}.",
    ),
    (
        "deployment label",
        "The deployment label for later is {value}.",
        "Which deployment label is stored?",
        "{value}.",
    ),
    (
        "incident code",
        "Remember the incident code as {value}.",
        "What incident code did I give you?",
        "The incident code is {value}.",
    ),
    (
        "workspace label",
        "My workspace label for this context is {value}.",
        "What workspace label should you remember?",
        "The workspace label is {value}.",
    ),
    (
        "device nickname",
        "Save this device nickname: {value}.",
        "Which device nickname did I provide?",
        "The device nickname is {value}.",
    ),
    (
        "meeting code",
        "Keep this meeting code in memory: {value}.",
        "What meeting code did I ask you to keep?",
        "The meeting code is {value}.",
    ),
    (
        "archive key",
        "The archive key I need later is {value}.",
        "What archive key should you retrieve?",
        "The archive key is {value}.",
    ),
    (
        "call sign",
        "Use {value} as the call sign for this session.",
        "What call sign did I give you?",
        "The call sign is {value}.",
    ),
    (
        "sample id",
        "Remember this sample id exactly: {value}.",
        "What sample id is saved?",
        "The sample id is {value}.",
    ),
    (
        "calendar tag",
        "The calendar tag for later is {value}.",
        "What calendar tag did I provide?",
        "The calendar tag is {value}.",
    ),
    (
        "shipping alias",
        "Store this shipping alias for later: {value}.",
        "Which shipping alias should you remember?",
        "The shipping alias is {value}.",
    ),
]


def make_v4_code_value(rng):
    left = pick_v4_name(rng).upper()
    middle = rng.randint(10, 999)
    right = pick_v4_name(rng).lower()
    if rng.random() < 0.50:
        return f"{left[:4]}-{middle}-{right[:5]}"
    return f"{left[:3]}{middle}{right[:4]}"


def make_v4_field_value(rng, field):
    if field == "preferred database":
        return pick(rng, DATABASES)
    if field == "current branch":
        return pick(rng, BRANCHES)
    if field in {"project codename", "label", "device nickname", "call sign", "shipping alias"}:
        return pick_v4_name(rng)
    if rng.random() < 0.65:
        return make_v4_code_value(rng)
    return pick_v4_name(rng)


V4_BRANCHPOINT_FACT_PROMPTS_TRAIN = [
    "My {kind}'s name is {name}.",
    "Remember the exact {kind} name: {name}.",
    "The current {kind} name is {name}.",
    "The exact spelling of my {kind}'s name is {name}.",
    "For this session, my {kind}'s name is exactly {name}.",
    "Store {name} as the name for my {kind}.",
    "The name to recall for my {kind} is {name}.",
    "Keep this short name in memory for my {kind}: {name}.",
    "Please save {name} as my {kind}'s name.",
    "When I ask about my {kind}, the answer should be {name}.",
]
V4_BRANCHPOINT_FACT_PROMPTS_EVAL = [
    "Please retain {name} as my {kind}'s name.",
    "The saved {kind} name should be {name}.",
    "I call my {kind} {name}; remember that exact value.",
    "For later recall, my {kind}'s name is {name}.",
]
V4_BRANCHPOINT_PROMPTS_TRAIN = dedupe(
    compose_prompt_variants(
        [
            "",
            "Use the stored spelling:",
            "Quick recall:",
            "Please retrieve the saved name:",
            "The answer may be unusual:",
            "Answer from memory:",
            "For the exact-copy test,",
            "Do the memory lookup:",
        ],
        [
            "what is my {kind}'s name?",
            "which name did I give for my {kind}?",
            "what did I call my {kind}?",
            "give the {kind} name I told you earlier.",
            "return the remembered {kind} name.",
            "what is the stored {kind} name?",
            "say the name associated with my {kind}.",
            "what name should you use for my {kind}?",
        ],
        [
            "",
            "Keep the exact letters.",
            "A short answer is enough.",
            "Use normal chat wording.",
            "Keep the spelling intact.",
        ],
    )
)
V4_BRANCHPOINT_PROMPTS_EVAL = dedupe(
    compose_prompt_variants(
        [
            "",
            "From the stored detail,",
            "Please recall exactly:",
            "For a compact answer,",
            "Retrieve the memory value:",
        ],
        [
            "what is my {kind} named?",
            "what name did I attach to my {kind}?",
            "what is the saved answer for my {kind}'s name?",
            "which {kind} name should you report?",
            "tell me the remembered name for my {kind}.",
        ],
        [
            "",
            "Keep it exact.",
            "Do not include extra context.",
            "Short answer.",
        ],
    )
)
V4_BRANCHPOINT_CONFUSER_PROMPTS_TRAIN = [
    "Use memory rather than the plausible value {negative}: what is my {kind}'s name?",
    "If another name comes to mind, ignore it and give the saved {kind} name.",
    "Choose the stored {kind} name over similar-looking alternatives.",
    "The name may look unusual; use the saved spelling for my {kind}.",
]
V4_BRANCHPOINT_CONFUSER_PROMPTS_EVAL = [
    "Avoid autocomplete drift; what is my {kind}'s saved name?",
    "What is the remembered {kind} name, even if a nearby spelling looks plausible?",
    "Return the saved name for my {kind} instead of a common default.",
]
V4_BRANCHPOINT_ANSWERS_TRAIN = [
    "{name}",
    "{name}.",
    "It's {name}.",
    "Your {kind}'s name is {name}.",
    "The stored name is {name}.",
    "I have {name}.",
]
V4_BRANCHPOINT_ANSWERS_EVAL = [
    "{name}.",
    "It is {name}.",
    "I remember {name}.",
    "The answer is {name}.",
]


def make_v4_exact_copy(rng):
    # High-cardinality exact-copy rows are the antidote to failures like
    # Buh -> Bist and Taloobrook -> Tavarraig. Keep pet-name coverage strong
    # for the deployed test, but leave enough non-pet exact-copy pressure that
    # the learned rule is "copy retrieved memory", not "answer dog templates".
    if rng.random() < 0.66:
        kind = "dog" if rng.random() < 0.62 else pick(rng, ["cat", "rabbit", "parrot", "ferret", "bird"])
        name = pick_v4_name(rng)
        question = pick_v4_exact_recall_question(rng).format(kind=kind)
        answer = pick_v4_exact_answer(rng).format(kind=kind, value=name)
        messages = [
            {"role": "user", "content": pick_v4_exact_fact_prompt(rng).format(kind=kind, value=name)},
            ack(rng),
        ]
        if rng.random() < 0.25:
            ordinary_q, ordinary_a = pick(rng, ORDINARY_CHAT_ROWS)
            messages.extend([{"role": "user", "content": ordinary_q}, {"role": "assistant", "content": ordinary_a}])
        messages.extend([{"role": "user", "content": question}, {"role": "assistant", "content": answer}])
        negatives = sample_v4_negatives(rng, name, k=12)
        value = name
        family = "curriculum_v4_exact_name_copy"
    else:
        field, fact_template, question, answer_template = pick(rng, V4_FIELD_SPECS)
        value = make_v4_field_value(rng, field)
        messages = [
            {"role": "user", "content": fact_template.format(value=value)},
            ack(rng),
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer_template.format(value=value)},
        ]
        negatives = [make_v4_code_value(rng) for _ in range(6)] + sample_v4_negatives(rng, value, k=6)
        family = f"curriculum_v4_exact_{field.replace(' ', '_')}_copy"
    validate(messages)
    return pack(messages, family, [build_fact_group(value, list(dict.fromkeys(negatives)))])


def make_v4_branchpoint_copy(rng):
    kind = "dog"
    name = pick_v4_name(rng)
    negatives = sample_v4_negatives(rng, name, k=14)
    visible_negative = pick(rng, negatives)
    fact = pick_train_eval(rng, V4_BRANCHPOINT_FACT_PROMPTS_TRAIN, V4_BRANCHPOINT_FACT_PROMPTS_EVAL).format(kind=kind, name=name)
    if rng.random() < 0.18:
        prompt = pick_train_eval(rng, V4_BRANCHPOINT_CONFUSER_PROMPTS_TRAIN, V4_BRANCHPOINT_CONFUSER_PROMPTS_EVAL).format(
            kind=kind,
            negative=visible_negative,
        )
    else:
        prompt = pick_train_eval(rng, V4_BRANCHPOINT_PROMPTS_TRAIN, V4_BRANCHPOINT_PROMPTS_EVAL).format(kind=kind)
    answer = pick_train_eval(rng, V4_BRANCHPOINT_ANSWERS_TRAIN, V4_BRANCHPOINT_ANSWERS_EVAL).format(kind=kind, name=name)
    messages = [
        {"role": "user", "content": fact},
        ack(rng),
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": answer},
    ]
    validate(messages)
    return pack(messages, "curriculum_v4_branchpoint_exact_copy", [build_fact_group(name, negatives)])


NO_MEMORY_TARGETS = [
    ("dog", "dog's name"),
    ("cat", "cat's name"),
    ("rabbit", "rabbit's name"),
    ("bird", "bird's name"),
    ("ferret", "ferret's name"),
    ("project codename", "project codename"),
    ("backup code", "backup code"),
    ("preferred database", "preferred database"),
    ("current branch", "current branch"),
    ("favorite editor", "favorite editor"),
    ("favorite color", "favorite color"),
    ("home city", "home city"),
    ("deployment region", "deployment region"),
    ("preferred language", "preferred programming language"),
]

NO_MEMORY_QUESTION_TEMPLATES = [
    "What is my {label}?",
    "Do you remember my {label}?",
    "Answer briefly: what is my {label}?",
    "If you know it from memory, what is my {label}?",
    "What {label} did I give you?",
    "Tell me the remembered {label}.",
    "Can you recall my {label}?",
    "What should you use as my {label}?",
    "What value do you have stored for my {label}?",
    "Do you have my {label} saved?",
    "Without guessing, what is my {label}?",
    "Only answer if I told you: what is my {label}?",
    "If that value is missing, say so: what is my {label}?",
    "Did I ever tell you my {label}?",
    "Do you actually know my {label}, or is it missing?",
    "What is the saved value for my {label}, if any?",
    "Please don't infer it. What is my {label}?",
    "From stored memory only, what is my {label}?",
]

NO_MEMORY_ANSWERS = [
    "I don't have that stored yet.",
    "I don't know yet; you have not told me.",
    "I don't have a remembered value for that.",
    "I do not have that in memory.",
    "I can't answer from memory because you have not given me that value.",
    "I don't have enough information to answer that.",
    "You have not told me that yet.",
    "I should not guess; I do not have that stored.",
    "I don't see a remembered value for that.",
    "I have no saved value for that yet.",
    "Not yet; I would need you to tell me first.",
    "I don't know that from memory.",
    "I cannot recall that because it has not been provided.",
    "I do not have a stored answer for that.",
    "I should not invent one; I do not have it stored.",
    "I don't have that value in memory, so I should not guess.",
    "You have not provided that detail yet.",
    "I do not know that from the information stored here.",
    "I have no remembered value for it yet.",
    "I cannot answer that from memory yet.",
]
NO_MEMORY_QUESTION_TEMPLATES_TRAIN = dedupe(
    NO_MEMORY_QUESTION_TEMPLATES
    + [
        "Have I given you my {label} yet?",
        "Is there a saved {label} for me?",
        "Check memory for my {label}.",
        "What do you have recorded as my {label}?",
        "Can you answer my {label} from saved context?",
        "If this was saved earlier, what is my {label}?",
        "Do you have enough saved context to answer my {label}?",
        "What is the memory value for my {label}?",
        "Tell me my {label} only if it was provided.",
        "Is my {label} available in memory?",
        "Can you retrieve my {label}?",
        "What answer is saved for my {label}?",
    ]
)
NO_MEMORY_QUESTION_TEMPLATES_EVAL = [
    "Have I told you my {label}?",
    "What stored value do you have for my {label}?",
    "Can you retrieve my {label} from memory?",
    "Is my {label} known in this session?",
    "What should you say if I ask for my {label}?",
    "Do you have a saved answer for my {label}?",
    "Please check whether my {label} is available.",
    "From memory, what is my {label}?",
]
NO_MEMORY_ANSWERS_TRAIN = dedupe(
    NO_MEMORY_ANSWERS
    + [
        "That detail is missing from memory.",
        "I need you to provide that before I can recall it.",
        "There is no saved value for that yet.",
        "I have no stored answer for that detail.",
        "That has not been saved in memory.",
        "I would need you to tell me first.",
        "There is not enough saved context for me to answer.",
        "I have no memory entry for that value.",
        "That value is unavailable from memory right now.",
        "I can answer once you provide that detail.",
        "I do not have a saved detail for that.",
        "That specific value has not been provided.",
        "I should ask you for that value instead of guessing.",
        "I do not see that value in memory.",
        "No saved value is available for that yet.",
        "I need the detail first; it is not stored yet.",
    ]
)
NO_MEMORY_ANSWERS_EVAL = [
    "I do not have that saved yet.",
    "That value is not in memory yet.",
    "I have no stored value for that.",
    "You have not provided that detail yet.",
    "I cannot retrieve that because it has not been saved.",
    "I should not guess; that value is missing.",
    "There is no remembered answer for that yet.",
    "I would need you to tell me first.",
]
NO_MEMORY_STYLE_PREFIXES_TRAIN = [
    "{question}",
    "Check memory first: {question}",
    "Answer honestly from stored context: {question}",
    "If the value is absent, say so: {question}",
    "Please avoid guessing: {question}",
    "Use saved details only: {question}",
    "For a direct answer: {question}",
    "Keep the reply short: {question}",
]
NO_MEMORY_STYLE_PREFIXES_EVAL = [
    "{question}",
    "From memory only: {question}",
    "Please be direct: {question}",
    "If it is missing, say that clearly: {question}",
]


def make_v4_no_memory_deference(rng):
    # These rows are deliberately guardrail-only: if no relevant memory was
    # written, the model should not invent Luna, Hashrouma, or a plausible code.
    kind, label = pick(rng, NO_MEMORY_TARGETS)
    messages = []
    context_case = rng.random()
    if context_case < 0.40:
        irrelevant = pick_v4_name(rng)
        messages = [
            {"role": "user", "content": f"For later, remember that my city is {pick(rng, CITIES)}."},
            ack(rng),
            {"role": "user", "content": f"Also remember the unrelated label {irrelevant}."},
            ack(rng),
        ]
    elif context_case < 0.72:
        messages.extend(normal_context_turn(rng))
        if rng.random() < 0.35:
            messages.extend(normal_context_turn(rng))
    else:
        messages = [
            {"role": "user", "content": pick(rng, [
                "Before the next question, keep answers short and avoid guessing.",
                "If I ask for a remembered value I have not given, say that you do not know.",
                "For this chat, do not invent personal details.",
                "Use memory only when I have actually provided the detail.",
            ])},
            {"role": "assistant", "content": pick(rng, [
                "Understood. I will avoid guessing.",
                "Understood. I will not invent missing details.",
                "Got it. I will only use details you have provided.",
                "Understood. If the value is missing, I will say so.",
            ])},
        ]
    if kind in {"dog", "cat", "rabbit"}:
        label = f"{kind}'s name"
    question = pick_train_eval(rng, NO_MEMORY_QUESTION_TEMPLATES_TRAIN, NO_MEMORY_QUESTION_TEMPLATES_EVAL).format(label=label)
    if rng.random() < 0.24:
        question = pick_train_eval(rng, NO_MEMORY_STYLE_PREFIXES_TRAIN, NO_MEMORY_STYLE_PREFIXES_EVAL).format(question=question)
    answer = pick_train_eval(rng, NO_MEMORY_ANSWERS_TRAIN, NO_MEMORY_ANSWERS_EVAL)
    messages.extend([{"role": "user", "content": question}, {"role": "assistant", "content": answer}])
    validate(messages)
    return guardrail_pack(messages, "curriculum_v4_no_memory_deference")


def make_v4_normal_chat_guardrail(rng):
    # Pure no-memory assistant behavior. These rows keep the model fluent and
    # useful while V4 spends many updates on exact-copy memory pressure.
    question, answer = pick(rng, V4_NORMAL_CHAT_ROWS)
    messages = normal_context_turn(rng)
    if rng.random() < 0.30:
        messages.extend(normal_context_turn(rng))
    messages.extend([{"role": "user", "content": question}, {"role": "assistant", "content": answer}])
    validate(messages)
    return guardrail_pack(messages, "curriculum_v4_normal_chat_guardrail")


def make_v4_irrelevant_memory_chat_guardrail(rng):
    # The model must learn the switch: memory can be present, but unrelated
    # turns should remain ordinary helpful chat instead of dragging in a name.
    kind = pick(rng, ["dog", "cat", "rabbit", "project codename", "backup code"])
    remembered_name = pick_v4_name(rng)
    messages = []
    if kind in {"dog", "cat", "rabbit"}:
        messages.extend(
            [
                {
                    "role": "user",
                    "content": pick_v4_exact_fact_prompt(rng).format(kind=kind, value=remembered_name),
                },
                ack(rng),
            ]
        )
    elif kind == "project codename":
        messages.extend(
            [
                {"role": "user", "content": f"Remember the project codename exactly: {remembered_name}."},
                ack(rng),
            ]
        )
    else:
        messages.extend(
            [
                {"role": "user", "content": f"Store this backup code exactly: {make_v4_code_value(rng)}."},
                ack(rng),
            ]
        )
    if rng.random() < 0.45:
        city = pick(rng, CITIES)
        messages.extend(
            [
                {"role": "user", "content": f"Also remember that I live in {city}."},
                ack(rng),
            ]
        )
    if rng.random() < 0.50:
        idx = rng.randrange(len(V4_MEMORY_IRRELEVANT_QUESTIONS))
        question = V4_MEMORY_IRRELEVANT_QUESTIONS[idx]
        answer = V4_MEMORY_IRRELEVANT_ANSWERS[idx]
    else:
        question, answer = pick(rng, V4_NORMAL_AND_ORDINARY_ROWS)
        question = f"{question} Do not mention remembered personal details unless they are relevant."
    messages.extend([{"role": "user", "content": question}, {"role": "assistant", "content": answer}])
    validate(messages)
    return guardrail_pack(messages, "curriculum_v4_irrelevant_memory_chat_guardrail")


V4_REAL_FAILURE_NAMES = [
    "Buh", "Qo", "Qob", "Taloobrook", "Taloobrook-Fen", "Pebble-Cloud",
    "Juniper", "Clover", "Miso", "Fig", "Biscuit", "Pepper", "Zenkigrove",
    "Lug", "Kleal", "Nap", "Tavarraig", "Hashrouma",
    "Taloobrook", "Buh", "Qo", "Pebble-Cloud", "Juniper", "Clover",
    "Rook-7", "Niv", "Vexa", "Olo", "Brindle-Snow", "Zirofen",
]

V4_REAL_FAILURE_CONFUSERS = [
    "Bist", "Tattooed", "Luna", "Polly", "Miso", "Figaro", "Cloverleaf",
    "Juniper-Bell", "Hashrouma", "JSTOR", "Milankov", "Bav", "Babglol",
    "Biscuit", "Pepper", "Coco", "Max", "Buddy", "Ruby", "Tallowbrook",
    "Talo-Brook", "Buhh", "Qob", "Cloud", "Pebble", "Juniperberry",
]


def make_v4_failure_replay(rng):
    # A compact replay buffer for the exact failures observed in chat/eval:
    # short names becoming plausible near-neighbors, long names smoothing into
    # words, old-vs-current reversals, and fluent no-memory defaults.
    case = pick(rng, ["exact", "exact", "current", "current", "no_memory"])
    kind = "dog" if rng.random() < 0.82 else pick(rng, ["cat", "rabbit", "project codename", "label"])
    if case == "no_memory":
        _, label = pick(rng, NO_MEMORY_TARGETS)
        messages = normal_context_turn(rng)
        if rng.random() < 0.45:
            messages.extend(
                [
                    {"role": "user", "content": f"Remember this unrelated city: {pick(rng, CITIES)}."},
                    ack(rng),
                ]
            )
        question = pick_train_eval(rng, NO_MEMORY_QUESTION_TEMPLATES_TRAIN, NO_MEMORY_QUESTION_TEMPLATES_EVAL).format(label=label)
        if rng.random() < 0.20:
            question = pick_train_eval(rng, NO_MEMORY_STYLE_PREFIXES_TRAIN, NO_MEMORY_STYLE_PREFIXES_EVAL).format(question=question)
        messages.extend([{"role": "user", "content": question}, {"role": "assistant", "content": pick_train_eval(rng, NO_MEMORY_ANSWERS_TRAIN, NO_MEMORY_ANSWERS_EVAL)}])
        validate(messages)
        return guardrail_pack(messages, "curriculum_v4_failure_replay_no_memory")

    if case == "current":
        old_value = pick(rng, V4_REAL_FAILURE_NAMES + V4_REAL_FAILURE_CONFUSERS)
        new_value = pick_other(rng, V4_REAL_FAILURE_NAMES + V4_NAME_POOL[:4096], old_value)
        field = f"{kind} name" if kind in {"dog", "cat", "rabbit"} else kind
        messages = [
            {"role": "user", "content": f"My {field} used to be {old_value}."},
            ack(rng),
            {"role": "user", "content": f"Correction: the current {field} is {new_value}, not {old_value}."},
            {"role": "assistant", "content": f"Understood. {new_value} is current; {old_value} is old."},
            {"role": "user", "content": pick(rng, [
                f"What is my current {field}?",
                f"After the correction, what is the current {field}?",
                f"Do not answer with the old value. What is the current {field}?",
                f"Which {field} is current now?",
            ])},
            {"role": "assistant", "content": pick(rng, [
                f"{new_value}.",
                f"It's {new_value}.",
                f"The current {field} is {new_value}.",
                f"Your current {field} is {new_value}.",
            ])},
        ]
        validate(messages)
        return pack(
            messages,
            "curriculum_v4_failure_replay_current_binding",
            [build_fact_group(new_value, [old_value] + sample_v4_negatives(rng, new_value, k=12))],
        )

    value = pick(rng, V4_REAL_FAILURE_NAMES + [pick_v4_name(rng) for _ in range(8)])
    confusers = list(dict.fromkeys(V4_REAL_FAILURE_CONFUSERS + sample_v4_negatives(rng, value, k=12)))
    visible_negative = pick(rng, confusers)
    if kind in {"dog", "cat", "rabbit"}:
        fact = pick_v4_exact_fact_prompt(rng).format(kind=kind, value=value)
        if rng.random() < 0.18:
            question = pick_train_eval(rng, V4_BRANCHPOINT_CONFUSER_PROMPTS_TRAIN, V4_BRANCHPOINT_CONFUSER_PROMPTS_EVAL).format(
                kind=kind,
                negative=visible_negative,
            )
        else:
            question = pick_train_eval(rng, V4_BRANCHPOINT_PROMPTS_TRAIN, V4_BRANCHPOINT_PROMPTS_EVAL).format(kind=kind)
        answer = pick_train_eval(rng, V4_BRANCHPOINT_ANSWERS_TRAIN, V4_BRANCHPOINT_ANSWERS_EVAL).format(kind=kind, name=value)
    else:
        fact = f"Store this exact {kind} for later: {value}."
        question = pick(rng, [
            f"What exact {kind} did I give you?",
            f"Return the exact remembered {kind}.",
            f"Do not guess {visible_negative}; what {kind} did I give you?",
        ])
        answer = pick(rng, [value, f"{value}.", f"The {kind} is {value}."])
    messages = [{"role": "user", "content": fact}, ack(rng)]
    if rng.random() < 0.35:
        messages.extend(normal_context_turn(rng))
    messages.extend([{"role": "user", "content": question}, {"role": "assistant", "content": answer}])
    validate(messages)
    return pack(
        messages,
        "curriculum_v4_failure_replay_exact_copy",
        [build_fact_group(value, confusers)],
    )


V5_CHAR_ALPHABET = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 -_."
V5_MEMORY_LABELS = [
    "dog name",
    "cat name",
    "project codename",
    "backup code",
    "deployment region",
    "database alias",
    "current branch",
    "incident id",
    "ticket code",
    "device nickname",
    "contact alias",
    "calendar tag",
    "shipping alias",
    "lab sample",
    "short note",
    "favorite editor",
    "workspace label",
    "feature flag",
    "test account",
    "secret hint",
]
V5_EXACT_FACT_TEMPLATES_TRAIN = [
    "Remember the exact value for {label}: [{value}]",
    "Store this exact {label} for later: [{value}]",
    "For future questions, the {label} is exactly [{value}].",
    "Save this {label} verbatim: [{value}]",
    "Memory write: {label} = [{value}]",
    "Please keep this exact string as my {label}: [{value}]",
    "The current {label}, character for character, is [{value}].",
    "Do not normalize this: my {label} is [{value}].",
    "Record the {label}; the exact value is [{value}].",
    "Use this later as the {label}: [{value}]",
    "The remembered {label} should be [{value}], not a similar word.",
    "Exact memory item: {label} -> [{value}]",
]
V5_EXACT_FACT_TEMPLATES_EVAL = [
    "Keep this exact {label} in memory: [{value}]",
    "The saved value for {label} is [{value}].",
    "Please remember this verbatim as the {label}: [{value}]",
    "Store {label} exactly as [{value}].",
]
V5_EXACT_QUESTIONS_TRAIN = [
    "What exact value did I give you for {label}?",
    "Return only the remembered {label}.",
    "What is the saved {label}?",
    "Answer with the exact {label}.",
    "Do you remember the exact value for {label}?",
    "Please recall my {label} exactly.",
    "What string is stored under {label}?",
    "From memory, what is {label}?",
    "No guessing: what is the remembered {label}?",
    "What should you output for {label}?",
    "Tell me the {label}, character for character.",
    "Use the stored value: what is {label}?",
]
V5_EXACT_QUESTIONS_EVAL = [
    "What is my remembered {label}?",
    "Recall the exact {label}.",
    "Which value did I store for {label}?",
    "What is the memory value for {label}?",
]
V5_EXACT_ANSWERS_TRAIN = [
    "{value}",
    "{value}.",
    "The {label} is {value}.",
    "It is {value}.",
    "The exact value is {value}.",
    "I have {value} stored for {label}.",
]
V5_EXACT_ANSWERS_EVAL = [
    "{value}",
    "The {label} is {value}.",
    "It is {value}.",
]
V5_NO_MEMORY_LABELS = dedupe(V5_MEMORY_LABELS + [label for _, label in NO_MEMORY_TARGETS])
V5_TOKEN_BUCKETS = [(1, 1), (2, 2), (3, 3), (4, 6), (7, 12)]
V5_KEY_ALPHABET = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_."

V5_RANDOM_KEY_WRITE_TEMPLATES_TRAIN = [
    "Remember this exact key/value pair: {key} = [{value}]",
    "Store field {key} with value [{value}].",
    "Memory write: key {key} has exact value [{value}].",
    "Save [{value}] under the field name {key}.",
    "For later lookup, {key} -> [{value}].",
    "Record this random field: {key}: [{value}]",
    "The value stored under {key} is exactly [{value}].",
    "Keep this binding verbatim: {key} maps to [{value}].",
]
V5_RANDOM_KEY_WRITE_TEMPLATES_EVAL = [
    "Store the exact binding {key} = [{value}].",
    "Please remember that field {key} is [{value}].",
    "Save this lookup entry: {key} -> [{value}].",
]

V5_RANDOM_KEY_QUERY_TEMPLATES_TRAIN = [
    "What did I store for {key}?",
    "Return the value for field {key}.",
    "Which value is bound to {key}?",
    "Look up {key} from memory.",
    "What is the exact value under key {key}?",
    "Use the saved binding: {key} equals what?",
    "No guessing: what value did I save for {key}?",
    "Read field {key}.",
]
V5_RANDOM_KEY_QUERY_TEMPLATES_EVAL = [
    "What value did I save under {key}?",
    "Retrieve {key}.",
    "What is stored for key {key}?",
]

V5_RANDOM_KEY_ANSWERS_TRAIN = [
    "{value}",
    "{value}.",
    "The value for {key} is {value}.",
    "I have {value} stored for {key}.",
    "Field {key} is {value}.",
]
V5_RANDOM_KEY_ANSWERS_EVAL = [
    "{value}",
    "The value for {key} is {value}.",
]

V5_RANDOM_KEY_MISSING_ANSWERS_TRAIN = [
    "I do not have a value stored for {key}.",
    "You did not store {key}.",
    "I do not know a saved value for {key}.",
    "There is no stored value for {key}.",
    "I only know values for keys you actually stored; {key} was not stored.",
]
V5_RANDOM_KEY_MISSING_ANSWERS_EVAL = [
    "I do not have a value stored for {key}.",
    "You did not store {key}.",
]


def make_v5_char_value(rng, min_len=1, max_len=15):
    for _ in range(128):
        raw = "".join(rng.choice(V5_CHAR_ALPHABET) for _ in range(rng.randint(min_len, max_len)))
        value = raw.strip()
        if value:
            return value
    return "Q7"


def make_v5_random_key(rng, min_len=2, max_len=12):
    for _ in range(128):
        raw = "".join(rng.choice(V5_KEY_ALPHABET) for _ in range(rng.randint(min_len, max_len)))
        key = raw.strip(".-_")
        if key:
            return key
    return "rK_2"


def mutate_v5_key(rng, key):
    if not key:
        return make_v5_random_key(rng)
    transforms = ["replace", "insert", "delete", "case", "suffix"]
    for _ in range(64):
        chars = list(key)
        transform = rng.choice(transforms)
        if transform == "replace" and chars:
            idx = rng.randrange(len(chars))
            chars[idx] = rng.choice(V5_KEY_ALPHABET.replace(chars[idx], "") or V5_KEY_ALPHABET)
        elif transform == "insert":
            idx = rng.randrange(len(chars) + 1)
            chars.insert(idx, rng.choice(V5_KEY_ALPHABET))
        elif transform == "delete" and len(chars) > 1:
            del chars[rng.randrange(len(chars))]
        elif transform == "case" and any(ch.isalpha() for ch in chars):
            idxs = [idx for idx, ch in enumerate(chars) if ch.isalpha()]
            idx = rng.choice(idxs)
            chars[idx] = chars[idx].swapcase()
        else:
            chars.append(rng.choice(V5_KEY_ALPHABET))
        candidate = "".join(chars).strip(".-_")
        if candidate and candidate != key:
            return candidate
    return make_v5_random_key(rng)


def make_v5_random_value(rng):
    return make_v5_token_value(rng) if rng.random() < 0.50 else make_v5_char_value(rng)


def is_clean_token_value(value):
    if not value:
        return False
    if len(value) > 48:
        return False
    if "<|" in value or "|>" in value or "[" in value or "]" in value:
        return False
    return all(32 <= ord(ch) <= 126 for ch in value)


def make_v5_token_value(rng, token_bucket=None):
    tokenizer = get_curriculum_tokenizer()
    if tokenizer is None:
        return make_v5_char_value(rng, min_len=3, max_len=15)
    if token_bucket is None:
        token_bucket = pick(rng, V5_TOKEN_BUCKETS)
    min_tokens, max_tokens = token_bucket
    special_ids = set()
    for special in tokenizer.get_special_tokens():
        try:
            special_ids.add(tokenizer.encode_special(special))
        except Exception:
            pass
    vocab_size = tokenizer.get_vocab_size()
    for _ in range(256):
        ids = []
        for _token_idx in range(rng.randint(min_tokens, max_tokens)):
            token_id = rng.randrange(vocab_size)
            if token_id in special_ids:
                continue
            ids.append(token_id)
        if not ids:
            continue
        value = tokenizer.decode(ids).replace("\n", " ").replace("\t", " ").strip()
        while "  " in value:
            value = value.replace("  ", " ")
        if is_clean_token_value(value):
            return value
    return make_v5_char_value(rng, min_len=3, max_len=15)


def mutate_v5_value(rng, value):
    if not value:
        return make_v5_char_value(rng)
    chars = list(value)
    idx = rng.randrange(len(chars))
    replacement = rng.choice(V5_CHAR_ALPHABET.replace(chars[idx], "") or V5_CHAR_ALPHABET)
    chars[idx] = replacement
    mutated = "".join(chars).strip()
    if mutated and mutated != value:
        return mutated
    return make_v5_char_value(rng, min_len=max(1, len(value) - 1), max_len=min(15, len(value) + 1))


def sample_v5_confusers(rng, value, k=12):
    confusers = []
    seen = {value}
    while len(confusers) < k:
        case = rng.random()
        if case < 0.38:
            candidate = mutate_v5_value(rng, value)
        elif case < 0.68:
            candidate = make_v5_char_value(rng, min_len=max(1, min(15, len(value) - 2)), max_len=min(15, len(value) + 2))
        elif case < 0.84:
            candidate = make_v5_token_value(rng)
        else:
            candidate = pick_v4_name(rng)
        if candidate and candidate not in seen:
            seen.add(candidate)
            confusers.append(candidate)
    return confusers


def make_v5_exact_row(rng, value, family, label=None):
    label = label or pick(rng, V5_MEMORY_LABELS)
    fact = pick_train_eval(rng, V5_EXACT_FACT_TEMPLATES_TRAIN, V5_EXACT_FACT_TEMPLATES_EVAL).format(label=label, value=value)
    question = pick_train_eval(rng, V5_EXACT_QUESTIONS_TRAIN, V5_EXACT_QUESTIONS_EVAL).format(label=label)
    answer = pick_train_eval(rng, V5_EXACT_ANSWERS_TRAIN, V5_EXACT_ANSWERS_EVAL).format(label=label, value=value)
    messages = [{"role": "user", "content": fact}, ack(rng)]
    if rng.random() < 0.40:
        messages.extend(normal_context_turn(rng))
    if rng.random() < 0.22:
        other_label = pick_other(rng, V5_MEMORY_LABELS, label)
        other_value = make_v5_char_value(rng)
        messages.extend(
            [
                {
                    "role": "user",
                    "content": pick_train_eval(rng, V5_EXACT_FACT_TEMPLATES_TRAIN, V5_EXACT_FACT_TEMPLATES_EVAL).format(
                        label=other_label,
                        value=other_value,
                    ),
                },
                ack(rng),
            ]
        )
    messages.extend([{"role": "user", "content": question}, {"role": "assistant", "content": answer}])
    validate(messages)
    return pack(messages, family, [build_fact_group(value, sample_v5_confusers(rng, value, k=14))])


def make_v5_arbitrary_char_copy(rng):
    return make_v5_exact_row(
        rng,
        make_v5_char_value(rng),
        "curriculum_v5_arbitrary_char_copy",
        label=pick(rng, V5_MEMORY_LABELS),
    )


def make_v5_arbitrary_token_copy(rng):
    return make_v5_exact_row(
        rng,
        make_v5_token_value(rng),
        "curriculum_v5_arbitrary_token_copy",
        label=pick(rng, V5_MEMORY_LABELS),
    )


def make_v5_memory_variety(rng):
    label = pick(rng, V5_MEMORY_LABELS)
    case = rng.random()
    if case < 0.30:
        value = pick_v4_name(rng)
    elif case < 0.58:
        value = make_v5_char_value(rng)
    elif case < 0.78:
        value = make_v5_token_value(rng)
    elif case < 0.90:
        value = make_v4_code_value(rng)
    else:
        value = pick(rng, BRANCHES + DATABASES + REGIONS + EDITORS)
    return make_v5_exact_row(rng, value, "curriculum_v5_memory_variety", label=label)


def make_v5_no_memory_guardrail(rng):
    label = pick(rng, V5_NO_MEMORY_LABELS)
    messages = []
    if rng.random() < 0.46:
        messages.extend(normal_context_turn(rng))
    if rng.random() < 0.55:
        remembered_label = pick_other(rng, V5_MEMORY_LABELS, label)
        remembered_value = make_v5_char_value(rng)
        messages.extend(
            [
                {
                    "role": "user",
                    "content": pick_train_eval(rng, V5_EXACT_FACT_TEMPLATES_TRAIN, V5_EXACT_FACT_TEMPLATES_EVAL).format(
                        label=remembered_label,
                        value=remembered_value,
                    ),
                },
                ack(rng),
            ]
        )
    question = pick_train_eval(rng, NO_MEMORY_QUESTION_TEMPLATES_TRAIN, NO_MEMORY_QUESTION_TEMPLATES_EVAL).format(label=label)
    if rng.random() < 0.30:
        question = pick_train_eval(rng, NO_MEMORY_STYLE_PREFIXES_TRAIN, NO_MEMORY_STYLE_PREFIXES_EVAL).format(question=question)
    answer = pick_train_eval(rng, NO_MEMORY_ANSWERS_TRAIN, NO_MEMORY_ANSWERS_EVAL)
    messages.extend([{"role": "user", "content": question}, {"role": "assistant", "content": answer}])
    validate(messages)
    return guardrail_pack(messages, "curriculum_v5_no_memory_guardrail")


def make_v5_irrelevant_memory_guardrail(rng):
    label = pick(rng, V5_MEMORY_LABELS)
    value = make_v5_token_value(rng) if rng.random() < 0.45 else make_v5_char_value(rng)
    messages = [
        {
            "role": "user",
            "content": pick_train_eval(rng, V5_EXACT_FACT_TEMPLATES_TRAIN, V5_EXACT_FACT_TEMPLATES_EVAL).format(
                label=label,
                value=value,
            ),
        },
        ack(rng),
    ]
    if rng.random() < 0.35:
        other_label = pick_other(rng, V5_MEMORY_LABELS, label)
        messages.extend(
            [
                {
                    "role": "user",
                    "content": pick_train_eval(rng, V5_EXACT_FACT_TEMPLATES_TRAIN, V5_EXACT_FACT_TEMPLATES_EVAL).format(
                        label=other_label,
                        value=make_v5_char_value(rng),
                    ),
                },
                ack(rng),
            ]
        )
    question, answer = pick(rng, V4_NORMAL_CHAT_ROWS)
    if rng.random() < 0.30:
        question = f"{question} Do not use unrelated saved values."
    messages.extend([{"role": "user", "content": question}, {"role": "assistant", "content": answer}])
    validate(messages)
    return guardrail_pack(messages, "curriculum_v5_irrelevant_memory_guardrail")


def v5_random_key_write_message(rng, key, value):
    return {
        "role": "user",
        "content": pick_train_eval(
            rng,
            V5_RANDOM_KEY_WRITE_TEMPLATES_TRAIN,
            V5_RANDOM_KEY_WRITE_TEMPLATES_EVAL,
        ).format(key=key, value=value),
    }


def v5_random_key_query(rng, key):
    return pick_train_eval(
        rng,
        V5_RANDOM_KEY_QUERY_TEMPLATES_TRAIN,
        V5_RANDOM_KEY_QUERY_TEMPLATES_EVAL,
    ).format(key=key)


def v5_random_key_answer(rng, key, value):
    return pick_train_eval(
        rng,
        V5_RANDOM_KEY_ANSWERS_TRAIN,
        V5_RANDOM_KEY_ANSWERS_EVAL,
    ).format(key=key, value=value)


def v5_random_key_missing_answer(rng, key):
    return pick_train_eval(
        rng,
        V5_RANDOM_KEY_MISSING_ANSWERS_TRAIN,
        V5_RANDOM_KEY_MISSING_ANSWERS_EVAL,
    ).format(key=key)


def make_v5_random_key_records(rng, min_pairs=1, max_pairs=4):
    count = rng.randint(min_pairs, max_pairs)
    records = []
    seen_keys = set()
    seen_values = set()
    while len(records) < count:
        key = make_v5_random_key(rng)
        value = make_v5_random_value(rng)
        if key in seen_keys or value in seen_values:
            continue
        seen_keys.add(key)
        seen_values.add(value)
        records.append((key, value))
    return records


def random_key_negative_query(rng, records):
    keys = [key for key, _value in records]
    values = [value for _key, value in records]
    case = rng.random()
    if case < 0.40:
        candidate = make_v5_random_key(rng)
    elif case < 0.72:
        candidate = mutate_v5_key(rng, rng.choice(keys))
    elif case < 0.88:
        candidate = rng.choice(values)
    else:
        candidate = make_v5_char_value(rng, min_len=2, max_len=12)
    seen = set(keys)
    for _ in range(128):
        if candidate and candidate not in seen:
            return candidate
        candidate = mutate_v5_key(rng, rng.choice(keys)) if rng.random() < 0.70 else make_v5_random_key(rng)
    return "missing_" + make_v5_random_key(rng)


def make_v5_random_key_binding(rng):
    """Random key/value binding with exact positives and exact negative lookup guardrails."""
    case = rng.random()
    records = make_v5_random_key_records(rng, 1, 1 if case < 0.52 else 4)
    messages = []
    for key, value in records:
        messages.extend([v5_random_key_write_message(rng, key, value), ack(rng)])
        if rng.random() < 0.10:
            messages.extend(normal_context_turn(rng))

    if case < 0.44:
        # Single-key or multi-key positive lookup: copy the value bound to the exact random key.
        key, value = rng.choice(records)
        question = v5_random_key_query(rng, key)
        answer = v5_random_key_answer(rng, key, value)
        hard_negatives = [other_value for other_key, other_value in records if other_key != key]
        hard_negatives.extend(sample_v5_confusers(rng, value, k=12))
        hard_negatives.extend([other_key for other_key, _other_value in records if other_key != key])
        messages.extend([{"role": "user", "content": question}, {"role": "assistant", "content": answer}])
        validate(messages)
        return pack(messages, "curriculum_v5_random_key_binding_positive", [build_fact_group(value, hard_negatives)])

    if case < 0.70:
        # Random absent key: the existence of some memory must not trigger a nearby value.
        query_key = random_key_negative_query(rng, records)
    elif case < 0.88:
        # Near miss: one edit/case/suffix away from a real key should still be missing.
        query_key = mutate_v5_key(rng, rng.choice(records)[0])
        while query_key in {key for key, _value in records}:
            query_key = mutate_v5_key(rng, query_key)
    else:
        # Role separation: the stored value is not itself a key unless it was explicitly stored.
        query_key = rng.choice(records)[1]

    question = v5_random_key_query(rng, query_key)
    answer = v5_random_key_missing_answer(rng, query_key)
    messages.extend([{"role": "user", "content": question}, {"role": "assistant", "content": answer}])
    validate(messages)
    return guardrail_pack(messages, "curriculum_v5_random_key_binding_negative")


def make_current_old_binding(rng):
    field = pick(rng, ["dog name", "cat name", "branch", "database", "plan"])
    value_pool = PET_NAME_POOL if "name" in field else BRANCHES if field == "branch" else DATABASES if field == "database" else PLAN_NAMES
    old_value = pick(rng, value_pool)
    new_value = pick_other(rng, value_pool, old_value)
    if field in {"branch", "database", "plan"}:
        intro = f"Please remember that the {field} is {old_value}."
        update = f"Correction: the current {field} is {new_value}, not {old_value}."
    else:
        intro = f"Please remember that my {field} is {old_value}."
        update = f"Correction: my current {field} is {new_value}, not {old_value}."

    ask_kind = rng.random()
    if ask_kind < 0.40:
        question = pick(rng, CURRENT_QUESTIONS).format(field=field)
        answer = f"The current {field} is {new_value}."
        groups = [build_fact_group(new_value, [old_value] + sample_negatives(rng, value_pool, new_value, k=5))]
    elif ask_kind < 0.70:
        question = pick(rng, OLD_QUESTIONS).format(field=field)
        answer = f"The old {field} was {old_value}."
        groups = [build_fact_group(old_value, [new_value] + sample_negatives(rng, value_pool, old_value, k=5))]
    else:
        question = pick(rng, CURRENT_OLD_QUESTIONS).format(field=field)
        answer = f"The current {field} is {new_value}, and the old one was {old_value}."
        groups = [
            build_fact_group(new_value, [old_value] + sample_negatives(rng, value_pool, new_value, k=5)),
            build_fact_group(old_value, [new_value] + sample_negatives(rng, value_pool, old_value, k=5)),
        ]

    messages = [
        {"role": "user", "content": intro},
        ack(rng),
        {"role": "user", "content": update},
        {"role": "assistant", "content": f"Got it. I'll treat {new_value} as current and {old_value} as old."},
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer},
    ]
    validate(messages)
    return pack(messages, "curriculum_current_old_binding", groups)


def make_multi_update_binding(rng):
    field = pick(rng, ["dog name", "branch", "database", "plan"])
    value_pool = PET_NAME_POOL if "name" in field else BRANCHES if field == "branch" else DATABASES if field == "database" else PLAN_NAMES
    first = pick(rng, value_pool)
    second = pick_other(rng, value_pool, first)
    third = pick_other(rng, value_pool, second)
    while third == first:
        third = pick_other(rng, value_pool, second)
    prefix = "my " if "name" in field else "the "
    messages = [
        {"role": "user", "content": f"Remember that {prefix}{field} starts as {first}."},
        ack(rng),
        {"role": "user", "content": f"Update {prefix}{field} to {second}; {first} is old now."},
        {"role": "assistant", "content": f"Got it. {second} is current and {first} is old."},
        {"role": "user", "content": f"Final correction: {prefix}{field} is {third}, not {second}."},
        {"role": "assistant", "content": f"Understood. The current {field} is {third}; {second} is no longer current."},
    ]
    ask = rng.random()
    if ask < 0.45:
        question = f"What is the current {field}?"
        answer = f"The current {field} is {third}."
        groups = [build_fact_group(third, [first, second] + sample_negatives(rng, value_pool, third, k=4))]
    elif ask < 0.75:
        question = f"Which {field} was immediately replaced by the final correction?"
        answer = f"{second} was replaced by {third}."
        groups = [
            build_fact_group(second, [first, third] + sample_negatives(rng, value_pool, second, k=4)),
            build_fact_group(third, [first, second] + sample_negatives(rng, value_pool, third, k=4)),
        ]
    else:
        question = f"Tell me the current {field} and the previous value."
        answer = f"The current {field} is {third}, and the previous value was {second}."
        groups = [
            build_fact_group(third, [first, second] + sample_negatives(rng, value_pool, third, k=4)),
            build_fact_group(second, [first, third] + sample_negatives(rng, value_pool, second, k=4)),
        ]
    messages.extend([{"role": "user", "content": question}, {"role": "assistant", "content": answer}])
    validate(messages)
    return pack(messages, "curriculum_multi_update_binding", groups)


def make_structured_fields(rng):
    repo = pick(rng, REPOS)
    region = pick(rng, REGIONS)
    branch = pick(rng, BRANCHES)
    cluster = pick(rng, CLUSTERS)
    python_version = pick(rng, PYTHON_VERSIONS)
    facts = {
        "repo": repo,
        "region": region,
        "branch": branch,
        "cluster": cluster,
        "python": python_version,
    }
    messages = [
        {
            "role": "user",
            "content": (
                "Remember this project setup: "
                f"repo {repo}, region {region}, branch {branch}, cluster {cluster}, "
                f"and Python {python_version}."
            ),
        },
        ack(rng),
    ]
    if rng.random() < 0.35:
        old_branch = branch
        branch = pick_other(rng, BRANCHES, branch)
        facts["branch"] = branch
        messages.extend(
            [
                {"role": "user", "content": f"Update the branch from {old_branch} to {branch}."},
                {"role": "assistant", "content": f"Understood. The current branch is {branch}."},
            ]
        )

    if rng.random() < 0.55:
        field, question = pick(rng, PROJECT_FIELD_QUESTIONS)
        value = facts[field]
        answer = f"The current {field} is {value}."
        pools = {
            "repo": REPOS,
            "region": REGIONS,
            "branch": BRANCHES,
            "cluster": CLUSTERS,
            "python": PYTHON_VERSIONS,
        }
        groups = [build_fact_group(value, sample_negatives(rng, pools[field], value, k=6))]
    else:
        question = pick(rng, STRUCTURED_ALL_QUESTIONS)
        answer = (
            f"Use repo {repo}, region {region}, branch {branch}, "
            f"cluster {cluster}, and Python {python_version}."
        )
        groups = [
            build_fact_group(repo, sample_negatives(rng, REPOS, repo, k=5)),
            build_fact_group(region, sample_negatives(rng, REGIONS, region, k=5)),
            build_fact_group(branch, sample_negatives(rng, BRANCHES, branch, k=5)),
            build_fact_group(cluster, sample_negatives(rng, CLUSTERS, cluster, k=5)),
            build_fact_group(python_version, sample_negatives(rng, PYTHON_VERSIONS, python_version, k=3)),
        ]

    messages.extend([{"role": "user", "content": question}, {"role": "assistant", "content": answer}])
    validate(messages)
    return pack(messages, "curriculum_structured_fields", groups)


def project_facts(rng):
    repo = pick(rng, REPOS)
    region = pick(rng, REGIONS)
    branch = pick(rng, BRANCHES)
    cluster = pick(rng, CLUSTERS)
    python_version = pick(rng, PYTHON_VERSIONS)
    return {
        "repo": repo,
        "region": region,
        "branch": branch,
        "cluster": cluster,
        "python": python_version,
    }


def project_intro(facts):
    return (
        "Remember this project setup: "
        f"repo {facts['repo']}, region {facts['region']}, branch {facts['branch']}, "
        f"cluster {facts['cluster']}, and Python {facts['python']}."
    )


def field_pool(field):
    return {
        "repo": REPOS,
        "region": REGIONS,
        "branch": BRANCHES,
        "cluster": CLUSTERS,
        "python": PYTHON_VERSIONS,
    }[field]


def make_structured_single_field(rng):
    facts = project_facts(rng)
    field, question = pick(rng, PROJECT_FIELD_QUESTIONS)
    value = facts[field]
    answer = pick(
        rng,
        [
            f"{value}.",
            f"The current {field} is {value}.",
            f"Use {value} for {field}.",
        ],
    )
    messages = [
        {"role": "user", "content": project_intro(facts)},
        ack(rng),
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer},
    ]
    validate(messages)
    return pack(
        messages,
        "curriculum_structured_single_field",
        [build_fact_group(value, sample_negatives(rng, field_pool(field), value, k=6))],
    )


def make_structured_pair_fields(rng):
    facts = project_facts(rng)
    fields, question = pick(rng, PAIR_FIELD_QUESTIONS)
    first, second = fields
    answer = pick(
        rng,
        [
            f"Use {facts[first]} for {first} and {facts[second]} for {second}.",
            f"{first}: {facts[first]}; {second}: {facts[second]}.",
            f"The {first} is {facts[first]}, and the {second} is {facts[second]}.",
        ],
    )
    messages = [
        {"role": "user", "content": project_intro(facts)},
        ack(rng),
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer},
    ]
    validate(messages)
    groups = [
        build_fact_group(facts[first], sample_negatives(rng, field_pool(first), facts[first], k=5)),
        build_fact_group(facts[second], sample_negatives(rng, field_pool(second), facts[second], k=5)),
    ]
    return pack(messages, "curriculum_structured_pair_fields", groups)


def make_customer_or_tool_binding(rng):
    if rng.random() < 0.5:
        a = pick(rng, CUSTOMER_NAMES)
        b = pick_other(rng, CUSTOMER_NAMES, a)
        plan_a = pick(rng, PLAN_NAMES)
        plan_b = pick_other(rng, PLAN_NAMES, plan_a)
        target = pick(rng, [a, b])
        plan = plan_a if target == a else plan_b
        other_plan = plan_b if target == a else plan_a
        messages = [
            {"role": "user", "content": f"Remember: {a} is on {plan_a}; {b} is on {plan_b}."},
            ack(rng),
            {"role": "user", "content": f"What plan goes with {target}?"},
            {"role": "assistant", "content": f"{target} is on {plan}."},
        ]
        groups = [build_fact_group(plan, [other_plan] + sample_negatives(rng, PLAN_NAMES, plan, k=4))]
        family = "curriculum_customer_binding"
    else:
        shell = pick(rng, SHELLS)
        editor = pick(rng, EDITORS)
        database = pick(rng, DATABASES)
        answer_format = pick(rng, FORMATS)
        question = pick(rng, [
            "What editor should you remember for me?",
            "Which database do I prefer?",
            "What shell do I use?",
            "How should you format answers?",
        ])
        if "editor" in question:
            value, field, pool = editor, "editor", EDITORS
        elif "database" in question:
            value, field, pool = database, "database", DATABASES
        elif "shell" in question:
            value, field, pool = shell, "shell", SHELLS
        else:
            value, field, pool = answer_format, "answer format", FORMATS
        messages = [
            {
                "role": "user",
                "content": (
                    f"Remember my setup: shell {shell}, editor {editor}, "
                    f"database {database}, and answer format {answer_format}."
                ),
            },
            ack(rng),
            {"role": "user", "content": question},
            {"role": "assistant", "content": f"Your {field} is {value}."},
        ]
        groups = [build_fact_group(value, sample_negatives(rng, pool, value, k=5))]
        family = "curriculum_tool_binding"
    validate(messages)
    return pack(messages, family, groups)


def make_memory_reasoning_response(rng):
    kind = pick(rng, ["dog", "cat", "rabbit"])
    name = pick(rng, PET_NAME_POOL)
    if rng.random() < 0.30:
        name, hard_negatives = pick(rng, HARD_NAME_PAIRS)
    else:
        hard_negatives = sample_negatives(rng, PET_NAME_POOL, name, k=6)
    task = pick(
        rng,
        [
            "Write a two-item packing checklist for taking my {kind} on a short walk.",
            "Draft one friendly sentence I could send to the vet about my {kind}.",
            "Give me a short reminder for tomorrow morning that uses my {kind}'s name.",
            "Suggest a calm one-sentence caption about my {kind}.",
        ],
    )
    question = task.format(kind=kind)
    if "packing checklist" in task:
        answer = f"For {name}: bring water, and pack a small snack."
    elif "vet" in task:
        answer = f"Hi, I am checking in about {name}'s appointment and wanted to confirm the time."
    elif "tomorrow morning" in task:
        answer = f"Reminder: check on {name} before starting the morning."
    else:
        answer = f"{name} is taking the day one small adventure at a time."
    messages = [
        {"role": "user", "content": pick(rng, IDENTITY_FACT_PROMPTS).format(kind=kind, name=name)},
        ack(rng),
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer},
    ]
    validate(messages)
    return pack(
        messages,
        "curriculum_memory_reasoning_response",
        [build_fact_group(name, list(dict.fromkeys(hard_negatives)))],
    )


def make_contextual_drag_reset_or_reuse(rng):
    kind = pick(rng, ["dog", "cat", "rabbit"])
    correct_name = pick(rng, PET_NAME_POOL)
    wrong_name = pick_other(rng, PET_NAME_POOL, correct_name)
    if rng.random() < 0.45:
        correct_name, hard_negatives = pick(rng, HARD_NAME_PAIRS)
        wrong_name = pick(rng, hard_negatives)
    else:
        hard_negatives = [wrong_name] + sample_negatives(rng, PET_NAME_POOL, correct_name, k=5)

    draft_is_correct = rng.random() < 0.45
    draft_name = correct_name if draft_is_correct else wrong_name
    draft_sentence = pick(
        rng,
        [
            f"Draft answer: your {kind}'s name is {draft_name}.",
            f"My draft says the {kind} name is {draft_name}.",
            f"Possible answer: {draft_name}.",
        ],
    )
    if draft_is_correct:
        challenge = pick(
            rng,
            [
                "Check that draft against memory. If it is correct, reuse it and answer final.",
                "Verify the draft and give the final answer.",
                "Is that draft consistent with what I told you? Answer final.",
            ],
        )
        final = pick(
            rng,
            [
                f"Correct. Your {kind}'s name is {correct_name}.",
                f"The draft is correct: {correct_name}.",
                f"Yes. Your {kind}'s name is {correct_name}.",
            ],
        )
        family = "curriculum_contextual_drag_reuse"
    else:
        challenge = pick(
            rng,
            [
                "That draft is wrong. Ignore it, check memory again, and answer final.",
                "The draft used the wrong name. Reset to the remembered value and answer.",
                "Do not continue from that draft. Use memory and give the correct final answer.",
            ],
        )
        final = pick(
            rng,
            [
                f"Resetting to memory: your {kind}'s name is {correct_name}.",
                f"The draft was wrong. Your {kind}'s name is {correct_name}.",
                f"The correct remembered name is {correct_name}.",
            ],
        )
        family = "curriculum_contextual_drag_reset"

    messages = [
        {"role": "user", "content": pick(rng, IDENTITY_FACT_PROMPTS).format(kind=kind, name=correct_name)},
        ack(rng),
        {"role": "user", "content": f"Now draft an answer to: what is my {kind}'s name?"},
        {"role": "assistant", "content": draft_sentence},
        {"role": "user", "content": challenge},
        {"role": "assistant", "content": final},
    ]
    validate(messages)
    return pack(
        messages,
        family,
        [build_fact_group(correct_name, list(dict.fromkeys([wrong_name] + hard_negatives)))],
    )


def make_heuristic_override_memory(rng):
    case = pick(rng, ["pet_default", "branch_default", "database_default", "plan_default"])
    if case == "pet_default":
        kind = pick(rng, ["dog", "cat"])
        correct = pick(rng, PET_NAME_POOL)
        common_pool = [name for name in HEURISTIC_DEFAULTS[kind] if name != correct] or HEURISTIC_DEFAULTS[kind]
        common = pick(rng, common_pool)
        messages = [
            {"role": "user", "content": pick(rng, IDENTITY_FACT_PROMPTS).format(kind=kind, name=correct)},
            ack(rng),
            {
                "role": "user",
                "content": (
                    f"Many examples use {common} as a default {kind} name, but this is about my remembered {kind}. "
                    f"What is my {kind}'s name?"
                ),
            },
            {"role": "assistant", "content": f"Your {kind}'s name is {correct}."},
        ]
        pool = PET_NAME_POOL + HEURISTIC_DEFAULTS[kind]
        groups = [build_fact_group(correct, [common] + sample_negatives(rng, pool, correct, k=5))]
    elif case == "branch_default":
        correct = pick(rng, BRANCHES)
        common = pick_other(rng, HEURISTIC_DEFAULTS["branch"] + BRANCHES, correct)
        messages = [
            {"role": "user", "content": f"Remember the current branch is {correct}."},
            ack(rng),
            {
                "role": "user",
                "content": f"People often assume the branch is {common}. Override that default with memory: which branch is current?",
            },
            {"role": "assistant", "content": f"The current branch is {correct}."},
        ]
        groups = [build_fact_group(correct, [common] + sample_negatives(rng, BRANCHES + HEURISTIC_DEFAULTS["branch"], correct, k=5))]
    elif case == "database_default":
        correct = pick(rng, DATABASES)
        common = pick_other(rng, HEURISTIC_DEFAULTS["database"] + DATABASES, correct)
        messages = [
            {"role": "user", "content": f"Remember that my current database is {correct}."},
            ack(rng),
            {
                "role": "user",
                "content": f"If you guessed from habit you might say {common}. Use the remembered value instead. Which database is current?",
            },
            {"role": "assistant", "content": f"The current database is {correct}."},
        ]
        groups = [build_fact_group(correct, [common] + sample_negatives(rng, DATABASES + HEURISTIC_DEFAULTS["database"], correct, k=5))]
    else:
        correct = pick(rng, PLAN_NAMES)
        common = pick_other(rng, HEURISTIC_DEFAULTS["plan"] + PLAN_NAMES, correct)
        customer = pick(rng, CUSTOMER_NAMES)
        messages = [
            {"role": "user", "content": f"Remember that {customer} is on the {correct} plan."},
            ack(rng),
            {
                "role": "user",
                "content": f"Do not use the usual default plan {common}; use memory. What plan goes with {customer}?",
            },
            {"role": "assistant", "content": f"{customer} is on the {correct} plan."},
        ]
        groups = [build_fact_group(correct, [common] + sample_negatives(rng, PLAN_NAMES + HEURISTIC_DEFAULTS["plan"], correct, k=5))]
    validate(messages)
    return pack(messages, "curriculum_heuristic_override_memory", groups)


def make_heuristic_override_irrelevant(rng):
    kind = pick(rng, ["dog", "cat"])
    remembered = pick(rng, PET_NAME_POOL)
    common_pool = [name for name in HEURISTIC_DEFAULTS[kind] if name != remembered] or HEURISTIC_DEFAULTS[kind]
    common = pick(rng, common_pool)
    question = pick(
        rng,
        [
            f"What is a common {kind} name people often use in examples? This is not asking about my remembered {kind}.",
            f"If memory is irrelevant, name one common {kind} name from ordinary examples.",
            f"Answer the general question, not my personal memory: give one common {kind} name.",
        ],
    )
    messages = [
        {"role": "user", "content": pick(rng, IDENTITY_FACT_PROMPTS).format(kind=kind, name=remembered)},
        ack(rng),
        {"role": "user", "content": question},
        {"role": "assistant", "content": f"{common} is a common {kind} name."},
    ]
    validate(messages)
    return guardrail_pack(messages, "curriculum_heuristic_override_irrelevant")


def make_guardrail_with_irrelevant_memory(rng):
    question, answer = pick(rng, ORDINARY_CHAT_ROWS)
    dog = pick(rng, PET_NAME_POOL)
    city = pick(rng, CITIES)
    repo = pick(rng, REPOS)
    branch = pick(rng, BRANCHES)
    messages = [
        {"role": "user", "content": f"For later, remember that my dog's name is {dog}."},
        ack(rng),
        {"role": "user", "content": f"Also remember that I live in {city}."},
        ack(rng),
    ]
    if rng.random() < 0.45:
        messages.extend(
            [
                {"role": "user", "content": f"Keep this project note too: repo {repo}, branch {branch}."},
                ack(rng),
            ]
        )
    messages.extend(
        [
            {"role": "user", "content": f"{question} Do not bring up remembered personal details unless they are relevant."},
            {"role": "assistant", "content": answer},
        ]
    )
    validate(messages)
    return guardrail_pack(messages, "curriculum_irrelevant_memory_guardrail")


def make_language_guardrail(rng):
    question, answer = pick(rng, ORDINARY_CHAT_ROWS)
    messages = []
    if rng.random() < 0.55:
        name = pick(rng, PET_NAME_POOL)
        messages.extend(
            [
                {"role": "user", "content": f"For later, remember that my dog's name is {name}."},
                ack(rng),
            ]
        )
        if rng.random() < 0.35:
            city = pick(rng, CITIES)
            messages.extend(
                [
                    {"role": "user", "content": f"Also remember that I live in {city}."},
                    ack(rng),
                ]
            )
    else:
        messages.extend(
            [
                {"role": "user", "content": "Before the real question, please keep the answer concise."},
                {"role": "assistant", "content": "Sure, I will keep it concise."},
            ]
        )
    messages.extend([{"role": "user", "content": question}, {"role": "assistant", "content": answer}])
    validate(messages)
    return guardrail_pack(messages)


def build_rows(rng, n, builders):
    rows = []
    for _ in range(n):
        builder = pick(rng, builders)
        rows.append(builder(rng))
    return rows


def main():
    global CURRICULUM_EVAL_MODE

    parser = argparse.ArgumentParser(description="Prepare balanced memory-curriculum JSONL shards")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--seed", type=int, default=20260426)
    parser.add_argument("--identity-size", type=int, default=20000)
    parser.add_argument("--live-session-size", type=int, default=60000)
    parser.add_argument("--answer-size", type=int, default=18000)
    parser.add_argument("--binding-size", type=int, default=20000)
    parser.add_argument("--structured-size", type=int, default=20000)
    parser.add_argument("--structured-single-size", type=int, default=12000)
    parser.add_argument("--structured-pair-size", type=int, default=12000)
    parser.add_argument("--reasoning-size", type=int, default=12000)
    parser.add_argument("--drag-size", type=int, default=12000)
    parser.add_argument("--heuristic-size", type=int, default=12000)
    parser.add_argument("--guardrail-size", type=int, default=14000)
    parser.add_argument("--distractor-guardrail-size", type=int, default=14000)
    parser.add_argument("--v4-exact-copy-size", type=int, default=0)
    parser.add_argument("--v4-branchpoint-copy-size", type=int, default=0)
    parser.add_argument("--v4-no-memory-size", type=int, default=0)
    parser.add_argument("--v4-normal-chat-size", type=int, default=0)
    parser.add_argument("--v4-irrelevant-memory-chat-size", type=int, default=0)
    parser.add_argument("--v4-failure-replay-size", type=int, default=0)
    parser.add_argument("--v5-arbitrary-char-size", type=int, default=0)
    parser.add_argument("--v5-arbitrary-token-size", type=int, default=0)
    parser.add_argument("--v5-memory-variety-size", type=int, default=0)
    parser.add_argument("--v5-no-memory-size", type=int, default=0)
    parser.add_argument("--v5-irrelevant-memory-size", type=int, default=0)
    parser.add_argument("--v5-random-key-binding-size", type=int, default=0)
    parser.add_argument("--eval-size", type=int, default=512)
    parser.add_argument("--v4-eval-size", type=int, default=1024)
    parser.add_argument("--v5-eval-size", type=int, default=2048)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    rng = random.Random(args.seed)
    files = []

    specs = [
        ("curriculum_live_session_names.jsonl", args.live_session_size, [make_live_session_name_recall]),
        ("curriculum_identity_varied.jsonl", args.identity_size, [make_identity_varied, make_stop_after_fact, make_hard_prefix_identity]),
        ("curriculum_answer_realization.jsonl", args.answer_size, [make_answer_realization, make_memory_reasoning_response, make_hard_prefix_identity]),
        ("curriculum_binding_current.jsonl", args.binding_size, [make_current_old_binding, make_multi_update_binding, make_customer_or_tool_binding]),
        ("curriculum_structured_fields.jsonl", args.structured_size, [make_structured_fields, make_customer_or_tool_binding]),
        ("curriculum_structured_single_field.jsonl", args.structured_single_size, [make_structured_single_field]),
        ("curriculum_structured_pair_fields.jsonl", args.structured_pair_size, [make_structured_pair_fields]),
        ("curriculum_memory_reasoning.jsonl", args.reasoning_size, [make_memory_reasoning_response]),
        ("curriculum_contextual_drag.jsonl", args.drag_size, [make_contextual_drag_reset_or_reuse]),
        ("curriculum_heuristic_override.jsonl", args.heuristic_size, [make_heuristic_override_memory, make_heuristic_override_irrelevant]),
        ("curriculum_language_guardrail.jsonl", args.guardrail_size, [make_language_guardrail]),
        ("curriculum_irrelevant_memory_guardrail.jsonl", args.distractor_guardrail_size, [make_guardrail_with_irrelevant_memory]),
    ]
    if args.v4_exact_copy_size > 0:
        specs.append(("curriculum_v4_exact_copy.jsonl", args.v4_exact_copy_size, [make_v4_exact_copy]))
    if args.v4_branchpoint_copy_size > 0:
        specs.append(("curriculum_v4_branchpoint_copy.jsonl", args.v4_branchpoint_copy_size, [make_v4_branchpoint_copy]))
    if args.v4_no_memory_size > 0:
        specs.append(("curriculum_v4_no_memory_deference.jsonl", args.v4_no_memory_size, [make_v4_no_memory_deference]))
    if args.v4_normal_chat_size > 0:
        specs.append(("curriculum_v4_normal_chat_guardrail.jsonl", args.v4_normal_chat_size, [make_v4_normal_chat_guardrail]))
    if args.v4_irrelevant_memory_chat_size > 0:
        specs.append(
            (
                "curriculum_v4_irrelevant_memory_chat_guardrail.jsonl",
                args.v4_irrelevant_memory_chat_size,
                [make_v4_irrelevant_memory_chat_guardrail],
            )
        )
    if args.v4_failure_replay_size > 0:
        specs.append(("curriculum_v4_failure_replay.jsonl", args.v4_failure_replay_size, [make_v4_failure_replay]))
    if args.v5_arbitrary_char_size > 0:
        specs.append(("curriculum_v5_arbitrary_char_copy.jsonl", args.v5_arbitrary_char_size, [make_v5_arbitrary_char_copy]))
    if args.v5_arbitrary_token_size > 0:
        specs.append(("curriculum_v5_arbitrary_token_copy.jsonl", args.v5_arbitrary_token_size, [make_v5_arbitrary_token_copy]))
    if args.v5_memory_variety_size > 0:
        specs.append(("curriculum_v5_memory_variety.jsonl", args.v5_memory_variety_size, [make_v5_memory_variety]))
    if args.v5_no_memory_size > 0:
        specs.append(("curriculum_v5_no_memory_guardrail.jsonl", args.v5_no_memory_size, [make_v5_no_memory_guardrail]))
    if args.v5_irrelevant_memory_size > 0:
        specs.append(("curriculum_v5_irrelevant_memory_guardrail.jsonl", args.v5_irrelevant_memory_size, [make_v5_irrelevant_memory_guardrail]))
    if args.v5_random_key_binding_size > 0:
        specs.append(("curriculum_v5_random_key_binding.jsonl", args.v5_random_key_binding_size, [make_v5_random_key_binding]))
    for filename, size, builders in specs:
        rows = build_rows(rng, size, builders)
        path = output_dir / filename
        write_jsonl(path, rows)
        valid_rows = sum(1 for row in rows if has_trainable_final_pair(row))
        files.append(
            {
                "name": filename,
                "path": str(path),
                "num_examples": len(rows),
                "trainable_examples": valid_rows,
                "skipped_by_chat_memory": len(rows) - valid_rows,
            }
        )
        print(f"Wrote {filename}: {len(rows):,} rows, {valid_rows:,} trainable by chat_memory")

    CURRICULUM_EVAL_MODE = True

    eval_specs = [
        ("curriculum_eval_live_session_names.jsonl", [make_live_session_name_recall]),
        ("curriculum_eval_identity.jsonl", [make_identity_varied, make_stop_after_fact, make_hard_prefix_identity]),
        ("curriculum_eval_answer_realization.jsonl", [make_answer_realization, make_memory_reasoning_response]),
        ("curriculum_eval_binding.jsonl", [make_current_old_binding, make_multi_update_binding, make_customer_or_tool_binding]),
        ("curriculum_eval_structured.jsonl", [make_structured_fields, make_structured_single_field, make_structured_pair_fields]),
        ("curriculum_eval_guardrail.jsonl", [make_language_guardrail, make_guardrail_with_irrelevant_memory]),
        ("curriculum_eval_contextual_drag.jsonl", [make_contextual_drag_reset_or_reuse]),
        ("curriculum_eval_heuristic_override.jsonl", [make_heuristic_override_memory, make_heuristic_override_irrelevant]),
    ]
    for filename, builders in eval_specs:
        rows = build_rows(rng, args.eval_size, builders)
        path = output_dir / filename
        write_jsonl(path, rows)
        valid_rows = sum(1 for row in rows if has_trainable_final_pair(row))
        files.append(
            {
                "name": filename,
                "path": str(path),
                "num_examples": len(rows),
                "trainable_examples": valid_rows,
                "skipped_by_chat_memory": len(rows) - valid_rows,
                "eval": True,
            }
        )
        print(f"Wrote {filename}: {len(rows):,} rows, {valid_rows:,} trainable by chat_memory")

    v4_eval_specs = [
        ("curriculum_eval_v4_exact_copy.jsonl", [make_v4_exact_copy]),
        ("curriculum_eval_v4_branchpoint_copy.jsonl", [make_v4_branchpoint_copy]),
        ("curriculum_eval_v4_no_memory_deference.jsonl", [make_v4_no_memory_deference]),
        ("curriculum_eval_v4_normal_chat_guardrail.jsonl", [make_v4_normal_chat_guardrail]),
        ("curriculum_eval_v4_irrelevant_memory_chat_guardrail.jsonl", [make_v4_irrelevant_memory_chat_guardrail]),
        ("curriculum_eval_v4_failure_replay.jsonl", [make_v4_failure_replay]),
    ]
    if (
        args.v4_exact_copy_size > 0
        or args.v4_branchpoint_copy_size > 0
        or args.v4_no_memory_size > 0
        or args.v4_normal_chat_size > 0
        or args.v4_irrelevant_memory_chat_size > 0
        or args.v4_failure_replay_size > 0
    ):
        for filename, builders in v4_eval_specs:
            rows = build_rows(rng, args.v4_eval_size, builders)
            path = output_dir / filename
            write_jsonl(path, rows)
            valid_rows = sum(1 for row in rows if has_trainable_final_pair(row))
            files.append(
                {
                    "name": filename,
                    "path": str(path),
                    "num_examples": len(rows),
                    "trainable_examples": valid_rows,
                    "skipped_by_chat_memory": len(rows) - valid_rows,
                    "eval": True,
                }
            )
            print(f"Wrote {filename}: {len(rows):,} rows, {valid_rows:,} trainable by chat_memory")

    v5_eval_specs = [
        ("curriculum_eval_v5_arbitrary_char_copy.jsonl", [make_v5_arbitrary_char_copy]),
        ("curriculum_eval_v5_arbitrary_token_copy.jsonl", [make_v5_arbitrary_token_copy]),
        ("curriculum_eval_v5_memory_variety.jsonl", [make_v5_memory_variety]),
        ("curriculum_eval_v5_no_memory_guardrail.jsonl", [make_v5_no_memory_guardrail]),
        ("curriculum_eval_v5_irrelevant_memory_guardrail.jsonl", [make_v5_irrelevant_memory_guardrail]),
        ("curriculum_eval_v5_random_key_binding.jsonl", [make_v5_random_key_binding]),
    ]
    if (
        args.v5_arbitrary_char_size > 0
        or args.v5_arbitrary_token_size > 0
        or args.v5_memory_variety_size > 0
        or args.v5_no_memory_size > 0
        or args.v5_irrelevant_memory_size > 0
        or args.v5_random_key_binding_size > 0
    ):
        for filename, builders in v5_eval_specs:
            rows = build_rows(rng, args.v5_eval_size, builders)
            path = output_dir / filename
            write_jsonl(path, rows)
            valid_rows = sum(1 for row in rows if has_trainable_final_pair(row))
            files.append(
                {
                    "name": filename,
                    "path": str(path),
                    "num_examples": len(rows),
                    "trainable_examples": valid_rows,
                    "skipped_by_chat_memory": len(rows) - valid_rows,
                    "eval": True,
                }
            )
            print(f"Wrote {filename}: {len(rows):,} rows, {valid_rows:,} trainable by chat_memory")

    CURRICULUM_EVAL_MODE = False

    manifest = {
        "seed": args.seed,
        "output_dir": str(output_dir),
        "files": files,
        "notes": [
            "Guardrail rows use memory_target.mode=guardrail_only, so they preserve normal chat without fake fact losses.",
            "Identity rows vary answer shape and include stop-after-fact examples to reduce repeated fact rambling.",
            "Answer-realization rows force the remembered datum to be used inside normal, varied responses instead of only in recall templates.",
            "Binding rows emphasize current-vs-old and same-field confusers.",
            "Structured rows are split into single-field, pair-field, and all-field variants so field binding can be learned before full bundles.",
            "Irrelevant-memory guardrails write memories first, then ask normal questions, so memory learns not to hijack unrelated chat.",
            "Contextual-drag rows balance incorrect-draft reset with correct-draft reuse, avoiding a one-sided reset habit.",
            "Heuristic-override rows create minimal conflicts between remembered constraints and common/default answers, plus irrelevant-memory controls.",
            f"The generated pet-name pool contains {len(PET_NAME_POOL):,} distinct names, so live-session recall cannot solve the task by memorizing a short closed list.",
            "Live-session name rows mirror deployed usage: a user-only fact write followed by a fresh read-only recall question.",
            f"V4 exact-copy rows use a separate pool of {len(V4_NAME_POOL):,} common, weird, short, hyphenated, and pseudo-word values.",
            "V4 exact-copy/branchpoint rows train every subtoken of the remembered value through normal teacher-forced CE plus hard-negative span metadata.",
            "V4 no-memory rows make absent-memory questions answer with deference instead of fluent invented defaults.",
            "V4 normal-chat rows are multi-turn guardrail-only examples, so chat_memory consumes them instead of silently skipping two-message rows.",
            "V4 failure-replay rows target observed errors: Luna/Polly defaults, Buh/Bist drift, Taloobrook smoothing, and old/current reversals.",
            "V5 arbitrary-character rows train exact copying for 1-15 character strings from [a-zA-Z0-9 -_.], including short values that the base model often smooths into familiar words.",
            "V5 arbitrary-token rows train the same operation over tokenizer-native spans bucketed by token length, so the model learns retrieval/copy as an operation rather than a name-list lookup.",
            "V5 memory-variety rows cover pets, projects, codes, branches, devices, contacts, notes, aliases, regions, and incident ids with varied ask/answer shapes.",
            "V5 no-memory and irrelevant-memory guardrails make memory arbitration explicit: use memory when relevant, say missing when absent, and keep normal chat normal when saved values are unrelated.",
            "V5 random-key binding rows use high-entropy keys and values to train the lookup contract: copy only for exact stored keys, and reject absent, near-miss, or value-as-key queries.",
            f"Ordinary-chat guardrails sample from {len(ORDINARY_CHAT_ROWS):,} base prompt/answer pairs before V4 prefix expansion.",
            f"V4 normal-chat rows sample from {len(V4_NORMAL_CHAT_ROWS):,} distinct prompt/answer variants to preserve ordinary assistant behavior without any memory requirement.",
            f"V4 irrelevant-memory chat rows sample from {len(V4_MEMORY_IRRELEVANT_ROWS):,} prompt/answer variants, then multiply them by real remembered names/codes/cities so memory learns not to hijack normal conversation.",
            "Heuristic-default override pools contain "
            + ", ".join(f"{field}={len(values):,}" for field, values in sorted(HEURISTIC_DEFAULTS.items()))
            + " confusers, avoiding tiny closed-list defaults.",
        ],
    }
    manifest_path = output_dir / "memory_curriculum_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Wrote memory curriculum data to {output_dir}")
    for item in files:
        print(f"- {item['name']}: {item['num_examples']}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
