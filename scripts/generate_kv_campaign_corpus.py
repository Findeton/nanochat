"""
Generate larger, richer staged datasets for K/V-memory training.

The goal is not only to scale the corpus, but to improve the structure of the
memory tasks themselves. The generated conversations deliberately cover:

- exact identity recall with templated answers
- paraphrased recall prompts
- overwrite / update behavior
- distractor resistance
- role/filler binding across similar entities
- multi-fact structured answers
- intermediate recall traces ("remembering remembering")
- broader work/project/service/customer/schedule memory

The generator writes:

- kv_stage1_large_train.jsonl
- kv_stage1_large_eval.jsonl
- kv_stage2_large_train.jsonl
- kv_stage2_large_eval.jsonl
- kv_stage3_large_train.jsonl
- kv_stage3_large_eval.jsonl
- kv_large_manifest.json
"""

import argparse
import json
import random
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT_DIR / "data" / "kv_generated"


FIRST_NAMES = [
    "Alicia", "Ben", "Camila", "Daniel", "Elena", "Felix", "Grace", "Hugo",
    "Ivy", "Jonah", "Kira", "Liam", "Marta", "Noah", "Olivia", "Priya",
    "Quinn", "Rafael", "Sara", "Theo", "Uma", "Vera", "Will", "Yasmin",
]
LAST_NAMES = [
    "Abbott", "Bennett", "Carter", "Diaz", "Ellis", "Flores", "Garcia", "Hayes",
    "Ibarra", "Jensen", "Khan", "Lopez", "Mendez", "Nash", "Ortega", "Patel",
    "Quintero", "Reed", "Santos", "Turner", "Usher", "Vargas", "Walker", "Young",
]
PET_NAMES = [
    "Tuna", "Maple", "Pebble-Cloud", "Aster-Pine", "Lumen", "Mochi", "Nori", "Hazel",
    "Poppy", "Ruby", "Pixel", "Olive", "Clover", "Miso", "Biscuit", "Pepper",
    "Saffron", "Topaz", "Cinder", "Willow", "Merlin", "Juniper", "Orchid", "Marble",
    "Harbor", "Ember", "Fable", "Rook", "Comet", "Drift",
]
PET_TYPES = ["dog", "cat", "rabbit", "beagle", "corgi", "parrot"]
CITIES = [
    "Austin", "Bogota", "Porto", "Medellin", "Lisbon", "Quito", "Seoul", "Kyoto",
    "Denver", "Tallinn", "Valencia", "Lima", "Bristol", "Curitiba", "Oslo", "Lyon",
    "Miami", "Atlanta", "Portland", "Monterrey", "Vienna", "Naples", "Prague", "Recife",
]
COLORS = [
    "green", "blue", "teal", "orange", "white", "navy", "coral", "indigo",
    "maroon", "olive", "gold", "black", "gray", "cyan",
]
DRINKS = [
    "matcha", "coffee", "tea", "sparkling water", "cold brew", "chai",
    "lemonade", "yerba mate", "mineral water", "oolong tea",
]
BREAKFASTS = [
    "eggs on toast", "yogurt with fruit", "oatmeal", "arepas", "avocado toast",
    "granola", "fruit and toast", "scrambled eggs", "chia pudding",
]
DATABASES = ["Postgres", "DuckDB", "SQLite", "MySQL", "ClickHouse", "Redis"]
SHELLS = ["zsh", "bash", "fish"]
EDITORS = ["Neovim", "VS Code", "Helix", "Emacs"]
OSES = ["macOS", "Linux", "Ubuntu", "Debian", "Fedora"]
PYTHON_VERSIONS = ["3.10", "3.11", "3.12"]
REPOS = [
    "billing-api", "identity-console", "search-worker", "chat-gateway",
    "client-router", "analytics-sync", "gitops-deployer", "auth-service",
    "cohort-planner", "release-orchestrator", "feature-flags", "tenant-importer",
]
REGIONS = ["prod1-euw1", "prod1-use1", "staging-euw2", "qa-use2", "dev-usw2", "lab-sae1"]
CLUSTERS = ["atlas", "boron", "cedar", "delta", "fjord", "glacier", "helios", "ion"]
NAMESPACES = ["team-aurora", "team-cedar", "team-harbor", "team-ember", "team-summit", "team-vista"]
CHANNELS = [
    "#shipping-room", "#ops-europe", "#incident-triage", "#client-rollout",
    "#platform-support", "#finops", "#auth-oncall", "#release-watch",
]
PLAN_NAMES = ["Growth", "Business", "Enterprise", "Premier"]
CUSTOMER_NAMES = [
    "Acorn Health", "Northstar Labs", "Blue Mesa", "Lumen Retail",
    "Cinder Foods", "Harbor Transit", "Juniper Schools", "Atlas Solar",
]
ROOMS = ["Rio 4", "Andes 2", "Maple 7", "Cedar 3", "Harbor 5", "Atlas 1"]
WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
TIMES = ["08:30", "09:15", "10:45", "13:00", "14:20", "16:10", "17:35"]
FORMATS = ["short paragraphs", "compact prose", "brief bullet lists", "step-by-step instructions"]
TONES = ["calm", "direct", "friendly", "concise"]
BRANCHES = [
    "feature/fast-retry", "feature/memory-bank", "feature/client-refresh",
    "hotfix/auth-timeout", "release/2026-04", "feature/safe-migrations",
]
DATES = ["September 18", "October 4", "November 12", "June 9", "July 21", "August 30"]
SERVICE_COMPONENTS = ["gateway", "worker", "sync", "router", "notifier", "registry", "api"]
PORT_BASES = [7000, 7200, 7400, 7600, 7800]

INTRO_FACTS = [
    "For future chats, please remember these details:",
    "Please keep the following details in memory for later:",
    "I want you to hold onto these details across sessions:",
    "Keep these notes for later follow-ups:",
]
ADD_FACTS = [
    "A couple more details to remember:",
    "Please also remember this:",
    "One more thing for your notes:",
    "Let me add two more details:",
]
UPDATE_FACTS = [
    "Small correction for your notes:",
    "Please update a couple of details:",
    "I need to revise two items you were tracking:",
]
ACKS = [
    "Got it. I'll remember that.",
    "Understood. I'll keep those details in mind.",
    "Sounds good. I'll store that for later.",
    "I have it. I'll remember those details.",
]
DISTRACTORS = [
    "Also, keep your replies concise.",
    "Please avoid long bullet lists.",
    "Use a calm tone in later replies.",
    "Try to keep answers compact unless I ask for detail.",
]

IDENTITY_QUESTION_TEMPLATES = [
    "What's my {pet_type}'s name?",
    "Can you tell me my {pet_type}'s name?",
    "Which {pet_type} name did I tell you?",
    "What {pet_type} name should you remember for me?",
]
IDENTITY_ANSWER_TEMPLATES = [
    "Your {pet_type}'s name is {pet}.",
    "The {pet_type} name you told me is {pet}.",
    "You told me your {pet_type}'s name is {pet}.",
]
PERSONAL_STRUCTURED_QUESTIONS = [
    "Without me repeating the earlier details, what is my pet's name, where do I live, and what drink do I prefer?",
    "Please remind me of my pet's name, my city, and my preferred drink.",
    "What pet name, city, and drink are you supposed to remember for me?",
]
PERSONAL_STRUCTURED_ANSWERS = [
    "Your pet's name is {pet}, you live in {city}, and you prefer {drink}.",
    "You told me your pet's name is {pet}, you live in {city}, and your preferred drink is {drink}.",
]
TOOLING_QUESTIONS = [
    "What shell and editor do I use, what database do I prefer, and how should you format answers?",
    "Remind me of my shell, editor, database, and response format.",
    "What working setup should you remember for me?",
]
TOOLING_ANSWERS = [
    "You use {shell} and {editor}, you prefer {database}, and I should answer in {answer_format}.",
    "Your setup is {shell}, {editor}, {database}, and answers in {answer_format}.",
]
PROJECT_QUESTIONS = [
    "For the current setup, what repo, region, branch, and Python version should we use?",
    "What repo, cluster, branch, and Python version are we using?",
    "Summarize the current project config I asked you to remember.",
]
PROJECT_ANSWERS = [
    "We should use the {repo} repo in {region}, the {branch} branch, the {cluster} cluster, and Python {python_version}.",
    "The current config is repo {repo}, region {region}, branch {branch}, cluster {cluster}, and Python {python_version}.",
]
SERVICE_QUESTIONS = [
    "What base URL, port, namespace, and Slack channel should we use for this service?",
    "Summarize the service runtime details I asked you to remember.",
    "Which URL, port, namespace, and channel go with this service?",
]
SERVICE_ANSWERS = [
    "We should use {url}, port {port}, namespace {namespace}, and the Slack channel {channel}.",
    "The runtime details are {url}, port {port}, namespace {namespace}, and channel {channel}.",
]
CUSTOMER_QUESTIONS = [
    "For this customer, what account ID, plan, renewal date, and customer success manager should we remember?",
    "Summarize the customer account details you are supposed to remember.",
    "What plan, renewal date, account ID, and owner go with this customer?",
]
CUSTOMER_ANSWERS = [
    "For {customer}, the account ID is {account_id}, the plan is {plan}, the renewal date is {renewal}, and the customer success manager is {csm}.",
    "{customer} is on {plan}, renews on {renewal}, has account ID {account_id}, and is owned by {csm}.",
]
SCHEDULE_QUESTIONS = [
    "When is the review, where is it, and who is hosting it?",
    "Please remind me of the meeting time, room, and host.",
    "What review schedule details are you supposed to remember?",
]
SCHEDULE_ANSWERS = [
    "The review is on {weekday} at {time_value} in {room}, and it is hosted by {host}.",
    "It is on {weekday} at {time_value} in {room}, hosted by {host}.",
]
PAIR_BINDING_QUESTIONS = [
    "Which Slack channel belongs to {service}?",
    "For {service}, what Slack channel should we use?",
    "What channel goes with {service} rather than the other service?",
]
PAIR_BINDING_ANSWERS = [
    "The Slack channel for {service} is {channel}.",
    "{service} uses the Slack channel {channel}.",
]
CUSTOMER_BINDING_QUESTIONS = [
    "Which plan belongs to {customer}?",
    "For {customer}, what plan should we remember?",
    "What plan goes with {customer} rather than the other account?",
]
CUSTOMER_BINDING_ANSWERS = [
    "{customer} is on the {plan} plan.",
    "The plan for {customer} is {plan}.",
]
DELTA_QUESTIONS = [
    "After the update, what changed and what should you remember now?",
    "What is the current value after the correction, and what was the old one?",
]


def pick(rng, values):
    return rng.choice(values)


def pick_other(rng, values, current):
    candidates = [value for value in values if value != current]
    return rng.choice(candidates)


def person(rng):
    return f"{pick(rng, FIRST_NAMES)} {pick(rng, LAST_NAMES)}"


def join_facts(rng, facts):
    if len(facts) == 1:
        return facts[0]
    if len(facts) == 2:
        return f"{facts[0]} and {facts[1]}"
    if rng.random() < 0.5:
        return "; ".join(facts[:-1]) + f"; and {facts[-1]}"
    return ", ".join(facts[:-1]) + f", and {facts[-1]}"


def make_fact_turn(rng, prompt_choices, facts):
    return {"role": "user", "content": f"{pick(rng, prompt_choices)} {join_facts(rng, facts)}."}


def ack_message(rng):
    return {"role": "assistant", "content": pick(rng, ACKS)}


def sample_negatives(rng, values, current, k=4):
    candidates = [value for value in values if value != current]
    rng.shuffle(candidates)
    return candidates[: min(k, len(candidates))]


def build_fact_group(text, hard_negatives):
    hard_negatives = [value for value in hard_negatives if value != text]
    return {
        "text": text,
        "hard_negatives": hard_negatives,
    }


def pack_example(messages, family, fact_groups):
    return {
        "messages": messages,
        "memory_target": {
            "family": family,
            "fact_groups": fact_groups,
        },
    }


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True))
            f.write("\n")


def validate_messages(messages):
    assert len(messages) >= 4
    for i, message in enumerate(messages):
        expected = "user" if i % 2 == 0 else "assistant"
        assert message["role"] == expected, f"Expected {expected} at {i}, got {message['role']}"


def build_identity_question(rng, pet_type):
    return pick(rng, IDENTITY_QUESTION_TEMPLATES).format(pet_type=pet_type)


def build_identity_answer(rng, pet_type, pet):
    return pick(rng, IDENTITY_ANSWER_TEMPLATES).format(pet_type=pet_type, pet=pet)


def maybe_add_distractor(messages, rng):
    if rng.random() < 0.55:
        messages.append({"role": "user", "content": pick(rng, DISTRACTORS)})
        messages.append({"role": "assistant", "content": "Understood. I'll keep that in mind too."})


def make_identity_template(rng):
    pet = pick(rng, PET_NAMES)
    pet_type = "dog"
    city = pick(rng, CITIES)
    color = pick(rng, COLORS)
    drink = pick(rng, DRINKS)
    messages = [
        make_fact_turn(rng, INTRO_FACTS, [f"my {pet_type}'s name is {pet}", f"I live in {city}", f"my favorite color is {color}"]),
        ack_message(rng),
        make_fact_turn(rng, ADD_FACTS, [f"my preferred drink is {drink}"]),
        ack_message(rng),
        {"role": "user", "content": build_identity_question(rng, pet_type)},
        {"role": "assistant", "content": build_identity_answer(rng, pet_type, pet)},
    ]
    validate_messages(messages)
    return pack_example(
        messages,
        "identity_template",
        [
            build_fact_group(pet, sample_negatives(rng, PET_NAMES, pet)),
        ],
    )


def make_identity_update(rng):
    pet_type = "dog"
    pet = pick(rng, PET_NAMES)
    new_pet = pick_other(rng, PET_NAMES, pet)
    city = pick(rng, CITIES)
    messages = [
        make_fact_turn(rng, INTRO_FACTS, [f"my {pet_type}'s name is {pet}", f"I live in {city}"]),
        ack_message(rng),
    ]
    maybe_add_distractor(messages, rng)
    messages.extend(
        [
            make_fact_turn(rng, UPDATE_FACTS, [f"my {pet_type}'s name is actually {new_pet}, not {pet}"]),
            {"role": "assistant", "content": f"Got it. I'll remember that your {pet_type}'s name is {new_pet} now."},
            {"role": "user", "content": build_identity_question(rng, pet_type)},
            {"role": "assistant", "content": build_identity_answer(rng, pet_type, new_pet)},
        ]
    )
    validate_messages(messages)
    return pack_example(
        messages,
        "identity_update",
        [
            build_fact_group(new_pet, [pet] + sample_negatives(rng, PET_NAMES, new_pet)),
        ],
    )


def make_identity_binding_pair(rng):
    dog_name = pick(rng, PET_NAMES)
    cat_name = pick_other(rng, PET_NAMES, dog_name)
    city = pick(rng, CITIES)
    messages = [
        make_fact_turn(rng, INTRO_FACTS, [f"my dog's name is {dog_name}", f"my cat's name is {cat_name}", f"I live in {city}"]),
        ack_message(rng),
    ]
    maybe_add_distractor(messages, rng)
    target_pet_type = pick(rng, ["dog", "cat"])
    target_name = dog_name if target_pet_type == "dog" else cat_name
    messages.extend(
        [
            {"role": "user", "content": build_identity_question(rng, target_pet_type)},
            {"role": "assistant", "content": build_identity_answer(rng, target_pet_type, target_name)},
        ]
    )
    validate_messages(messages)
    return pack_example(
        messages,
        "identity_binding_pair",
        [
            build_fact_group(target_name, [cat_name if target_name == dog_name else dog_name] + sample_negatives(rng, PET_NAMES, target_name)),
        ],
    )


def make_identity_intermediate_recall(rng):
    pet = pick(rng, PET_NAMES)
    city = pick(rng, CITIES)
    drink = pick(rng, DRINKS)
    messages = [
        make_fact_turn(rng, INTRO_FACTS, [f"my dog's name is {pet}", f"I live in {city}", f"my preferred drink is {drink}"]),
        ack_message(rng),
        {"role": "user", "content": "Before we move on, where do I live and what drink do I prefer?"},
        {"role": "assistant", "content": f"You live in {city} and you prefer {drink}."},
        {"role": "user", "content": "Thanks. One last recall: what's my dog's name?"},
        {"role": "assistant", "content": f"Your dog's name is {pet}."},
    ]
    validate_messages(messages)
    return pack_example(
        messages,
        "identity_intermediate_recall",
        [
            build_fact_group(pet, sample_negatives(rng, PET_NAMES, pet)),
        ],
    )


def make_identity_with_delta(rng):
    pet = pick(rng, PET_NAMES)
    new_pet = pick_other(rng, PET_NAMES, pet)
    pet_type = "dog"
    messages = [
        make_fact_turn(rng, INTRO_FACTS, [f"my {pet_type}'s name is {pet}"]),
        ack_message(rng),
        make_fact_turn(rng, UPDATE_FACTS, [f"my {pet_type}'s name is now {new_pet} instead of {pet}"]),
        {"role": "assistant", "content": f"Understood. I'll remember {new_pet} as the current {pet_type} name now."},
        {"role": "user", "content": pick(rng, DELTA_QUESTIONS)},
        {"role": "assistant", "content": f"The current {pet_type} name is {new_pet}, and the old one was {pet}."},
    ]
    validate_messages(messages)
    return pack_example(
        messages,
        "identity_with_delta",
        [
            build_fact_group(new_pet, [pet] + sample_negatives(rng, PET_NAMES, new_pet)),
            build_fact_group(pet, [new_pet] + sample_negatives(rng, PET_NAMES, pet)),
        ],
    )


def make_personal_profile_structured(rng):
    pet = pick(rng, PET_NAMES)
    city = pick(rng, CITIES)
    color = pick(rng, COLORS)
    drink = pick(rng, DRINKS)
    breakfast = pick(rng, BREAKFASTS)
    messages = [
        make_fact_turn(rng, INTRO_FACTS, [f"my dog's name is {pet}", f"I live in {city}", f"my favorite color is {color}"]),
        ack_message(rng),
        make_fact_turn(rng, ADD_FACTS, [f"my preferred drink is {drink}", f"my usual breakfast is {breakfast}"]),
        ack_message(rng),
        {"role": "user", "content": pick(rng, PERSONAL_STRUCTURED_QUESTIONS)},
        {"role": "assistant", "content": pick(rng, PERSONAL_STRUCTURED_ANSWERS).format(pet=pet, city=city, drink=drink)},
    ]
    validate_messages(messages)
    return pack_example(
        messages,
        "personal_profile_structured",
        [
            build_fact_group(pet, sample_negatives(rng, PET_NAMES, pet)),
            build_fact_group(city, sample_negatives(rng, CITIES, city)),
            build_fact_group(drink, sample_negatives(rng, DRINKS, drink)),
        ],
    )


def make_tooling_memory(rng):
    shell = pick(rng, SHELLS)
    editor = pick(rng, EDITORS)
    os_name = pick(rng, OSES)
    database = pick(rng, DATABASES)
    answer_format = pick(rng, FORMATS)
    tone = pick(rng, TONES)
    python_version = pick(rng, PYTHON_VERSIONS)
    messages = [
        make_fact_turn(rng, INTRO_FACTS, [f"I usually work in {shell}", f"my editor is {editor}", f"I prefer answers in {answer_format}"]),
        ack_message(rng),
        make_fact_turn(rng, ADD_FACTS, [f"I am on {os_name}", f"my preferred database is {database}", f"I am targeting Python {python_version}", f"keep the tone {tone}"]),
        ack_message(rng),
        {"role": "user", "content": pick(rng, TOOLING_QUESTIONS)},
        {"role": "assistant", "content": pick(rng, TOOLING_ANSWERS).format(shell=shell, editor=editor, database=database, answer_format=answer_format)},
    ]
    validate_messages(messages)
    return pack_example(
        messages,
        "tooling_memory",
        [
            build_fact_group(shell, sample_negatives(rng, SHELLS, shell)),
            build_fact_group(editor, sample_negatives(rng, EDITORS, editor)),
            build_fact_group(database, sample_negatives(rng, DATABASES, database)),
            build_fact_group(answer_format, sample_negatives(rng, FORMATS, answer_format)),
        ],
    )


def make_tooling_update(rng):
    shell = pick(rng, SHELLS)
    editor = pick(rng, EDITORS)
    database = pick(rng, DATABASES)
    new_database = pick_other(rng, DATABASES, database)
    answer_format = pick(rng, FORMATS)
    messages = [
        make_fact_turn(rng, INTRO_FACTS, [f"I usually work in {shell}", f"my editor is {editor}", f"my preferred database is {database}"]),
        ack_message(rng),
        make_fact_turn(rng, UPDATE_FACTS, [f"switch my preferred database from {database} to {new_database}", f"I still want answers in {answer_format}"]),
        {"role": "assistant", "content": f"Got it. I'll remember that you now prefer {new_database} and still want {answer_format}."},
        {"role": "user", "content": pick(rng, TOOLING_QUESTIONS)},
        {"role": "assistant", "content": pick(rng, TOOLING_ANSWERS).format(shell=shell, editor=editor, database=new_database, answer_format=answer_format)},
    ]
    validate_messages(messages)
    return pack_example(
        messages,
        "tooling_update",
        [
            build_fact_group(shell, sample_negatives(rng, SHELLS, shell)),
            build_fact_group(editor, sample_negatives(rng, EDITORS, editor)),
            build_fact_group(new_database, [database] + sample_negatives(rng, DATABASES, new_database)),
            build_fact_group(answer_format, sample_negatives(rng, FORMATS, answer_format)),
        ],
    )


def make_project_rollout(rng):
    repo = pick(rng, REPOS)
    region = pick(rng, REGIONS)
    cluster = pick(rng, CLUSTERS)
    branch = pick(rng, BRANCHES)
    namespace = pick(rng, NAMESPACES)
    python_version = pick(rng, PYTHON_VERSIONS)
    messages = [
        make_fact_turn(rng, INTRO_FACTS, [f"the repo is {repo}", f"the deployment region is {region}", f"the cluster is {cluster}"]),
        ack_message(rng),
        make_fact_turn(rng, ADD_FACTS, [f"the branch is {branch}", f"the namespace is {namespace}", f"the Python version is {python_version}"]),
        ack_message(rng),
        {"role": "user", "content": pick(rng, PROJECT_QUESTIONS)},
        {"role": "assistant", "content": pick(rng, PROJECT_ANSWERS).format(repo=repo, region=region, branch=branch, cluster=cluster, python_version=python_version)},
    ]
    validate_messages(messages)
    return pack_example(
        messages,
        "project_rollout",
        [
            build_fact_group(repo, sample_negatives(rng, REPOS, repo)),
            build_fact_group(region, sample_negatives(rng, REGIONS, region)),
            build_fact_group(branch, sample_negatives(rng, BRANCHES, branch)),
            build_fact_group(cluster, sample_negatives(rng, CLUSTERS, cluster)),
            build_fact_group(python_version, sample_negatives(rng, PYTHON_VERSIONS, python_version)),
        ],
    )


def make_project_update(rng):
    repo = pick(rng, REPOS)
    region = pick(rng, REGIONS)
    cluster = pick(rng, CLUSTERS)
    branch = pick(rng, BRANCHES)
    new_branch = pick_other(rng, BRANCHES, branch)
    python_version = pick(rng, PYTHON_VERSIONS)
    messages = [
        make_fact_turn(rng, INTRO_FACTS, [f"the repo is {repo}", f"the region is {region}", f"the branch is {branch}"]),
        ack_message(rng),
        make_fact_turn(rng, UPDATE_FACTS, [f"use {new_branch} instead of {branch} for the current rollout", f"the cluster remains {cluster}", f"the Python version is {python_version}"]),
        {"role": "assistant", "content": f"Understood. I'll remember the branch as {new_branch} now."},
        {"role": "user", "content": pick(rng, PROJECT_QUESTIONS)},
        {"role": "assistant", "content": pick(rng, PROJECT_ANSWERS).format(repo=repo, region=region, branch=new_branch, cluster=cluster, python_version=python_version)},
    ]
    validate_messages(messages)
    return pack_example(
        messages,
        "project_update",
        [
            build_fact_group(repo, sample_negatives(rng, REPOS, repo)),
            build_fact_group(region, sample_negatives(rng, REGIONS, region)),
            build_fact_group(new_branch, [branch] + sample_negatives(rng, BRANCHES, new_branch)),
            build_fact_group(cluster, sample_negatives(rng, CLUSTERS, cluster)),
            build_fact_group(python_version, sample_negatives(rng, PYTHON_VERSIONS, python_version)),
        ],
    )


def make_service_runtime(rng):
    service = pick(rng, REPOS)
    region = pick(rng, REGIONS)
    namespace = pick(rng, NAMESPACES)
    channel = pick(rng, CHANNELS)
    port = pick(rng, PORT_BASES) + rng.randrange(50)
    url = f"https://api.{service}.{region}.example.com"
    messages = [
        make_fact_turn(rng, INTRO_FACTS, [f"the service is {service}", f"the base URL is {url}", f"the port is {port}"]),
        ack_message(rng),
        make_fact_turn(rng, ADD_FACTS, [f"the namespace is {namespace}", f"the Slack channel is {channel}"]),
        ack_message(rng),
        {"role": "user", "content": pick(rng, SERVICE_QUESTIONS)},
        {"role": "assistant", "content": pick(rng, SERVICE_ANSWERS).format(url=url, port=port, namespace=namespace, channel=channel)},
    ]
    validate_messages(messages)
    negative_url = f"https://api.{pick_other(rng, REPOS, service)}.{pick_other(rng, REGIONS, region)}.example.com"
    negative_port = str(pick(rng, PORT_BASES) + rng.randrange(50))
    return pack_example(
        messages,
        "service_runtime",
        [
            build_fact_group(url, [negative_url]),
            build_fact_group(str(port), [negative_port]),
            build_fact_group(namespace, sample_negatives(rng, NAMESPACES, namespace)),
            build_fact_group(channel, sample_negatives(rng, CHANNELS, channel)),
        ],
    )


def make_service_update(rng):
    service = pick(rng, REPOS)
    region = pick(rng, REGIONS)
    namespace = pick(rng, NAMESPACES)
    channel = pick(rng, CHANNELS)
    new_channel = pick_other(rng, CHANNELS, channel)
    port = pick(rng, PORT_BASES) + rng.randrange(50)
    url = f"https://api.{service}.{region}.example.com"
    messages = [
        make_fact_turn(rng, INTRO_FACTS, [f"the service is {service}", f"the base URL is {url}", f"the namespace is {namespace}"]),
        ack_message(rng),
        make_fact_turn(rng, UPDATE_FACTS, [f"use {new_channel} instead of {channel} as the main Slack channel", f"the port is {port}"]),
        {"role": "assistant", "content": f"Got it. I'll remember {new_channel} as the main channel now."},
        {"role": "user", "content": pick(rng, SERVICE_QUESTIONS)},
        {"role": "assistant", "content": pick(rng, SERVICE_ANSWERS).format(url=url, port=port, namespace=namespace, channel=new_channel)},
    ]
    validate_messages(messages)
    negative_url = f"https://api.{pick_other(rng, REPOS, service)}.{pick_other(rng, REGIONS, region)}.example.com"
    negative_port = str(pick(rng, PORT_BASES) + rng.randrange(50))
    return pack_example(
        messages,
        "service_update",
        [
            build_fact_group(url, [negative_url]),
            build_fact_group(str(port), [negative_port]),
            build_fact_group(namespace, sample_negatives(rng, NAMESPACES, namespace)),
            build_fact_group(new_channel, [channel] + sample_negatives(rng, CHANNELS, new_channel)),
        ],
    )


def make_customer_account(rng):
    customer = pick(rng, CUSTOMER_NAMES)
    plan = pick(rng, PLAN_NAMES)
    renewal = pick(rng, DATES)
    account_id = f"{customer.split()[0][:2].upper()}-{1000 + rng.randrange(9000)}"
    csm = person(rng)
    messages = [
        make_fact_turn(rng, INTRO_FACTS, [f"{customer} is on the {plan} plan", f"the account ID is {account_id}", f"the renewal date is {renewal}"]),
        ack_message(rng),
        make_fact_turn(rng, ADD_FACTS, [f"the customer success manager is {csm}"]),
        ack_message(rng),
        {"role": "user", "content": pick(rng, CUSTOMER_QUESTIONS)},
        {"role": "assistant", "content": pick(rng, CUSTOMER_ANSWERS).format(customer=customer, account_id=account_id, plan=plan, renewal=renewal, csm=csm)},
    ]
    validate_messages(messages)
    negative_account_id = f"{customer.split()[0][:2].upper()}-{1000 + rng.randrange(9000)}"
    return pack_example(
        messages,
        "customer_account",
        [
            build_fact_group(customer, sample_negatives(rng, CUSTOMER_NAMES, customer)),
            build_fact_group(account_id, [negative_account_id]),
            build_fact_group(plan, sample_negatives(rng, PLAN_NAMES, plan)),
            build_fact_group(renewal, sample_negatives(rng, DATES, renewal)),
            build_fact_group(csm, [person(rng), person(rng)]),
        ],
    )


def make_customer_update(rng):
    customer = pick(rng, CUSTOMER_NAMES)
    plan = pick(rng, PLAN_NAMES)
    new_plan = pick_other(rng, PLAN_NAMES, plan)
    renewal = pick(rng, DATES)
    account_id = f"{customer.split()[0][:2].upper()}-{1000 + rng.randrange(9000)}"
    csm = person(rng)
    messages = [
        make_fact_turn(rng, INTRO_FACTS, [f"{customer} is on the {plan} plan", f"the account ID is {account_id}", f"the customer success manager is {csm}"]),
        ack_message(rng),
        make_fact_turn(rng, UPDATE_FACTS, [f"move {customer} from the {plan} plan to the {new_plan} plan", f"the renewal date is {renewal}"]),
        {"role": "assistant", "content": f"Understood. I'll remember {customer} as being on the {new_plan} plan now."},
        {"role": "user", "content": pick(rng, CUSTOMER_QUESTIONS)},
        {"role": "assistant", "content": pick(rng, CUSTOMER_ANSWERS).format(customer=customer, account_id=account_id, plan=new_plan, renewal=renewal, csm=csm)},
    ]
    validate_messages(messages)
    negative_account_id = f"{customer.split()[0][:2].upper()}-{1000 + rng.randrange(9000)}"
    return pack_example(
        messages,
        "customer_update",
        [
            build_fact_group(customer, sample_negatives(rng, CUSTOMER_NAMES, customer)),
            build_fact_group(account_id, [negative_account_id]),
            build_fact_group(new_plan, [plan] + sample_negatives(rng, PLAN_NAMES, new_plan)),
            build_fact_group(renewal, sample_negatives(rng, DATES, renewal)),
            build_fact_group(csm, [person(rng), person(rng)]),
        ],
    )


def make_schedule_memory(rng):
    weekday = pick(rng, WEEKDAYS)
    time_value = pick(rng, TIMES)
    room = pick(rng, ROOMS)
    host = person(rng)
    city = pick(rng, CITIES)
    messages = [
        make_fact_turn(rng, INTRO_FACTS, [f"the review is on {weekday} at {time_value}", f"the room is {room}", f"the city is {city}"]),
        ack_message(rng),
        make_fact_turn(rng, ADD_FACTS, [f"the host is {host}"]),
        ack_message(rng),
        {"role": "user", "content": pick(rng, SCHEDULE_QUESTIONS)},
        {"role": "assistant", "content": pick(rng, SCHEDULE_ANSWERS).format(weekday=weekday, time_value=time_value, room=room, host=host)},
    ]
    validate_messages(messages)
    return pack_example(
        messages,
        "schedule_memory",
        [
            build_fact_group(weekday, sample_negatives(rng, WEEKDAYS, weekday)),
            build_fact_group(time_value, sample_negatives(rng, TIMES, time_value)),
            build_fact_group(room, sample_negatives(rng, ROOMS, room)),
            build_fact_group(host, [person(rng), person(rng)]),
        ],
    )


def make_schedule_update(rng):
    weekday = pick(rng, WEEKDAYS)
    time_value = pick(rng, TIMES)
    new_time = pick_other(rng, TIMES, time_value)
    room = pick(rng, ROOMS)
    host = person(rng)
    messages = [
        make_fact_turn(rng, INTRO_FACTS, [f"the review is on {weekday} at {time_value}", f"the room is {room}", f"the host is {host}"]),
        ack_message(rng),
        make_fact_turn(rng, UPDATE_FACTS, [f"move the review from {time_value} to {new_time}"]),
        {"role": "assistant", "content": f"Got it. I'll remember the review time as {new_time} now."},
        {"role": "user", "content": pick(rng, SCHEDULE_QUESTIONS)},
        {"role": "assistant", "content": pick(rng, SCHEDULE_ANSWERS).format(weekday=weekday, time_value=new_time, room=room, host=host)},
    ]
    validate_messages(messages)
    return pack_example(
        messages,
        "schedule_update",
        [
            build_fact_group(weekday, sample_negatives(rng, WEEKDAYS, weekday)),
            build_fact_group(new_time, [time_value] + sample_negatives(rng, TIMES, new_time)),
            build_fact_group(room, sample_negatives(rng, ROOMS, room)),
            build_fact_group(host, [person(rng), person(rng)]),
        ],
    )


def make_service_binding_pair(rng):
    service_a = pick(rng, REPOS)
    service_b = pick_other(rng, REPOS, service_a)
    channel_a = pick(rng, CHANNELS)
    channel_b = pick_other(rng, CHANNELS, channel_a)
    region = pick(rng, REGIONS)
    messages = [
        make_fact_turn(rng, INTRO_FACTS, [f"{service_a} uses {channel_a}", f"{service_b} uses {channel_b}", f"both are in {region}"]),
        ack_message(rng),
    ]
    target_service = pick(rng, [service_a, service_b])
    target_channel = channel_a if target_service == service_a else channel_b
    messages.extend(
        [
            {"role": "user", "content": pick(rng, PAIR_BINDING_QUESTIONS).format(service=target_service)},
            {"role": "assistant", "content": pick(rng, PAIR_BINDING_ANSWERS).format(service=target_service, channel=target_channel)},
        ]
    )
    validate_messages(messages)
    return pack_example(
        messages,
        "service_binding_pair",
        [
            build_fact_group(target_channel, [channel_b if target_channel == channel_a else channel_a] + sample_negatives(rng, CHANNELS, target_channel)),
        ],
    )


def make_customer_binding_pair(rng):
    customer_a = pick(rng, CUSTOMER_NAMES)
    customer_b = pick_other(rng, CUSTOMER_NAMES, customer_a)
    plan_a = pick(rng, PLAN_NAMES)
    plan_b = pick_other(rng, PLAN_NAMES, plan_a)
    messages = [
        make_fact_turn(rng, INTRO_FACTS, [f"{customer_a} is on the {plan_a} plan", f"{customer_b} is on the {plan_b} plan"]),
        ack_message(rng),
    ]
    target_customer = pick(rng, [customer_a, customer_b])
    target_plan = plan_a if target_customer == customer_a else plan_b
    messages.extend(
        [
            {"role": "user", "content": pick(rng, CUSTOMER_BINDING_QUESTIONS).format(customer=target_customer)},
            {"role": "assistant", "content": pick(rng, CUSTOMER_BINDING_ANSWERS).format(customer=target_customer, plan=target_plan)},
        ]
    )
    validate_messages(messages)
    return pack_example(
        messages,
        "customer_binding_pair",
        [
            build_fact_group(target_plan, [plan_b if target_plan == plan_a else plan_a] + sample_negatives(rng, PLAN_NAMES, target_plan)),
        ],
    )


STAGE_FAMILY_SPECS = {
    "kv_stage1_large": [
        {"name": "identity_template", "weight": 5, "builder": make_identity_template},
        {"name": "identity_update", "weight": 4, "builder": make_identity_update},
        {"name": "identity_binding_pair", "weight": 3, "builder": make_identity_binding_pair},
        {"name": "identity_intermediate_recall", "weight": 2, "builder": make_identity_intermediate_recall},
        {"name": "identity_with_delta", "weight": 2, "builder": make_identity_with_delta},
        {"name": "personal_profile_structured", "weight": 2, "builder": make_personal_profile_structured},
    ],
    "kv_stage2_large": [
        {"name": "personal_profile_structured", "weight": 4, "builder": make_personal_profile_structured},
        {"name": "tooling_memory", "weight": 3, "builder": make_tooling_memory},
        {"name": "tooling_update", "weight": 3, "builder": make_tooling_update},
        {"name": "project_rollout", "weight": 3, "builder": make_project_rollout},
        {"name": "service_runtime", "weight": 3, "builder": make_service_runtime},
        {"name": "customer_account", "weight": 3, "builder": make_customer_account},
        {"name": "schedule_memory", "weight": 2, "builder": make_schedule_memory},
        {"name": "service_binding_pair", "weight": 2, "builder": make_service_binding_pair},
        {"name": "customer_binding_pair", "weight": 2, "builder": make_customer_binding_pair},
    ],
    "kv_stage3_large": [
        {"name": "identity_update", "weight": 2, "builder": make_identity_update},
        {"name": "identity_binding_pair", "weight": 2, "builder": make_identity_binding_pair},
        {"name": "identity_intermediate_recall", "weight": 2, "builder": make_identity_intermediate_recall},
        {"name": "personal_profile_structured", "weight": 3, "builder": make_personal_profile_structured},
        {"name": "tooling_update", "weight": 3, "builder": make_tooling_update},
        {"name": "project_update", "weight": 3, "builder": make_project_update},
        {"name": "service_update", "weight": 3, "builder": make_service_update},
        {"name": "customer_update", "weight": 3, "builder": make_customer_update},
        {"name": "schedule_update", "weight": 2, "builder": make_schedule_update},
        {"name": "service_binding_pair", "weight": 2, "builder": make_service_binding_pair},
        {"name": "customer_binding_pair", "weight": 2, "builder": make_customer_binding_pair},
        {"name": "identity_with_delta", "weight": 2, "builder": make_identity_with_delta},
    ],
}


DEFAULT_SIZES = {
    "kv_stage1_large_train.jsonl": 220_000,
    "kv_stage1_large_eval.jsonl": 4_000,
    "kv_stage2_large_train.jsonl": 260_000,
    "kv_stage2_large_eval.jsonl": 4_000,
    "kv_stage3_large_train.jsonl": 320_000,
    "kv_stage3_large_eval.jsonl": 4_000,
}


def choose_weighted_family(rng, family_specs):
    total = sum(spec["weight"] for spec in family_specs)
    cursor = rng.uniform(0, total)
    running = 0.0
    for spec in family_specs:
        running += spec["weight"]
        if cursor <= running:
            return spec
    return family_specs[-1]


def generate_rows(stage_key, n, seed):
    rng = random.Random(seed)
    family_specs = STAGE_FAMILY_SPECS[stage_key]
    rows = []
    family_counts = {spec["name"]: 0 for spec in family_specs}
    for _ in range(n):
        spec = choose_weighted_family(rng, family_specs)
        row = spec["builder"](rng)
        rows.append(row)
        family_counts[spec["name"]] += 1
    return rows, family_counts


def generate_corpus(output_dir, sizes, seed):
    output_dir.mkdir(parents=True, exist_ok=True)
    written = []
    stage_info = [
        ("kv_stage1_large_train.jsonl", "kv_stage1_large", seed + 11),
        ("kv_stage1_large_eval.jsonl", "kv_stage1_large", seed + 12),
        ("kv_stage2_large_train.jsonl", "kv_stage2_large", seed + 21),
        ("kv_stage2_large_eval.jsonl", "kv_stage2_large", seed + 22),
        ("kv_stage3_large_train.jsonl", "kv_stage3_large", seed + 31),
        ("kv_stage3_large_eval.jsonl", "kv_stage3_large", seed + 32),
    ]
    for filename, stage_key, sub_seed in stage_info:
        rows, family_counts = generate_rows(stage_key, sizes[filename], sub_seed)
        path = output_dir / filename
        write_jsonl(path, rows)
        written.append(
            {
                "name": filename,
                "path": str(path),
                "num_examples": len(rows),
                "stage_key": stage_key,
                "seed": sub_seed,
                "family_counts": family_counts,
            }
        )

    manifest = {
        "seed": seed,
        "output_dir": str(output_dir),
        "files": written,
        "stage_families": {
            stage_key: [{"name": spec["name"], "weight": spec["weight"]} for spec in specs]
            for stage_key, specs in STAGE_FAMILY_SPECS.items()
        },
        "notes": [
            "train files are intended for staged K/V campaigns",
            "eval files are held-out synthetic checks, not replacements for the existing focus/binding/hq eval family",
            "all conversations keep valid alternating user/assistant turns",
            "the generator intentionally includes updates, binding, intermediate recall, and structured-answer families",
        ],
    }
    manifest_path = output_dir / "kv_large_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return manifest_path


def parse_args():
    parser = argparse.ArgumentParser(description="Generate larger staged K/V-memory datasets")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--seed", type=int, default=42)
    for filename, default in DEFAULT_SIZES.items():
        arg = "--" + filename.replace(".jsonl", "").replace("_", "-")
        parser.add_argument(arg, type=int, default=default)
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    sizes = {filename: getattr(args, filename.replace(".jsonl", "")) for filename in DEFAULT_SIZES}
    manifest_path = generate_corpus(output_dir, sizes, args.seed)
    print(f"Wrote large K/V corpus to {output_dir}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
