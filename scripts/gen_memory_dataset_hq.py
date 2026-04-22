"""
Generate a higher-quality recall-focused dataset for Hopfield memory training.

This generator uses curated scenario templates with more natural dialogue and
multi-step contexts. The final user/assistant pair is the intended training
target, so pair this dataset with:

python -m scripts.chat_memory --target-selection final --custom-json data/memory_train_hq.jsonl
"""

import argparse
import json
import os
import random


parser = argparse.ArgumentParser(description="Generate high-quality memory training datasets")
parser.add_argument("--train-size", type=int, default=24000, help="number of training conversations")
parser.add_argument("--eval-size", type=int, default=2000, help="number of eval conversations")
parser.add_argument("--seed", type=int, default=42, help="random seed")
parser.add_argument("--output-dir", type=str, default=None, help="directory for output JSONL files")
args = parser.parse_args()


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = args.output_dir or os.path.join(ROOT_DIR, "data")


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
    "Poppy", "Maple", "Miso", "Tuna", "Clover", "Nori", "Pico", "Milo",
    "Luna", "Pepper", "Biscuit", "Pixel", "Ruby", "Hazel", "Mochi", "Olive",
]

PET_TYPES = [
    "dog", "cat", "corgi", "beagle", "rabbit", "parrot",
]

CITIES = [
    "Austin", "Bogota", "Porto", "Medellin", "Lisbon", "Quito", "Seoul", "Kyoto",
    "Denver", "Tallinn", "Valencia", "Lima", "Bristol", "Curitiba", "Oslo", "Lyon",
]

COLORS = [
    "green", "blue", "teal", "orange", "white", "navy", "coral", "indigo",
]

DRINKS = [
    "matcha", "coffee", "tea", "sparkling water", "cold brew", "chai",
]

BREAKFASTS = [
    "eggs on toast", "yogurt with fruit", "oatmeal", "arepas", "avocado toast", "granola",
]

DATABASES = [
    "Postgres", "DuckDB", "SQLite", "MySQL", "ClickHouse", "Redis",
]

SHELLS = [
    "zsh", "bash", "fish",
]

EDITORS = [
    "Neovim", "VS Code", "Helix", "Emacs",
]

OSES = [
    "macOS", "Linux", "Ubuntu", "Debian", "Fedora",
]

PYTHON_VERSIONS = [
    "3.10", "3.11", "3.12",
]

REPOS = [
    "billing-api", "identity-console", "search-worker", "chat-gateway",
    "client-router", "analytics-sync", "gitops-deployer", "auth-service",
]

REGIONS = [
    "prod1-euw1", "prod1-use1", "staging-euw2", "qa-use2", "dev-usw2", "lab-sae1",
]

CLUSTERS = [
    "atlas", "boron", "cedar", "delta", "fjord", "glacier", "helios", "ion",
]

NAMESPACES = [
    "team-aurora", "team-cedar", "team-harbor", "team-ember", "team-summit", "team-vista",
]

CHANNELS = [
    "#shipping-room", "#ops-europe", "#incident-triage", "#client-rollout",
    "#platform-support", "#finops", "#auth-oncall", "#release-watch",
]

PLAN_NAMES = [
    "Growth", "Business", "Enterprise", "Premier",
]

CUSTOMER_NAMES = [
    "Acorn Health", "Northstar Labs", "Blue Mesa", "Lumen Retail",
    "Cinder Foods", "Harbor Transit", "Juniper Schools", "Atlas Solar",
]

CODENAMES = [
    "Project Lantern", "Project Cedar", "Project Marlin", "Project Solstice",
    "Project Orchard", "Project Meridian", "Project Nimbus", "Project Harbor",
]

ROOMS = [
    "Rio 4", "Andes 2", "Maple 7", "Cedar 3", "Harbor 5", "Atlas 1",
]

WEEKDAYS = [
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
]

TIMES = [
    "08:30", "09:15", "10:45", "13:00", "14:20", "16:10", "17:35",
]

FORMATS = [
    "short paragraphs", "compact prose", "brief bullet lists", "step-by-step instructions",
]

TONES = [
    "calm", "direct", "friendly", "concise",
]

BRANCHES = [
    "feature/fast-retry", "feature/memory-bank", "feature/client-refresh",
    "hotfix/auth-timeout", "release/2026-04", "feature/safe-migrations",
]

ACCOUNT_IDS = [
    "AH-4821", "NS-1938", "BM-5504", "LR-2719", "CF-8840", "HT-6132",
]

DATES = [
    "September 18", "October 4", "November 12", "June 9", "July 21", "August 30",
]


INTRO_FACTS = [
    "For future chats, please remember these details:",
    "Please keep the following details in memory for later:",
    "I want you to hold onto these details across sessions:",
    "Keep these notes for later follow-ups:",
]

ACKS = [
    "Got it. I'll remember that.",
    "Understood. I'll keep those details in mind.",
    "Sounds good. I'll store that for later.",
    "I have it. I'll remember those details.",
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


def pick(rng, values):
    return rng.choice(values)


def sample_distinct(rng, values, k):
    return rng.sample(values, k)


def person(rng):
    return f"{pick(rng, FIRST_NAMES)} {pick(rng, LAST_NAMES)}"


def account_id(rng, index):
    base = pick(rng, ACCOUNT_IDS)
    prefix = base.split("-")[0]
    return f"{prefix}-{1000 + index % 9000}"


def answer_style_triplet(rng):
    return pick(rng, FORMATS), pick(rng, TONES), pick(rng, DATABASES)


def conv_personal_memory(rng, index):
    pet_name = pick(rng, PET_NAMES)
    pet_type = pick(rng, PET_TYPES)
    city = pick(rng, CITIES)
    color = pick(rng, COLORS)
    drink = pick(rng, DRINKS)
    breakfast = pick(rng, BREAKFASTS)
    messages = [
        {"role": "user", "content": f"{pick(rng, INTRO_FACTS)} my {pet_type}'s name is {pet_name}, I live in {city}, and my favorite color is {color}."},
        {"role": "assistant", "content": pick(rng, ACKS)},
        {"role": "user", "content": f"{pick(rng, ADD_FACTS)} my preferred drink is {drink}, and my usual breakfast is {breakfast}."},
        {"role": "assistant", "content": pick(rng, ACKS)},
        {"role": "user", "content": "Without me repeating the earlier details, what is my pet's name, where do I live, and what drink do I prefer?"},
        {"role": "assistant", "content": f"Your pet's name is {pet_name}, you live in {city}, and you prefer {drink}."},
    ]
    return messages


def conv_work_preferences(rng, index):
    shell = pick(rng, SHELLS)
    editor = pick(rng, EDITORS)
    os_name = pick(rng, OSES)
    answer_format, tone, database = answer_style_triplet(rng)
    python_version = pick(rng, PYTHON_VERSIONS)
    messages = [
        {"role": "user", "content": f"{pick(rng, INTRO_FACTS)} I usually work in {shell}, my editor is {editor}, and I prefer answers in {answer_format}."},
        {"role": "assistant", "content": pick(rng, ACKS)},
        {"role": "user", "content": f"{pick(rng, ADD_FACTS)} I am on {os_name}, my preferred database is {database}, and I am targeting Python {python_version}. Keep the tone {tone}."},
        {"role": "assistant", "content": pick(rng, ACKS)},
        {"role": "user", "content": "What shell and editor do I use, what database do I prefer, and how should you format answers?"},
        {"role": "assistant", "content": f"You use {shell} and {editor}, you prefer {database}, and I should answer in {answer_format}."},
    ]
    return messages


def conv_project_config(rng, index):
    repo = pick(rng, REPOS)
    region = pick(rng, REGIONS)
    cluster = pick(rng, CLUSTERS)
    branch = pick(rng, BRANCHES)
    namespace = pick(rng, NAMESPACES)
    python_version = pick(rng, PYTHON_VERSIONS)
    messages = [
        {"role": "user", "content": f"{pick(rng, INTRO_FACTS)} the repo is {repo}, the deployment region is {region}, and the cluster is {cluster}."},
        {"role": "assistant", "content": pick(rng, ACKS)},
        {"role": "user", "content": f"{pick(rng, ADD_FACTS)} the branch is {branch}, the namespace is {namespace}, and the Python version is {python_version}."},
        {"role": "assistant", "content": pick(rng, ACKS)},
        {"role": "user", "content": "For the current setup, what repo, region, branch, and Python version should we use?"},
        {"role": "assistant", "content": f"We should use the {repo} repo in {region}, the {branch} branch, and Python {python_version}."},
    ]
    return messages


def conv_service_runtime(rng, index):
    service = pick(rng, REPOS)
    region = pick(rng, REGIONS)
    namespace = pick(rng, NAMESPACES)
    channel = pick(rng, CHANNELS)
    port = 7000 + (index % 2000)
    url = f"https://api.{service}.{region}.example.com"
    ticket = f"INC-{2000 + index}"
    messages = [
        {"role": "user", "content": f"{pick(rng, INTRO_FACTS)} the service is {service}, the base URL is {url}, and the port is {port}."},
        {"role": "assistant", "content": pick(rng, ACKS)},
        {"role": "user", "content": f"{pick(rng, ADD_FACTS)} the namespace is {namespace}, the incident ticket is {ticket}, and the Slack channel is {channel}."},
        {"role": "assistant", "content": pick(rng, ACKS)},
        {"role": "user", "content": "What base URL, port, namespace, and Slack channel should we use for this service?"},
        {"role": "assistant", "content": f"We should use {url}, port {port}, namespace {namespace}, and the Slack channel {channel}."},
    ]
    return messages


def conv_schedule(rng, index):
    weekday = pick(rng, WEEKDAYS)
    time_value = pick(rng, TIMES)
    room = pick(rng, ROOMS)
    city = pick(rng, CITIES)
    host = person(rng)
    code = f"MEET-{3000 + index}"
    messages = [
        {"role": "user", "content": f"{pick(rng, INTRO_FACTS)} the review is on {weekday} at {time_value}, and the room is {room}."},
        {"role": "assistant", "content": pick(rng, ACKS)},
        {"role": "user", "content": f"{pick(rng, ADD_FACTS)} the city is {city}, the host is {host}, and the reference code is {code}."},
        {"role": "assistant", "content": pick(rng, ACKS)},
        {"role": "user", "content": "When is the review, where is it, and who is hosting it?"},
        {"role": "assistant", "content": f"The review is on {weekday} at {time_value} in {room}, and it is hosted by {host}."},
    ]
    return messages


def conv_customer_account(rng, index):
    customer = pick(rng, CUSTOMER_NAMES)
    plan = pick(rng, PLAN_NAMES)
    renewal = pick(rng, DATES)
    customer_id = account_id(rng, index)
    csm = person(rng)
    channel = pick(rng, CHANNELS)
    messages = [
        {"role": "user", "content": f"{pick(rng, INTRO_FACTS)} {customer} is on the {plan} plan, the account ID is {customer_id}, and the renewal date is {renewal}."},
        {"role": "assistant", "content": pick(rng, ACKS)},
        {"role": "user", "content": f"{pick(rng, ADD_FACTS)} the customer success manager is {csm}, and the working Slack channel is {channel}."},
        {"role": "assistant", "content": pick(rng, ACKS)},
        {"role": "user", "content": f"For {customer}, what account ID, plan, renewal date, and customer success manager should we remember?"},
        {"role": "assistant", "content": f"For {customer}, the account ID is {customer_id}, the plan is {plan}, the renewal date is {renewal}, and the customer success manager is {csm}."},
    ]
    return messages


def conv_team_ownership(rng, index):
    owner_a, owner_b, owner_c, escalation = sample_distinct(rng, [person(rng) for _ in range(8)], 4)
    area_a, area_b, area_c = sample_distinct(rng, ["billing", "auth", "search", "analytics", "deployments", "client sync"], 3)
    repo = pick(rng, REPOS)
    channel = pick(rng, CHANNELS)
    messages = [
        {"role": "user", "content": f"{pick(rng, INTRO_FACTS)} {owner_a} owns {area_a}, {owner_b} owns {area_b}, and {owner_c} owns {area_c}."},
        {"role": "assistant", "content": pick(rng, ACKS)},
        {"role": "user", "content": f"{pick(rng, ADD_FACTS)} if {area_a} is blocked, escalate to {escalation} in {channel}. The repo involved is {repo}."},
        {"role": "assistant", "content": pick(rng, ACKS)},
        {"role": "user", "content": f"Who owns {area_a}, who owns {area_b}, and who should we escalate {area_a} issues to?"},
        {"role": "assistant", "content": f"{owner_a} owns {area_a}, {owner_b} owns {area_b}, and {area_a} issues should be escalated to {escalation}."},
    ]
    return messages


def conv_codename_launch(rng, index):
    repo = pick(rng, REPOS)
    codename = pick(rng, CODENAMES)
    region = pick(rng, REGIONS)
    date = pick(rng, DATES)
    branch = pick(rng, BRANCHES)
    messages = [
        {"role": "user", "content": f"{pick(rng, INTRO_FACTS)} the launch codename is {codename}, the repo is {repo}, and the rollout region is {region}."},
        {"role": "assistant", "content": pick(rng, ACKS)},
        {"role": "user", "content": f"{pick(rng, ADD_FACTS)} the rollout date is {date}, and the branch is {branch}."},
        {"role": "assistant", "content": pick(rng, ACKS)},
        {"role": "user", "content": "What codename, repo, region, and branch are tied to this launch?"},
        {"role": "assistant", "content": f"The launch uses codename {codename}, repo {repo}, region {region}, and branch {branch}."},
    ]
    return messages


def conv_update_preferences(rng, index):
    original_db = pick(rng, DATABASES)
    updated_db = pick(rng, [db for db in DATABASES if db != original_db])
    original_format = pick(rng, FORMATS)
    updated_format = pick(rng, [fmt for fmt in FORMATS if fmt != original_format])
    shell = pick(rng, SHELLS)
    messages = [
        {"role": "user", "content": f"{pick(rng, INTRO_FACTS)} I use {shell}, my preferred database is {original_db}, and I like answers in {original_format}."},
        {"role": "assistant", "content": pick(rng, ACKS)},
        {"role": "user", "content": f"{pick(rng, UPDATE_FACTS)} switch my database preference to {updated_db}, and switch my preferred answer format to {updated_format}."},
        {"role": "assistant", "content": pick(rng, ACKS)},
        {"role": "user", "content": "What shell do I use, what database do I currently prefer, and how should you format answers now?"},
        {"role": "assistant", "content": f"You use {shell}, you currently prefer {updated_db}, and I should answer in {updated_format}."},
    ]
    return messages


def conv_update_project(rng, index):
    repo = pick(rng, REPOS)
    region_a, region_b = sample_distinct(rng, REGIONS, 2)
    branch_a, branch_b = sample_distinct(rng, BRANCHES, 2)
    cluster = pick(rng, CLUSTERS)
    python_version = pick(rng, PYTHON_VERSIONS)
    messages = [
        {"role": "user", "content": f"{pick(rng, INTRO_FACTS)} the repo is {repo}, the region is {region_a}, the branch is {branch_a}, and the cluster is {cluster}."},
        {"role": "assistant", "content": pick(rng, ACKS)},
        {"role": "user", "content": f"{pick(rng, UPDATE_FACTS)} move the region to {region_b}, change the branch to {branch_b}, and keep Python {python_version}."},
        {"role": "assistant", "content": pick(rng, ACKS)},
        {"role": "user", "content": "What repo, current region, current branch, and Python version should we use now?"},
        {"role": "assistant", "content": f"We should use the {repo} repo, region {region_b}, branch {branch_b}, and Python {python_version}."},
    ]
    return messages


def conv_contact_book(rng, index):
    person_a, person_b, person_c = sample_distinct(rng, [person(rng) for _ in range(8)], 3)
    role_a, role_b = sample_distinct(rng, ["legal", "security", "billing", "data platform", "client success"], 2)
    email_domain = pick(rng, ["example.com", "acme.dev", "northstar.io", "harbor.app"])
    phone_suffix = 1000 + index % 9000
    messages = [
        {"role": "user", "content": f"{pick(rng, INTRO_FACTS)} {person_a} handles {role_a}, and {person_b} handles {role_b}."},
        {"role": "assistant", "content": pick(rng, ACKS)},
        {"role": "user", "content": f"{pick(rng, ADD_FACTS)} the escalation contact is {person_c}, their email is {person_c.split()[0].lower()}@{email_domain}, and their extension is {phone_suffix}."},
        {"role": "assistant", "content": pick(rng, ACKS)},
        {"role": "user", "content": f"Who handles {role_a}, and who is the escalation contact with the email you should remember?"},
        {"role": "assistant", "content": f"{person_a} handles {role_a}, and the escalation contact is {person_c} at {person_c.split()[0].lower()}@{email_domain}."},
    ]
    return messages


SCENARIOS = [
    conv_personal_memory,
    conv_work_preferences,
    conv_project_config,
    conv_service_runtime,
    conv_schedule,
    conv_customer_account,
    conv_team_ownership,
    conv_codename_launch,
    conv_update_preferences,
    conv_update_project,
    conv_contact_book,
]


def build_dataset(count, seed):
    rng = random.Random(seed)
    rows = []
    for index in range(count):
        scenario = SCENARIOS[index % len(SCENARIOS)]
        rows.append(scenario(rng, index))
    return rows


def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for messages in rows:
            f.write(json.dumps(messages, ensure_ascii=False))
            f.write("\n")


train_rows = build_dataset(args.train_size, args.seed)
eval_rows = build_dataset(args.eval_size, args.seed + 1000)

train_path = os.path.join(OUTPUT_DIR, "memory_train_hq.jsonl")
eval_path = os.path.join(OUTPUT_DIR, "memory_eval_hq.jsonl")

write_jsonl(train_path, train_rows)
write_jsonl(eval_path, eval_rows)

print(f"Wrote {len(train_rows)} training conversations to {train_path}")
print(f"Wrote {len(eval_rows)} eval conversations to {eval_path}")
