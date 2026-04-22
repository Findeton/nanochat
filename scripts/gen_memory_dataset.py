"""
Generate synthetic recall-focused chat data for Hopfield memory training.

Each conversation is structured so every eligible user->assistant target pair
requires recalling facts written in earlier turns. This matches how
scripts.chat_memory builds training episodes.

Example:
python -m scripts.gen_memory_dataset --train-size 4096 --eval-size 512
"""

import argparse
import json
import os
import random


parser = argparse.ArgumentParser(description="Generate synthetic recall-focused JSONL datasets")
parser.add_argument("--train-size", type=int, default=4096, help="number of training conversations")
parser.add_argument("--eval-size", type=int, default=512, help="number of eval conversations")
parser.add_argument("--seed", type=int, default=42, help="random seed")
parser.add_argument("--output-dir", type=str, default=None, help="directory for memory_train.jsonl and memory_eval.jsonl")
parser.add_argument("--num-recalls", type=int, default=3, choices=[1, 2, 3], help="number of recall user/assistant pairs per conversation")
args = parser.parse_args()


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = args.output_dir or os.path.join(ROOT_DIR, "data")


ADJECTIVES = [
    "amber", "brisk", "cinder", "daring", "ember", "frozen", "golden", "harbor",
    "indigo", "juniper", "keen", "lively", "mellow", "navy", "opal", "prairie",
    "quartz", "rustic", "silver", "tidal", "umber", "velvet", "willow", "zephyr",
]

ANIMALS = [
    "otter", "badger", "falcon", "panda", "lynx", "heron", "beaver", "tiger",
    "koala", "eagle", "yak", "fox", "whale", "ibis", "gecko", "rabbit",
]

CITIES = [
    "Austin", "Bogota", "Lisbon", "Oslo", "Seoul", "Medellin", "Valencia", "Kyoto",
    "Quito", "Tallinn", "Lima", "Porto", "Denver", "Bristol", "Lyon", "Curitiba",
]

COLORS = [
    "green", "blue", "orange", "teal", "red", "gray", "yellow", "black",
    "white", "maroon", "cyan", "olive", "navy", "coral", "indigo", "brown",
]

DRINKS = [
    "coffee", "tea", "sparkling water", "matcha", "cold brew", "lemonade", "mate", "chai",
]

DATABASES = [
    "Postgres", "SQLite", "DuckDB", "MySQL", "ClickHouse", "Redis",
]

SHELLS = [
    "zsh", "bash", "fish",
]

EDITORS = [
    "Neovim", "VS Code", "Helix", "Emacs", "Sublime Text",
]

PYTHON_VERSIONS = [
    "3.10", "3.11", "3.12",
]

REGIONS = [
    "prod1-euw1", "prod1-use1", "staging-euw2", "qa-use2", "dev-usw2", "lab-sae1",
]

CLUSTERS = [
    "atlas", "boron", "cedar", "delta", "ember", "fjord", "glacier", "helios",
]

STYLES = [
    "short paragraphs", "brief bullet lists", "compact prose", "step-by-step instructions",
]

OSES = [
    "macOS", "Linux", "Ubuntu", "Fedora", "Debian",
]

REPO_TOPICS = [
    "payments", "identity", "chat", "search", "billing", "analytics", "gitops", "client",
]

SERVICE_TOPICS = [
    "gateway", "console", "worker", "sync", "notifier", "registry", "dashboard", "router",
]

WEEKDAYS = [
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
]

TIMES = [
    "08:30", "09:15", "10:45", "13:00", "14:20", "16:10", "17:35",
]


def pick(rng, values):
    return rng.choice(values)


def combo(rng):
    return f"{pick(rng, ADJECTIVES)}-{pick(rng, ANIMALS)}"


def make_branch(rng, index):
    return f"feature/{combo(rng)}-{100 + index % 900}"


def make_repo(rng):
    return f"{pick(rng, REPO_TOPICS)}-{pick(rng, SERVICE_TOPICS)}"


def make_namespace(rng, index):
    return f"team-{pick(rng, ADJECTIVES)}-{index % 17}"


def make_ticket(rng, index):
    prefix = pick(rng, ["PX", "OPS", "ENG", "REL", "APP", "DB"])
    return f"{prefix}-{1000 + index}"


def make_url(rng, index):
    return f"https://api.{combo(rng)}-{index % 97}.example/v{1 + index % 3}"


def make_port(index):
    return str(7000 + (index % 2000))


def render_messages(facts, prompts, answers):
    intro = "Remember these details for later: " + "; ".join(facts) + "."
    ack = "Understood. I will keep those details in mind."
    messages = [
        {"role": "user", "content": intro},
        {"role": "assistant", "content": ack},
    ]
    for prompt, answer in zip(prompts[:args.num_recalls], answers[:args.num_recalls]):
        messages.append({"role": "user", "content": prompt})
        messages.append({"role": "assistant", "content": answer})
    return messages


def conversation_profile(rng, index):
    pet = combo(rng).title()
    city = pick(rng, CITIES)
    color = pick(rng, COLORS)
    drink = pick(rng, DRINKS)
    facts = [
        f"my dog's name is {pet}",
        f"I live in {city}",
        f"my favorite color is {color}",
        f"my preferred drink is {drink}",
    ]
    prompts = [
        "What is my dog's name?",
        "Where do I live and what is my favorite color?",
        "Summarize my dog, city, favorite color, and preferred drink in one sentence.",
    ]
    answers = [
        f"Your dog's name is {pet}.",
        f"You live in {city}, and your favorite color is {color}.",
        f"Your dog's name is {pet}, you live in {city}, your favorite color is {color}, and your preferred drink is {drink}.",
    ]
    return render_messages(facts, prompts, answers)


def conversation_tooling(rng, index):
    database = pick(rng, DATABASES)
    shell = pick(rng, SHELLS)
    editor = pick(rng, EDITORS)
    style = pick(rng, STYLES)
    os_name = pick(rng, OSES)
    facts = [
        f"my preferred database is {database}",
        f"I usually work in {shell}",
        f"my editor is {editor}",
        f"I want answers in {style}",
        f"I am on {os_name}",
    ]
    prompts = [
        "Which database do I prefer?",
        "What shell and editor do I use?",
        "How should you format answers for me, and what operating system am I on?",
    ]
    answers = [
        f"You prefer {database}.",
        f"You use {shell} and {editor}.",
        f"I should answer in {style}, and you are on {os_name}.",
    ]
    return render_messages(facts, prompts, answers)


def conversation_deploy(rng, index):
    repo = make_repo(rng)
    region = pick(rng, REGIONS)
    cluster = pick(rng, CLUSTERS)
    python_version = pick(rng, PYTHON_VERSIONS)
    branch = make_branch(rng, index)
    facts = [
        f"the repo is {repo}",
        f"the region is {region}",
        f"the cluster is {cluster}",
        f"the Python version is {python_version}",
        f"the branch is {branch}",
    ]
    prompts = [
        "What repo and region are we working in?",
        "Which cluster and Python version should we use?",
        "Give me the repo, branch, cluster, region, and Python version.",
    ]
    answers = [
        f"We are working in the {repo} repo and the {region} region.",
        f"We should use the {cluster} cluster and Python {python_version}.",
        f"The repo is {repo}, the branch is {branch}, the cluster is {cluster}, the region is {region}, and the Python version is {python_version}.",
    ]
    return render_messages(facts, prompts, answers)


def conversation_runtime(rng, index):
    service = make_repo(rng)
    url = make_url(rng, index)
    namespace = make_namespace(rng, index)
    port = make_port(index)
    ticket = make_ticket(rng, index)
    facts = [
        f"the service name is {service}",
        f"the base URL is {url}",
        f"the namespace is {namespace}",
        f"the port is {port}",
        f"the change ticket is {ticket}",
    ]
    prompts = [
        "What base URL should we use?",
        "What namespace and port should I keep in mind?",
        "State the service name, base URL, namespace, port, and change ticket.",
    ]
    answers = [
        f"We should use {url}.",
        f"You should keep {namespace} and port {port} in mind.",
        f"The service name is {service}, the base URL is {url}, the namespace is {namespace}, the port is {port}, and the change ticket is {ticket}.",
    ]
    return render_messages(facts, prompts, answers)


def conversation_schedule(rng, index):
    weekday = pick(rng, WEEKDAYS)
    time_value = pick(rng, TIMES)
    city = pick(rng, CITIES)
    ticket = make_ticket(rng, index)
    room = f"{pick(rng, ADJECTIVES)}-{1 + index % 12}"
    facts = [
        f"the review is on {weekday}",
        f"the meeting time is {time_value}",
        f"the city is {city}",
        f"the room is {room}",
        f"the reference code is {ticket}",
    ]
    prompts = [
        "When is the review?",
        "What city and room are associated with it?",
        "Give me the day, time, city, room, and reference code.",
    ]
    answers = [
        f"The review is on {weekday} at {time_value}.",
        f"It is associated with {city} and room {room}.",
        f"The review is on {weekday} at {time_value}, the city is {city}, the room is {room}, and the reference code is {ticket}.",
    ]
    return render_messages(facts, prompts, answers)


TEMPLATES = [
    conversation_profile,
    conversation_tooling,
    conversation_deploy,
    conversation_runtime,
    conversation_schedule,
]


def build_dataset(count, seed):
    rng = random.Random(seed)
    rows = []
    for index in range(count):
        template = TEMPLATES[index % len(TEMPLATES)]
        rows.append(template(rng, index))
    return rows


def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for messages in rows:
            f.write(json.dumps(messages))
            f.write("\n")


train_rows = build_dataset(args.train_size, args.seed)
eval_rows = build_dataset(args.eval_size, args.seed + 1000)

suffix = "" if args.num_recalls == 3 else f"_r{args.num_recalls}"
train_path = os.path.join(OUTPUT_DIR, f"memory_train{suffix}.jsonl")
eval_path = os.path.join(OUTPUT_DIR, f"memory_eval{suffix}.jsonl")

write_jsonl(train_path, train_rows)
write_jsonl(eval_path, eval_rows)

print(f"Wrote {len(train_rows)} training conversations to {train_path}")
print(f"Wrote {len(eval_rows)} eval conversations to {eval_path}")
