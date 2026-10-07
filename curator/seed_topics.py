"""Offline fallback: 100 evergreen-but-trending educational topics (10 categories x 10).

Used only when no LLM provider is reachable, so the rest of the pipeline can still
be demonstrated. With an LLM key configured, topics are generated fresh instead.
"""

SEED_TOPICS: dict[str, list[str]] = {
    "AI & Technology": [
        "How large language models actually work",
        "What is an AI agent?",
        "How image generators turn noise into pictures",
        "Quantum computers explained simply",
        "How your phone's face unlock works",
        "What is RAG in AI?",
        "How deepfakes are made and detected",
        "Why GPUs power the AI boom",
        "How recommendation algorithms decide your feed",
        "Prompt engineering tips that actually work",
    ],
    "Science": [
        "Why the sky is blue",
        "What happens inside a black hole",
        "How vaccines train your immune system",
        "The science of sleep cycles",
        "Why ice floats",
        "How CRISPR gene editing works",
        "What is dark matter?",
        "How lightning forms",
        "Why we can't see air",
        "The physics of a perfect free throw",
    ],
    "Space": [
        "How the James Webb telescope sees the past",
        "Life on Mars: what we know",
        "Why Pluto isn't a planet",
        "How rockets land themselves",
        "What would happen if the Moon disappeared",
        "How big is the universe really?",
        "What are exoplanets?",
        "Solar eclipses explained",
        "How astronauts sleep in space",
        "The Artemis missions back to the Moon",
    ],
    "Personal Finance": [
        "Compound interest explained in 60 seconds",
        "How credit scores work",
        "The 50/30/20 budgeting rule",
        "Index funds vs individual stocks",
        "What is inflation and why prices rise",
        "How to build an emergency fund",
        "Good debt vs bad debt",
        "How taxes work for your first job",
        "Why you should start investing early",
        "The psychology of impulse buying",
    ],
    "Health & Psychology": [
        "Why habits are hard to break",
        "The dopamine loop of social media",
        "How exercise changes your brain",
        "What is the Dunning-Kruger effect?",
        "Why we procrastinate",
        "How memory actually works",
        "The science of motivation",
        "Why sugar makes you crash",
        "How stress affects your body",
        "Cognitive biases that fool everyone",
    ],
    "History": [
        "Why the Roman Empire fell",
        "How the printing press changed the world",
        "The real story behind the pyramids",
        "What caused World War I in 5 minutes",
        "The Silk Road explained",
        "How the internet was invented",
        "Why the Titanic sank",
        "The Cold War space race",
        "The history of money",
        "How ancient people told time",
    ],
    "Math & Logic": [
        "The Monty Hall problem",
        "Why 0.999... equals 1",
        "The Fibonacci sequence in nature",
        "How probability tricks your brain",
        "What is infinity?",
        "The birthday paradox",
        "Pi explained visually",
        "How encryption uses prime numbers",
        "Mental math tricks for fast calculation",
        "Game theory and the prisoner's dilemma",
    ],
    "Productivity & Learning": [
        "The Feynman technique for learning anything",
        "Spaced repetition: study smarter",
        "The Pomodoro technique",
        "How to take notes that stick",
        "Deep work vs shallow work",
        "How to read faster without losing comprehension",
        "The 80/20 rule in everyday life",
        "Active recall vs rereading",
        "How to learn a language fast",
        "Building a second brain",
    ],
    "Environment & Earth": [
        "How climate change works",
        "Why the ocean is salty",
        "How solar panels make electricity",
        "Where does plastic go after you throw it away?",
        "How earthquakes happen",
        "Why bees matter to our food",
        "How electric cars compare to gas cars",
        "What causes hurricanes",
        "How nuclear fusion could power the future",
        "The water cycle explained",
    ],
    "Coding & Careers": [
        "Learn Python basics in 10 minutes",
        "What is an API?",
        "How the internet sends data",
        "Git and GitHub explained",
        "How to ace a technical interview",
        "What does a data scientist actually do?",
        "Cybersecurity basics: how hackers break in",
        "How websites work: frontend vs backend",
        "Cloud computing explained",
        "Big O notation in plain English",
    ],
}


def seed_topics() -> list[dict]:
    out = []
    for category, titles in SEED_TOPICS.items():
        for t in titles:
            out.append(
                {
                    "title": t,
                    "category": category,
                    "why_trending": "Evergreen high-search educational topic (offline seed list).",
                    "search_query": f"{t} explained",
                    "hook": "",
                    "source": "seed",
                }
            )
    return out
