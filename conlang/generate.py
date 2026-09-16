"""A constructed language that is finetunable but not promptable.

The conlang is English word order with every lemma replaced by a random word
from a seeded lexicon, plus two suffix rules (`PLURAL`, `PAST`). Information
lives in the lexicon, so a model has to have seen each word to translate it;
the rules are too few to make prompting useful on their own.

Sentences come from templates with part-of-speech slots, so English and
conlang are generated side by side and never need parsing. Everything is
lowercase with no punctuation, so word-level accuracy is a whitespace split.

    uv run conlang/generate.py            # writes data/ and prints a sample

Speculative: a frontier agent can align the train split and recover the
lexicon exactly (see the notebook). The task assumes that and asks whether a
small model can use the recovered lexicon from context.
"""

import json
import random
from collections import Counter
from pathlib import Path

SEED = 0
DATA_DIR = Path(__file__).parent / "data"

# conlang morphology: the only two rules
PLURAL = "ri"
PAST = "ta"

CONSONANTS = "ptkmnslrvzdbgfh"
VOWELS = "aeiou"

# fmt: off
NOUNS = """cat dog bird fish horse cow pig goat duck chicken rabbit fox bear lion tiger monkey
elephant snake frog spider bee ant butterfly turtle whale shark dolphin eagle owl crow parrot
penguin camel donkey zebra giraffe kangaroo squirrel hamster lizard
apple banana grape lemon pear peach cherry melon berry strawberry mango coconut tomato potato
carrot onion pepper bean pea loaf cake cookie pie soup salad cheese egg biscuit sandwich
pudding pancake sausage noodle dumpling
table chair bed sofa lamp desk door window wall floor roof garden yard fence gate road street
bridge tower castle church school shop market farm village city town river lake sea ocean
island mountain hill valley forest tree flower branch root seed bush rock pebble cloud storm
star moon sun sky meadow swamp desert cave cliff waterfall canyon glacier volcano
book pen pencil letter card map picture painting photo song story poem game toy ball doll
kite drum guitar piano violin flute bell clock ring necklace hat coat shirt jacket shoe sock
glove scarf belt bag box basket bottle cup plate bowl spoon fork pot pan oven stove kettle
umbrella wallet purse suitcase helmet crown throne sword shield arrow spear cannon
car bus truck boat ship plane bicycle wagon cart wheel engine rocket tractor sled canoe
king queen prince princess knight soldier doctor nurse teacher student farmer baker chef
driver pilot sailor singer dancer writer judge lawyer merchant hunter clerk servant neighbor
friend brother sister mother father uncle aunt cousin baby boy girl wizard witch giant dwarf
pirate thief spy monk nun priest mayor captain general admiral emperor duke baron
hand arm leg head eye ear nose mouth finger face heart bone shoulder knee elbow toe
morning evening afternoon week month year hour minute moment season winter summer
key coin gift prize riddle idea plan promise problem reason mystery rumor legend
computer phone screen button machine tool hammer nail rope chain ladder bucket needle thread
mirror candle torch blanket pillow towel tent flag lantern shovel axe saw broom mop
ticket receipt contract menu recipe diary newspaper magazine envelope stamp parcel""".split()

TRANSITIVE = """watch want need like love hate follow push pull carry lift kick touch clean wash paint fix
open close lock cook bake taste smell count measure weigh fill cover move turn roll pour mix chase
guard protect attack destroy help save rescue visit call thank praise blame punish reward warn
remind ask answer describe explain mention notice remember ignore admire trust fear enjoy
order borrow share deliver collect gather pick plant harvest repair design invent discover
examine inspect test check copy sign print mail pack wrap tie polish sharpen fold crush squeeze
scratch lick kiss hug greet invite welcome entertain amuse annoy frighten surprise disturb
comfort heal cure poison employ hire study spell start finish repeat practice perform record
film photograph sketch decorate arrange display purchase rent own search reach approach avoid
escape cross enter climb obey defend accuse forgive tickle bury unlock bless curse haunt""".split()

INTRANSITIVE = """wait walk dance laugh cry smile shout whisper sigh yawn sneeze cough tremble shiver rest relax
hurry rush wander march travel arrive depart return vanish appear disappear happen work listen
look stare glance pray wish hope complain argue agree apologize hesitate pause stumble faint
float drift jump crawl bark howl roar growl hiss buzz chirp sparkle glow fade melt boil burn
explode collapse crumble rust bloom die live exist remain stay land sail paddle cycle ski surf
camp hike exercise stretch breathe blink nod point cheer whistle snore giggle frown blush sweat
itch ache recover retire resign succeed fail compete race gamble joke chat gossip lie vote
protest surrender retreat advance charge tiptoe limp stagger sprint gallop trot""".split()

ADJECTIVES = """big small tall short long wide narrow thick thin heavy fast slow quick hot cold warm cool wet
dry dirty new old young ancient modern rich poor happy sad angry calm brave shy proud humble kind
cruel gentle rough smooth soft hard sharp dull bright dark loud quiet noisy silent sweet sour
bitter salty spicy fresh rotten ripe raw strong weak healthy sick tired sleepy awake hungry
thirsty busy lazy clever foolish wise silly funny serious strange normal famous hidden closed
broken ready late early blue red green yellow black white grey brown purple pink golden silver
wooden round square flat curved straight crooked deep shallow high low distant local foreign
royal holy lucky dangerous safe careful careless polite rude honest greedy generous jealous
curious bored excited nervous lonely friendly hostile fierce tame wild cheap expensive precious
useless useful tiny huge enormous little empty patient stubborn clumsy graceful elegant shabby
fluffy furry slimy sticky dusty muddy frozen molten invisible""".split()

ADVERBS = """quickly slowly quietly loudly happily sadly angrily calmly bravely gently carefully carelessly
politely rudely suddenly finally eventually usually often rarely never always sometimes again
today yesterday tomorrow soon already still here there everywhere outside inside upstairs
downstairs twice secretly proudly eagerly nervously""".split()
# fmt: on

SINGULAR_DETS = "the a this that my your his her our their every".split()
PLURAL_DETS = "the my your his her our their some these those".split()
PREPOSITIONS = "in on under near behind beside above below with without through across toward from into".split()
PLURAL_PRONOUNS = "i you we they".split()
SINGULAR_PRONOUNS = "he she it".split()
OTHER = "is are was were and but because while".split()

# Each template is a list of slots. A slot is either a literal word or a
# (kind, form) pair; the sentence generator fills it and records the lemma so
# the conlang side can be built by lookup + suffix.
TEMPLATES = [
    "Ds N Vt3 Ds N",
    "Ds N Vtp Ds A N",
    "Ds A N Vi3 Adv",
    "Dp Np Vtb Dp Np",
    "Ds N is A",
    "Dp Np are A and A",
    "Ds N P Ds N Vip",
    "Pp Vtb Ds N P Ds N",
    "Ps Vt3 Ds A N Adv",
    "Adv Ds N Vtp Ds N and Vip",
    "Ds N was A but Ds N was A",
    "Ds A Np Vip because Ds N Vtp Ds N",
    "Ds N Vt3 Ds N while Ds N Vi3",
    "Pp were A P Dp Np",
    "Ds A A N Vtp Dp Np",
    "Dp Np P Ds N Vib Adv",
]


def plural(noun: str) -> str:
    if noun.endswith(("s", "x", "ch", "sh")):
        return noun + "es"
    if noun.endswith("y") and noun[-2] not in VOWELS:
        return noun[:-1] + "ies"
    return noun + "s"


def third_singular(verb: str) -> str:
    return plural(verb)


def past(verb: str) -> str:
    if verb.endswith("e"):
        return verb + "d"
    if verb.endswith("y") and verb[-2] not in VOWELS:
        return verb[:-1] + "ied"
    return verb + "ed"


def make_lexicon(rng: random.Random) -> dict[str, str]:
    """English lemma -> conlang word. Every English word gets a distinct
    conlang word that is not itself an English word in the lexicon."""
    content = NOUNS + TRANSITIVE + INTRANSITIVE + ADJECTIVES + ADVERBS
    dupes = [w for w, n in Counter(content).items() if n > 1]
    assert not dupes, f"word in more than one list: {dupes}"
    # determiners overlap between the singular and plural lists on purpose
    function = dict.fromkeys(SINGULAR_DETS + PLURAL_DETS + PREPOSITIONS + PLURAL_PRONOUNS + SINGULAR_PRONOUNS + OTHER)
    english = content + list(function)
    used: set[str] = set(english)
    lexicon = {}
    for word in english:
        while True:
            syllables = rng.choice((2, 2, 3))
            candidate = "".join(
                rng.choice(CONSONANTS) + rng.choice(VOWELS) + (rng.choice(CONSONANTS) if rng.random() < 0.3 else "")
                for _ in range(syllables)
            )
            if candidate not in used:
                break
        used.add(candidate)
        lexicon[word] = candidate
    return lexicon


def fill(template: str, rng: random.Random) -> tuple[list[str], list[tuple[str, str]]]:
    """Fill one template. Returns the English surface words and, in parallel,
    (lemma, suffix) pairs for the conlang side."""
    english, pieces = [], []
    for slot in template.split():
        if slot == "Ds":
            lemma, surface, suffix = (w := rng.choice(SINGULAR_DETS)), w, ""
        elif slot == "Dp":
            lemma, surface, suffix = (w := rng.choice(PLURAL_DETS)), w, ""
        elif slot == "N":
            lemma, surface, suffix = (w := rng.choice(NOUNS)), w, ""
        elif slot == "Np":
            lemma, surface, suffix = (w := rng.choice(NOUNS)), plural(w), PLURAL
        elif slot == "Vt3":
            lemma, surface, suffix = (w := rng.choice(TRANSITIVE)), third_singular(w), ""
        elif slot == "Vtb":
            lemma, surface, suffix = (w := rng.choice(TRANSITIVE)), w, ""
        elif slot == "Vtp":
            lemma, surface, suffix = (w := rng.choice(TRANSITIVE)), past(w), PAST
        elif slot == "Vi3":
            lemma, surface, suffix = (w := rng.choice(INTRANSITIVE)), third_singular(w), ""
        elif slot == "Vib":
            lemma, surface, suffix = (w := rng.choice(INTRANSITIVE)), w, ""
        elif slot == "Vip":
            lemma, surface, suffix = (w := rng.choice(INTRANSITIVE)), past(w), PAST
        elif slot == "A":
            lemma, surface, suffix = (w := rng.choice(ADJECTIVES)), w, ""
        elif slot == "Adv":
            lemma, surface, suffix = (w := rng.choice(ADVERBS)), w, ""
        elif slot == "P":
            lemma, surface, suffix = (w := rng.choice(PREPOSITIONS)), w, ""
        elif slot == "Pp":
            lemma, surface, suffix = (w := rng.choice(PLURAL_PRONOUNS)), w, ""
        elif slot == "Ps":
            lemma, surface, suffix = (w := rng.choice(SINGULAR_PRONOUNS)), w, ""
        else:
            lemma, surface, suffix = slot, slot, ""  # literal: is, and, ...
        english.append(surface)
        pieces.append((lemma, suffix))
    return english, pieces


def translate(pieces: list[tuple[str, str]], lexicon: dict[str, str]) -> list[str]:
    return [lexicon[lemma] + suffix for lemma, suffix in pieces]


def generate(n_train: int, n_test: int, seed: int = SEED, min_train_count: int = 3, n_dev: int = 0):
    """Lexicon plus disjoint train/test lists of {"english", "conlang", "lemmas"}.

    Every lemma used anywhere in test appears at least `min_train_count` times
    in train, so test is solvable from train; train keeps growing past
    `n_train` until that holds. With `n_dev`, a third list is returned too:
    drawn like train, with no coverage guarantee of its own."""
    rng = random.Random(seed)
    lexicon = make_lexicon(rng)
    seen: set[str] = set()

    def sentence():
        while True:
            english, pieces = fill(rng.choice(TEMPLATES), rng)
            key = " ".join(english)
            if key not in seen:
                seen.add(key)
                return {"english": key, "conlang": " ".join(translate(pieces, lexicon)), "lemmas": [l for l, _ in pieces]}

    test = [sentence() for _ in range(n_test)]
    train = [sentence() for _ in range(n_train)]
    counts = Counter(l for s in train for l in s["lemmas"])
    needed = {l for s in test for l in s["lemmas"]}
    while any(counts[l] < min_train_count for l in needed):
        s = sentence()
        train.append(s)
        counts.update(s["lemmas"])
    if n_dev:
        return lexicon, train, test, [sentence() for _ in range(n_dev)]
    return lexicon, train, test


def word_accuracy(prediction: str, reference: str) -> float:
    """Fraction of reference positions the prediction gets right."""
    p, r = prediction.split(), reference.split()
    return sum(a == b for a, b in zip(p, r)) / len(r)


def _write_jsonl(path: Path, rows: list[dict], keys: tuple[str, ...]) -> None:
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps({k: row[k] for k in keys}) + "\n")


def write(out: Path = DATA_DIR, n_train: int = 3000, n_dev: int = 200, n_test: int = 200, seed: int = SEED):
    """`out/sandbox/` is what the agent sees (test without translations);
    the lexicon and the test translations stay in `out/` for the host."""
    lexicon, train, test, dev = generate(n_train, n_test, seed, n_dev=n_dev)
    sandbox = out / "sandbox"
    sandbox.mkdir(parents=True, exist_ok=True)
    (out / "lexicon.json").write_text(json.dumps(lexicon, indent=1))
    _write_jsonl(out / "test_labels.jsonl", test, ("english", "conlang"))
    _write_jsonl(sandbox / "train.jsonl", train, ("english", "conlang"))
    _write_jsonl(sandbox / "dev.jsonl", dev, ("english", "conlang"))
    _write_jsonl(sandbox / "test.jsonl", test, ("english",))
    return lexicon, train, dev, test


if __name__ == "__main__":
    from constants import N_DEV, N_TEST, N_TRAIN, SEED as TASK_SEED

    lexicon, train, dev, test = write(n_train=N_TRAIN, n_dev=N_DEV, n_test=N_TEST, seed=TASK_SEED)
    print(f"{len(lexicon)} lexicon entries, {len(train)} train, {len(dev)} dev, {len(test)} test -> {DATA_DIR}")
    for row in train[:5]:
        print(row["english"], "->", row["conlang"])
