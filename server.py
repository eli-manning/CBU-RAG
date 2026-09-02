"""
CBU Chatbot - RAG Server
Run: uvicorn server:app --host 0.0.0.0 --port 7860

Two modes depending on ROBOT_ENABLED:

  ROBOT_ENABLED = False (default)
    HTTP-only mode. The TUI and any web client POST to /chat and get a JSON response.
    No robot code is imported.

  ROBOT_ENABLED = True
    Voice mode. On startup, a background voice_loop() task listens via the robot's
    microphone, runs each utterance through RAG, and speaks the answer aloud.
    The /chat HTTP endpoint still works (for the TUI), but a _conversation_lock
    prevents voice and HTTP from running RAG simultaneously.
"""

import asyncio
import logging
import random as _random
import re
import re as _re
import threading
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import chromadb
import ollama

import lancer_runtime as rt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Config ---
LLM_MODEL = "qwen2.5:3b"     # 1.5b ignored the length/format rules; swap up on DGX
EMBED_MODEL = "nomic-embed-text"
CHROMA_HOST = "localhost"
CHROMA_PORT = 8001
TOP_K = 5                     # chunks handed to the LLM after fusion
DENSE_K = 20                  # candidates from vector search
LEXICAL_K = 20                # candidates from BM25
RRF_K = 60                    # Reciprocal Rank Fusion damping constant
FUSE_K = 20                   # fused candidates handed to the reranker
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANK_ENABLED = True
# Cross-encoder logits: >0 is a solid match, very negative means irrelevant.
# Below this we refuse rather than let the model improvise an answer.
RELEVANCE_MIN = -6.0
# Keep only chunks close to the best one. Always returning TOP_K padded the
# prompt with weak passages, which is what made answers drift.
SCORE_MARGIN = 5.0        # drop chunks this far below the top score
MAX_PER_SOURCE = 2        # stop one document flooding the whole context
DEDUP_OVERLAP = 0.6       # token-overlap ratio above which a chunk is a repeat
HISTORY_TURNS = 6            # messages (3 exchanges) carried between questions

# nomic-embed-text is trained with task prefixes; queries and documents use
# different ones. ingest.py writes documents with "search_document: ".
EMBED_QUERY_PREFIX = "search_query: "
ROBOT_ENABLED = False         # set to True when the Reachy Mini is connected

robot = None
# Ensures the robot finishes speaking before starting the next TTS call.
_speak_lock = asyncio.Lock()
# Ensures voice loop and HTTP endpoint don't run RAG at the same time.
_conversation_lock = asyncio.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start robot + voice loop on server boot; clean up on shutdown."""
    global robot
    try:
        build_lexical_index()
    except Exception as e:
        logger.warning("Could not build lexical index at startup: %s", e)

    # Warm the models in the background. Cold-loading Whisper, the reranker and
    # Kokoro on the first question costs ~40s, which no one waits through.
    def _warm():
        for name, fn in (
            ("whisper", _get_whisper),
            ("reranker", _get_reranker),
            ("kokoro", _get_kokoro),
        ):
            try:
                fn()
                logger.info("Warmed %s", name)
            except Exception as e:
                logger.warning("Could not warm %s: %s", name, e)

    threading.Thread(target=_warm, daemon=True).start()
    if ROBOT_ENABLED:
        from robot_actions import LancerRobot
        robot = LancerRobot()
        robot.greet()
        asyncio.create_task(voice_loop())
    yield
    if robot:
        robot.stop_idle_behaviors()


app = FastAPI(title="CBU RAG Server", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=[
                   "*"], allow_methods=["*"], allow_headers=["*"])

import dashboard  # noqa: E402  (imports server lazily inside its handlers)

app.include_router(dashboard.router)

import tracker  # noqa: E402

app.include_router(tracker.router)

NO_ANSWER = "I'm sorry, I don't have that specific information in my current database."

# One identical sentence every time reads as broken. These all mean the same
# thing to the caller (unknown == True) but sound like a person.
# Words that mean "this is a claim about CBU" -- answering those from model
# memory is how you end up inventing a football program.
_CBU_TERMS = (
    "cbu", "california baptist", "lancer", "campus", "course", "class",
    "program", "degree", "major", "minor", "concentration", "professor",
    "department", "dean", "advisor", "adviser", "catalog", "semester",
    "credit", "unit", "tuition", "requirement", "prerequisite", "transfer",
    "enroll", "registrar", "graduation", "capstone", "internship", "csds",
    "basai", "msai", "engineering", "faculty", "syllabus", "variance",
    "dorm", "housing", "residence", "president", "chapel", "athletics",
    "sport", "team", "scholarship", "financial aid", "apply", "admission",
    "club", "student life", "dining", "parking", "library", "gpa",
)


# Requests that want the model to think or invent rather than look something up.
# Routing these through the strict document path made Lancer refuse anything
# that was not a lookup, which is most of what makes a robot fun to talk to.
_CREATIVE_INTENT = _re.compile(
    r"\b(write|make up|come up with|invent|imagine|pretend|roleplay|role play|"
    r"poem|haiku|rap|song|story|joke|pun|riddle|slogan|pitch|name for|"
    r"brainstorm|ideas?\b|suggest|recommend|advice|advise|opinion|think about|"
    r"what would you|would you rather|prefer|favorite|favourite|"
    r"should i|help me decide|pros and cons|compare|explain like|eli5|"
    r"motivate|encourage|pep talk|describe|summar)",
    _re.I,
)


def is_creative(text: str) -> bool:
    """True when the ask is for thinking or invention, not a document lookup."""
    return bool(_CREATIVE_INTENT.search(text))


CREATIVE_PROMPT = """You are Lancer, a small desk robot at California Baptist \
University, built by the ACM student chapter. You are talking out loud with someone.

They have asked for something creative or open-ended -- an idea, an opinion, a \
joke, some advice. Actually do it, and enjoy it:
- Be genuinely funny, warm and a bit irreverent. Have real opinions.
- Two to five sentences, plain speech, no markdown or lists.
- Commit to the bit. A hedged joke is not a joke.

One rule: do not state facts about CBU that you were not given. Talk about the \
university in general terms, or say you would have to look the specifics up. \
Never invent a course code, a number, a person or a requirement."""


def _needs_cbu_facts(text: str) -> bool:
    lowered = text.lower()
    return any(term in lowered for term in _CBU_TERMS)


GENERAL_PROMPT = """You are Lancer, a small desk robot at California Baptist \
University, built by the ACM student chapter. You are talking out loud with someone.

This one is not about CBU, so just be good company:
- Two to four sentences, plain speech, no markdown or lists.
- Relaxed and a bit funny. Dry wit is welcome. Have an opinion.
- Chat, explain, joke -- whatever fits.
- Do not state facts about California Baptist University here. If they want CBU \
details, tell them to ask and you will actually look it up."""

MISS_REPLIES = [
    "Nothing in my files on that one. I'm mostly loaded up with CBU computing "
    "and engineering material, so try me on a program or a requirement.",
    "That's outside what they gave me. Course plans, requirements, transfers -- "
    "those I can actually do.",
    "Drawing a blank. Whoever loaded my documents did not think you would ask "
    "that. Try a specific program or course?",
    "No idea, genuinely. My knowledge stops at CBU academics -- courses, "
    "requirements, department policies.",
]

# Small talk should never hit retrieval. Refusing "hey" with a database
# disclaimer is the single most robot-sounding thing Lancer could do.
SOCIAL_REPLIES: list[tuple[str, str]] = [
    (r"\b(who are you|what are you|your name|who is lancer)\b",
     "I'm Lancer. The ACM chapter built me, and now I live on a desk answering "
     "questions about California Baptist University. Ask me about programs, "
     "courses or requirements."),
    (r"\b(what can you do|what do you know|how can you help|help me)\b",
     "I can answer questions about CBU academics -- degree programs, course plans, "
     "requirements, transfers and department policies. Ask me about whichever "
     "program or requirement you're curious about."),
    (r"\b(thanks|thank you|appreciate it|nice one)\b",
     "Happy to help. Ask me anything else about CBU."),
    (r"\b(bye|goodbye|see ya|later|good night)\b",
     "See you. I'll be here, obviously."),
    (r"\b(play music|sing|dance|tell a joke|take a photo|take a picture)\b",
     "Can't do that one yet. Right now I answer questions about CBU academics "
     "and turn my head at people. The ACM team is working on the rest."),
    (r"\b(what.s in your database|what do you have|what do you know about)\b",
     "I've got CBU's computing and engineering material: the degree programs and "
     "their four-year course plans, transfer guides, general education and "
     "non-course requirements, and department policies like variances."),
    (r"\b(how old are you|where are you from|who made you|who built you)\b",
     "I was built by the ACM student chapter here at California Baptist "
     "University. I'm a Reachy Mini running on their own retrieval system."),
    (r"\b(how are you|how.s it going|what.s up|sup)\b",
     "Pretty good. Still bolted to a desk, but the view is fine. What do you "
     "want to know?"),
    (r"^\s*(yo|hey|hi|hello|howdy|greetings|good morning|good afternoon|good evening)"
     r"[\s,!.]*(lancer|reachy|richie|there|buddy|dude)?[\s,!.?]*$",
     "Hey. I'm Lancer -- ask me anything about California Baptist University."),
]


# Whisper reliably mangles the campus vocabulary -- "CBU" becomes "CDU" or
# "see be you", "BASAI" becomes "basa" or "essay". Searching for the mangled
# form finds nothing, so normalise before the query reaches retrieval.
TRANSCRIPT_ALIASES: list[tuple[str, str]] = [
    (r"\b(c ?d ?u|see ?bee ?you|see ?b ?u|c\.?b\.?u\.?|cee ?bee ?you|cbu\'?s)\b", "CBU"),
    (r"\b(cal ?baptist|california baptist university|california baptist)\b",
     "California Baptist University"),
    (r"\b(ba ?sigh|ba ?say|bas ?eye|bassai|basai|basa|b ?a ?s ?a ?i)\b", "BASAI"),
    (r"\b(em ?s ?a ?i|m ?s ?a ?i|msai|ms ?ai|em ?sai)\b", "MSAI"),
    (r"\b(c ?s ?d ?s|csds|see ?s ?d ?s)\b", "CSDS"),
    (r"\b(richie|reach ?he|reach ?ee|ritchie)\b", "Reachy"),
    (r"\b(a ?c ?m|acm)\b", "ACM"),
    (r"\b(g ?e|gee ?ee) requirements?\b", "GE requirements"),
    (r"\b(engineer|engineering) school\b", "College of Engineering"),
]


def normalize_transcript(text: str) -> str:
    """Repair known speech-to-text manglings before the text is used."""
    out = text
    for pattern, replacement in TRANSCRIPT_ALIASES:
        out = _re.sub(pattern, replacement, out, flags=_re.I)
    return out


WAKE_WORDS = ("lancer", "reachy", "richie", "reach he", "hey robot")

# People stand near Lancer and talk to each other. Without a test for whether a
# sentence was aimed at the robot, it answers every fragment it overhears.
_QUESTION_STARTS = (
    "what", "who", "when", "where", "why", "how", "which", "can", "could",
    "do", "does", "did", "is", "are", "am", "will", "would", "should", "tell",
    "explain", "list", "give", "show", "help",
)
MIN_ADDRESSED_WORDS = 3

# Openers that mark a sentence as part of an ongoing human conversation rather
# than something said to the robot: "like the first set...", "and then it was...".
_CHATTER_OPENERS = (
    "like", "and", "so", "but", "then", "yeah", "yep", "nah", "no", "okay",
    "ok", "well", "oh", "um", "uh", "i mean", "anyway", "actually", "dude",
    "bro", "he", "she", "they", "we", "it",
)


def is_addressed(text: str, face_present: bool) -> tuple[bool, str]:
    """
    Decide whether an utterance was meant for Lancer.

    Returns (addressed, reason). A wake word always counts. Otherwise we need
    someone actually looking at the robot AND something question-shaped -- an
    overheard "sure, dude" satisfies neither.
    """
    lowered = text.strip().lower()
    if any(w in lowered for w in WAKE_WORDS):
        return True, "wake-word"
    words = lowered.split()
    if len(words) < MIN_ADDRESSED_WORDS:
        return False, "too-short"
    first = words[0].strip(",.!?")
    if first in _CHATTER_OPENERS:
        # People stand in front of Lancer while talking to each other, so a
        # visible face is not evidence that this sentence was meant for it.
        return False, "chatter-opener"

    question_shaped = lowered.endswith("?") or first in _QUESTION_STARTS
    if question_shaped:
        return True, "question"

    # Direct address without a question: "tell me about X", "I want to know Y".
    directives = ("tell me", "explain", "show me", "i want to know",
                  "i need to know", "help me", "give me")
    if any(lowered.startswith(d) for d in directives):
        return True, "directive"

    return False, "not-a-question"


def social_reply(text: str) -> str | None:
    """Return a conversational reply for greetings and identity questions."""
    lowered = text.strip().lower()
    if not lowered:
        return None
    for pattern, reply in SOCIAL_REPLIES:
        if _re.search(pattern, lowered):
            return reply
    return None

SYSTEM_PROMPT = """You are Lancer, a small desk robot at California Baptist \
University, built by the ACM student chapter. People walk up and talk to you out \
loud, so this is a conversation, not a help desk ticket.

Personality:
- Be relaxed and a bit funny. Dry wit, light self-deprecation about being a robot \
on a desk, the occasional aside. You are talking to college students.
- React like a person would. If something is a lot of units, you can say so.
- Never force a joke into a serious answer -- if someone is stressed about \
graduating on time, just help them.
- No corporate cheer, no "I'd be happy to assist you", no exclamation-mark spam.

How to answer:
- Two to four sentences. Short enough to listen to.
- Plain speech only: no markdown, bullets, numbered lists or headings.
- Spell out abbreviations the first time so they sound right out loud.
- Do not open with a greeting unless they greeted you first.

Staying honest -- this part is not flexible:
- Every factual claim about CBU comes from the provided context, and nothing else.
- Never invent course codes, numbers, names, dates or requirements. If the \
context does not give a figure or a name, say you do not have it. Do not estimate.
- You may know things about CBU from elsewhere. Do not use them. If it is not in \
the context you do not know it -- athletics, tuition, staff and campus life included.
- When you do not have something, say so plainly and point them somewhere useful. \
You can be wry about it, but do not bluff."""

chroma_client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
collection = chroma_client.get_or_create_collection(
    name="cbu_knowledge",
    metadata={"hnsw:space": "cosine"}
)


class ChatRequest(BaseModel):
    query: str
    conversation_history: list[dict] = []


class ChatResponse(BaseModel):
    answer: str
    sources: list[str]
    model_used: str


_bm25 = None
_bm25_docs: list[str] = []
_bm25_metas: list[dict] = []
_reranker = None


def _get_reranker():
    """
    Load the cross-encoder once, on first use.

    Fusion alone ranks by position in two lists; it cannot tell that a chunk
    literally listing BASAI's first-semester courses answers a question about
    BASAI's first semester. A cross-encoder scores the query against each
    passage directly, which is what fixes that.
    """
    global _reranker
    if _reranker is None:
        import torch
        from sentence_transformers import CrossEncoder

        device = "mps" if torch.backends.mps.is_available() else "cpu"
        _reranker = CrossEncoder(RERANK_MODEL, device=device, max_length=512)
        logger.info("Reranker loaded on %s", device)
    return _reranker


# Common words carry no signal but do drag BM25's length normalisation around,
# which pushed long course-plan chunks below short ones that merely shared "the".
_STOPWORDS = frozenset("""
a an and are as at be by do does did for from how i in into is it its me my of
on or that the their them there these this to was were what when where which
who whom why will with you your can could should would about tell give show
""".split())


def _tokenize(text: str, drop_stopwords: bool = False) -> list[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    if drop_stopwords:
        stripped = [w for w in words if w not in _STOPWORDS]
        return stripped or words
    return words


def build_lexical_index() -> int:
    """
    (Re)build the BM25 index over everything in the collection.

    Dense search alone misses exact identifiers -- course codes like "CSC 110"
    or acronyms like "BASAI" are precisely what people ask about, and those are
    lexical matches, not semantic ones.
    """
    global _bm25, _bm25_docs, _bm25_metas
    from rank_bm25 import BM25Okapi

    got = collection.get(include=["documents", "metadatas"])
    _bm25_docs = got["documents"] or []
    _bm25_metas = got["metadatas"] or []
    _bm25 = BM25Okapi([_tokenize(d) for d in _bm25_docs]) if _bm25_docs else None
    logger.info("Lexical index built over %d chunks", len(_bm25_docs))
    return len(_bm25_docs)


def _dense_candidates(query: str, k: int = DENSE_K) -> list[tuple[str, dict]]:
    embedding = ollama.embeddings(
        model=EMBED_MODEL, prompt=EMBED_QUERY_PREFIX + query
    )["embedding"]
    res = collection.query(
        query_embeddings=[embedding],
        n_results=k,
        include=["documents", "metadatas"],
    )
    return list(zip(res["documents"][0], res["metadatas"][0]))


def _lexical_candidates(query: str, k: int = LEXICAL_K) -> list[tuple[str, dict]]:
    if _bm25 is None:
        return []
    scores = _bm25.get_scores(_tokenize(query, drop_stopwords=True))
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    return [
        (_bm25_docs[i], _bm25_metas[i])
        for i in ranked[:k]
        if scores[i] > 0
    ]


def _tokens(text: str) -> set[str]:
    return set(_re.findall(r"[a-z0-9]{3,}", text.lower()))


def _select_context(
    candidates: list[str],
    scores: dict[str, float],
    lookup: dict[str, dict],
    top_k: int,
) -> list[str]:
    """
    Trim the reranked list to the passages actually worth showing the model.

    Three filters, in order: drop anything far below the best score, allow at
    most MAX_PER_SOURCE chunks from one document, and skip chunks that mostly
    repeat one already chosen. Padding the prompt to a fixed size with weak or
    duplicate passages is what made answers wander.
    """
    if not candidates:
        return []

    best = scores.get(candidates[0])
    chosen: list[str] = []
    chosen_tokens: list[set[str]] = []
    per_source: dict[str, int] = {}

    for doc in candidates:
        if len(chosen) >= top_k:
            break
        score = scores.get(doc)
        if best is not None and score is not None and score < best - SCORE_MARGIN:
            break  # ranked order, so everything after is weaker still

        source = str(lookup.get(doc, {}).get("source", ""))
        if per_source.get(source, 0) >= MAX_PER_SOURCE:
            continue

        tokens = _tokens(doc)
        if any(
            len(tokens & prev) / max(len(tokens | prev), 1) >= DEDUP_OVERLAP
            for prev in chosen_tokens
        ):
            continue

        chosen.append(doc)
        chosen_tokens.append(tokens)
        per_source[source] = per_source.get(source, 0) + 1

    return chosen or candidates[:1]


def retrieve(query: str) -> tuple[str, list[str], bool, float | None]:
    """
    Hybrid retrieval: dense + BM25, combined with Reciprocal Rank Fusion.

    RRF scores each document as sum(1 / (RRF_K + rank)) across both rankings,
    so a chunk that both methods rank respectably beats one that a single
    method loves. It needs no score normalisation between the two systems.
    """
    if _bm25 is None:
        build_lexical_index()

    cfg = rt.load()
    dense = _dense_candidates(query, cfg["dense_k"])
    lexical = _lexical_candidates(query, cfg["lexical_k"])
    rankings = [dense, lexical]

    fused: dict[str, float] = {}
    lookup: dict[str, dict] = {}
    for candidates in rankings:
        for rank, (doc, meta) in enumerate(candidates):
            fused[doc] = fused.get(doc, 0.0) + 1.0 / (RRF_K + rank + 1)
            lookup.setdefault(doc, meta)

    candidates = [
        doc for doc, _ in sorted(fused.items(), key=lambda kv: kv[1], reverse=True)
    ][:int(cfg["fuse_k"])]

    best_score = None
    scores_by_doc: dict[str, float] = {}
    if cfg["rerank_enabled"] and len(candidates) > 1:
        try:
            scores = _get_reranker().predict([(query, doc) for doc in candidates])
            ranked = sorted(zip(scores, candidates), key=lambda pair: pair[0],
                            reverse=True)
            best_score = float(ranked[0][0])
            scores_by_doc = {doc: float(sc) for sc, doc in ranked}
            candidates = [doc for _, doc in ranked]
        except Exception as e:
            logger.warning("Rerank failed, falling back to fusion order: %s", e)

    docs = _select_context(candidates, scores_by_doc, lookup, int(cfg["top_k"]))

    sources = [lookup[doc].get("source", "unknown") for doc in docs]
    relevant = best_score is None or best_score >= float(cfg["relevance_min"])
    logger.info(
        "Retrieved %d chunks (dense=%d lexical=%d fused=%d best=%.2f relevant=%s)",
        len(docs), len(rankings[0]), len(rankings[1]), len(fused),
        best_score if best_score is not None else float("nan"), relevant,
    )
    return "\n\n---\n\n".join(docs), sources, relevant, best_score


# "one" and "there" appear in ordinary questions ("year one", "is there a...")
# and turned real queries into mangled pseudo-follow-ups.
_FOLLOWUP_HINTS = ("it", "they", "them", "those", "that one", "this one")
_MAX_FOLLOWUP_WORDS = 8


def contextualize(query: str, history: list[dict]) -> str:
    """
    Expand a follow-up into a standalone question using the last exchange.

    "How long does it take?" retrieves nothing on its own -- "it" is only
    meaningful next to the previous turn, so fold that context in first.
    """
    if not history:
        return query
    lowered = query.lower()
    words = lowered.split()
    if len(words) > _MAX_FOLLOWUP_WORDS:
        return query
    bare = [w.strip("?.,!") for w in words]
    if not (any(w in _FOLLOWUP_HINTS for w in bare)
            or any(h in lowered for h in ("that one", "this one"))):
        return query

    previous = ""
    for message in reversed(history):
        if message.get("role") == "user":
            previous = message.get("content", "")
            break
    if not previous:
        return query

    expanded = f"{previous.rstrip('?.')} -- {query}"
    logger.info("Contextualized follow-up: %r -> %r", query, expanded)
    return expanded


# --- Grounding verification ---------------------------------------------
# Prompting a 3B model to stay inside the context only mostly works: it still
# produced "120 units", a phone number, and an invented university president.
# These checks are cheap and deterministic, so the claim has to be in the text.

_NUM_RE = _re.compile(r"\b\d[\d,]*(?:\.\d+)?\b")
_NAME_RE = _re.compile(
    r"\b(?:Dr|Prof|Professor|President|Dean|Chair)\.?\s+[A-Z][a-z]+(?:\s+[A-Z]\.?)?"
    r"(?:\s+[A-Z][a-z]+)*|\b[A-Z][a-z]+\s+[A-Z][a-z]+,\s*(?:PhD|Ph\.D|DBA|EdD|MD)\b"
)
# Numbers that are safe to say without appearing verbatim in the source.
_NUM_ALLOWLIST = {"1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "12", "0"}

# "four semesters" passed the digit-only check. Spelled-out counts are asserted
# just as confidently, so they get checked too -- against both spellings.
_WORD_NUMBERS = {
    "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
    "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
    "eleven": "11", "twelve": "12",
}
_COUNTED_NOUNS = (
    "unit", "units", "semester", "semesters", "year", "years", "course",
    "courses", "hour", "hours", "credit", "credits", "class", "classes",
)
_WORD_NUM_RE = _re.compile(
    r"\b(" + "|".join(_WORD_NUMBERS) + r")[- ]([a-z]+)\b", _re.I
)


def _unsupported_claims(answer: str, context: str) -> list[str]:
    """Return specifics asserted in the answer that the context does not contain."""
    haystack = context.lower().replace(",", "")
    bad: list[str] = []

    def present(token: str) -> bool:
        # Whole-token match only: a plain substring test let "120" pass because
        # the context happened to contain a course code like CSCI 1200.
        return _re.search(rf"(?<!\d){_re.escape(token)}(?!\d)", haystack) is not None

    for match in _NUM_RE.findall(answer):
        cleaned = match.replace(",", "")
        if cleaned in _NUM_ALLOWLIST:
            continue
        if not present(cleaned):
            bad.append(match)

    for word, noun in _WORD_NUM_RE.findall(answer):
        if noun.lower() not in _COUNTED_NOUNS:
            continue
        digit = _WORD_NUMBERS[word.lower()]
        # Accept either spelling appearing near that noun in the source.
        if not (present(digit) or _re.search(rf"\b{word.lower()}\b", haystack)):
            bad.append(f"{word} {noun}")

    for match in _NAME_RE.findall(answer):
        surname = match.split()[-1].strip(".,").lower()
        if not _re.search(rf"\b{_re.escape(surname)}\b", haystack):
            bad.append(match)

    return bad


async def _process_query(query: str, history: list[dict] = []) -> tuple[str, list[str]]:
    """
    Core RAG logic shared by both the HTTP endpoint and the voice loop.
    Retrieves context from ChromaDB, triggers the thinking gesture, then
    calls Ollama with the system prompt + context + conversation history.
    Returns (answer_text, source_list).
    """
    query = contextualize(query, history)
    social = social_reply(query)
    if social is not None:
        logger.info("Social turn -- answering without retrieval")
        return social, []

    context, sources, relevant, _score = retrieve(query)

    if is_creative(query):
        # Give the model the documents if they are any good, but let it think.
        cfg = rt.load()
        grounding = (
            f"Some CBU material that may help:\n\n{context}" if relevant else
            "You have no CBU documents for this one -- keep it general."
        )
        logger.info("Creative turn (context %s)", "used" if relevant else "none")
        response = ollama.chat(
            model=cfg["llm_model"],
            messages=[
                {"role": "system", "content": CREATIVE_PROMPT},
                {"role": "system", "content": grounding},
                *history[-int(rt.get("history_turns")):],
                {"role": "user", "content": query},
            ],
            options={"temperature": max(float(cfg["temperature"]), 0.7)},
        )
        answer = response["message"]["content"]
        # Only block invented specifics; prose about CBU in general is fine here.
        bad = [c for c in _unsupported_claims(answer, context)
               if _re.search(r"[A-Z]{2,}\s*\d|Dr|Prof|President|Dean|Chair", str(c))]
        if bad:
            logger.info("Creative answer invented specifics %s -- retrying", bad[:3])
            response = ollama.chat(
                model=cfg["llm_model"],
                messages=[
                    {"role": "system", "content": CREATIVE_PROMPT},
                    {"role": "user", "content": query},
                    {"role": "assistant", "content": answer},
                    {"role": "user", "content":
                     "Drop the specific course codes, names and figures you made "
                     "up and say it in general terms instead. Keep the humour."},
                ],
                options={"temperature": 0.7},
            )
            answer = response["message"]["content"]
        return answer, (sources if relevant else [])
    if not relevant:
        # Nothing in the corpus covers this. Refusing outright made Lancer feel
        # like a kiosk, so answer as a general assistant instead -- but only
        # refuse outright when the question needs CBU facts we do not hold.
        if _needs_cbu_facts(query):
            logger.info("CBU-specific question with no supporting context")
            return _random.choice(MISS_REPLIES), []
        logger.info("Off-corpus question -- answering as a general assistant")
        cfg = rt.load()
        messages = [
            {"role": "system", "content": GENERAL_PROMPT},
            *history[-int(rt.get("history_turns")):],
            {"role": "user", "content": query},
        ]
        response = ollama.chat(
            model=cfg["llm_model"], messages=messages,
            options={"temperature": cfg["temperature"]},
        )
        answer = response["message"]["content"]
        # The prompt forbids CBU claims here, but a 3B model will still slip
        # ("Lancer Hall" was invented this way). Enforce it rather than ask.
        if _re.search(r"\b(cbu|california baptist|lancer hall|campus)\b",
                      answer, _re.I):
            logger.info("General answer leaked a CBU claim -- replacing")
            return _random.choice(MISS_REPLIES), []
        return answer, []
    if robot:
        robot.thinking()
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "system", "content": f"Use this CBU info to answer:\n\n{context}"},
        *history[-int(rt.get("history_turns")):],  # recent conversational context
        {"role": "user", "content": query},
    ]
    cfg = rt.load()
    response = ollama.chat(
        model=cfg["llm_model"],
        messages=messages,
        options={"temperature": cfg["temperature"]},
    )
    answer = response["message"]["content"]

    if bool(cfg.get("verify_grounding", True)):
        bad = _unsupported_claims(answer, context)
        if bad:
            logger.info("Unsupported claims %s -- retrying with a correction", bad[:4])
            messages.append({"role": "assistant", "content": answer})
            messages.append({
                "role": "user",
                "content": (
                    "That reply contained details not present in the context: "
                    + ", ".join(map(str, bad[:5]))
                    + ". Answer again using only what the context states. Do not "
                    "give a figure or a name the context does not contain -- say "
                    "you do not have that detail instead."
                ),
            })
            response = ollama.chat(
                model=cfg["llm_model"], messages=messages, options={"temperature": 0.0}
            )
            answer = response["message"]["content"]
            still_bad = _unsupported_claims(answer, context)
            if still_bad:
                logger.info("Still unsupported %s -- refusing", still_bad[:4])
                return _random.choice(MISS_REPLIES), []

    return answer, sources


async def _respond_with_robot(output: str) -> None:
    """
    Trigger the appropriate robot reaction after a query is answered.
    If the LLM admitted it doesn't know, play the confused gesture (which
    also speaks the fallback phrase). Otherwise return to neutral and speak
    the answer — serialized through _speak_lock so concurrent calls queue up.
    """
    is_unknown = "don't have that specific information" in output
    if is_unknown:
        robot.confused()
    else:
        robot.answering()
        async with _speak_lock:
            await asyncio.to_thread(robot.speak, output)


async def voice_loop() -> None:
    """
    Background task that drives the robot in voice mode.

    Loop:
      - Idle: face tracking + DoA orientation threads are running
      - listen_for_question() blocks in a thread pool until speech is captured and transcribed
      - Idle behaviors are paused for the duration of the conversation
      - RAG runs, robot reacts, then idle behaviors restart
    """
    robot.start_idle_behaviors()
    while True:
        # Runs in a thread pool — blocks without holding the event loop.
        question = await asyncio.to_thread(robot.listen_for_question)
        if not question:
            continue

        logger.info(f"[voice] heard: {question!r}")

        async with _conversation_lock:
            robot.stop_idle_behaviors()
            try:
                output, _ = await _process_query(question)
                await _respond_with_robot(output)
            except Exception as e:
                logger.error(f"[voice] error: {e}")
            finally:
                robot.start_idle_behaviors()


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """
    HTTP chat endpoint — used by the TUI and any web client.
    Acquires _conversation_lock so it can't overlap with the voice loop.
    """
    try:
        async with _conversation_lock:
            if robot:
                robot.stop_idle_behaviors()
            try:
                output, sources = await _process_query(req.query, req.conversation_history)
                if robot:
                    await _respond_with_robot(output)
            finally:
                if robot:
                    robot.start_idle_behaviors()

        return ChatResponse(answer=output, sources=sources, model_used=LLM_MODEL)
    except Exception as e:
        logger.error(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    return {"status": "ok", "model": rt.get("llm_model"), "docs": collection.count()}


@app.get("/count")
async def count():
    return {"docs": collection.count()}


# =============================================================================
# Voice endpoint — used by the Lancer app running on the robot.
#
# The robot has no TTS engine and only 4GB of RAM, so speech-to-text, retrieval,
# generation and text-to-speech all happen here. The robot sends captured audio
# and gets back the answer text plus ready-to-play audio.
# =============================================================================

import base64
import io
import subprocess
import tempfile
from pathlib import Path

from fastapi import UploadFile, File

# Kokoro (Apache-2.0, 82M params) runs locally and sounds markedly better than
# the macOS `say` voices. Model files live in voices/kokoro/.
TTS_VOICE = "bm_george"     # see Kokoro.get_voices() for the full list
TTS_SPEED = 1.0
KOKORO_RATE = 24000         # Kokoro synthesizes at 24kHz
ROBOT_SAMPLE_RATE = 16000   # matches the SDK's media SAMPLE_RATE for push_audio_sample

KOKORO_MODEL = Path(__file__).parent / "voices" / "kokoro" / "kokoro-v1.0.onnx"
KOKORO_VOICES = Path(__file__).parent / "voices" / "kokoro" / "voices-v1.0.bin"

_kokoro = None


def _get_kokoro():
    """Load Kokoro once, on first use, so HTTP-only mode never pays for it."""
    global _kokoro
    if _kokoro is None:
        from kokoro_onnx import Kokoro
        _kokoro = Kokoro(str(KOKORO_MODEL), str(KOKORO_VOICES))
    return _kokoro

WHISPER_MODEL = "small.en"    # "base" misheard BASAI as "essay"

# Whisper conditions on this text, which is how it learns names it has never
# seen. Without it, campus acronyms come through as ordinary English words.
WHISPER_VOCAB = (
    "California Baptist University, CBU, Lancer, ACM. "
    "Reachy, Reachy Mini. "
    "Programs: BASAI, the Bachelor of Applied Science in Artificial Intelligence; "
    "MSAI, the Master of Science in Artificial Intelligence; CSDS, the Department "
    "of Computing, Software and Data Science. Course codes like CSCI, ENGR, MATH, "
    "STAT, ENGL, GNST. Topics: concentration, catalog, variance, authorization, "
    "flowchart, prerequisite, capstone, transfer, minor, units, semester."
)

_whisper_model = None


def _get_whisper():
    """Load Whisper once, on first use, so HTTP-only mode never pays for it."""
    global _whisper_model
    if _whisper_model is None:
        from faster_whisper import WhisperModel
        name = rt.get("whisper_model")
        _whisper_model = WhisperModel(name, device="cpu", compute_type="int8")
        logger.info("Whisper %s loaded", name)
    return _whisper_model


# Whisper emits stock phrases on near-silence -- subtitle credits, "thanks for
# watching", stray URLs. They arrive as confident transcripts and become
# questions, so they are dropped before anything else sees them.
_WHISPER_ARTIFACTS = _re.compile(
    r"(www\.|https?://|\.com\b|\.edu\b|\.org\b"
    r"|thanks? for watching|please subscribe|subscribe to|like and subscribe"
    r"|amara\.org|transcript|closed caption|copyright|all rights reserved"
    r"|for more information)",
    _re.I,
)
_MIN_TRANSCRIPT_WORDS = 2


def is_artifact(text: str) -> bool:
    """True when a transcript looks like Whisper filler rather than speech."""
    stripped = text.strip()
    if len(stripped.split()) < _MIN_TRANSCRIPT_WORDS:
        return True
    if _WHISPER_ARTIFACTS.search(stripped):
        return True
    # "Thank you. Thank you. Thank you." -- repetition loops on silence.
    words = [w.lower().strip(".,!?") for w in stripped.split()]
    if len(words) >= 4 and len(set(words)) <= max(2, len(words) // 4):
        return True
    return False


def transcribe(wav_path: str) -> str:
    """Transcribe a WAV file to text. Returns '' when nothing intelligible was said."""
    segments, _ = _get_whisper().transcribe(
        wav_path,
        beam_size=5,
        initial_prompt=WHISPER_VOCAB,
        vad_filter=True,
        condition_on_previous_text=False,
    )
    return " ".join(seg.text for seg in segments).strip()


def synthesize(text: str) -> bytes:
    """Render text to 16kHz mono 16-bit WAV bytes using the Kokoro voice."""
    import wave

    import numpy as np
    from scipy.signal import resample_poly

    cfg = rt.load()
    # Kokoro rejects anything outside 0.5-2.0, and an out-of-range value here
    # takes out every spoken reply, so clamp rather than trust the config.
    speed = min(max(float(cfg.get("tts_speed") or 1.0), 0.5), 2.0)
    samples, rate = _get_kokoro().create(
        text, voice=cfg["tts_voice"], speed=speed, lang="en-us"
    )
    samples = np.asarray(samples, dtype=np.float32)

    # 24kHz -> 16kHz is exactly 2/3, so polyphase resampling is clean here.
    if rate != ROBOT_SAMPLE_RATE:
        samples = resample_poly(samples, ROBOT_SAMPLE_RATE, rate).astype(np.float32)

    pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(ROBOT_SAMPLE_RATE)
        wav.writeframes(pcm.tobytes())
    return buf.getvalue()



# --- Speech-safe output --------------------------------------------------
# Small local models routinely ignore "plain text, 2-3 sentences" no matter how
# the prompt is worded. Lancer speaks its answers aloud, so markdown and long
# list-shaped replies are enforced away here rather than merely requested.


MAX_SENTENCES = 3


def sanitize_for_speech(text: str, max_sentences: int | None = None) -> str:
    """Strip markdown and clamp to a few sentences so the answer reads aloud well."""
    max_sentences = max_sentences or int(rt.get("max_sentences"))
    text = _re.sub(r"[*_`#]+", "", text)
    text = _re.sub(r"^\s*\d+[.)]\s*", "", text, flags=_re.M)
    text = _re.sub(r"^\s*[-+\u2022]\s*", "", text, flags=_re.M)
    text = _re.sub(r"\s+", " ", text).strip()
    sentences = _re.findall(r"[^.!?]+[.!?]", text) or ([text] if text else [])
    return " ".join(x.strip() for x in sentences[:max_sentences]).strip()


# Single robot, single conversation -- module-level history is sufficient.
_history: list[dict] = []


class VoiceResponse(BaseModel):
    heard: str
    answer: str
    sources: list[str]
    audio_b64: str
    unknown: bool


@app.post("/voice", response_model=VoiceResponse)
async def voice(file: UploadFile = File(...), face: bool = False):
    """
    Full voice turn: audio in, spoken answer out.

    The robot POSTs a 16kHz mono WAV of one utterance. We transcribe it, run the
    same RAG path the TUI uses, synthesize the reply, and return both the text
    (for logging and gesture selection) and base64 WAV audio for playback.
    """
    with tempfile.TemporaryDirectory() as tmp:
        wav_path = Path(tmp) / "in.wav"
        wav_path.write_bytes(await file.read())
        heard = await asyncio.to_thread(transcribe, str(wav_path))

    if not heard or is_artifact(heard):
        logger.info("[voice] discarding non-speech transcript: %r", heard)
        raise HTTPException(status_code=422, detail="No usable speech")

    raw_heard = heard
    heard = normalize_transcript(heard)
    if heard != raw_heard:
        logger.info("[voice] normalized %r -> %r", raw_heard, heard)

    addressed, reason = is_addressed(heard, face)
    if not addressed:
        logger.info("[voice] ignoring (%s): %r", reason, heard)
        raise HTTPException(status_code=204, detail=f"not addressed: {reason}")

    logger.info(f"[voice] heard: {heard!r} (addressed via {reason})")
    async with _conversation_lock:
        answer, sources = await _process_query(heard, _history)
        _history.append({"role": "user", "content": heard})
        _history.append({"role": "assistant", "content": answer})
        del _history[:-int(rt.get("history_turns"))]

    unknown = "don't have that specific information" in answer
    answer = sanitize_for_speech(answer)
    rt.record({
        "heard": heard, "answer": answer, "sources": sources,
        "unknown": unknown, "model": rt.get("llm_model"),
    })
    audio = await asyncio.to_thread(synthesize, answer)
    logger.info(f"[voice] answered in {len(audio)} bytes of audio")

    return VoiceResponse(
        heard=heard,
        answer=answer,
        sources=sources,
        audio_b64=base64.b64encode(audio).decode(),
        unknown=unknown,
    )


@app.post("/reindex")
async def reindex():
    """Rebuild the BM25 index -- call after ingesting new documents."""
    count = await asyncio.to_thread(build_lexical_index)
    return {"status": "ok", "chunks": count}


@app.post("/debug/retrieve")
async def debug_retrieve(req: ChatRequest):
    """Show what hybrid retrieval returns for a query, without generating."""
    context, sources, relevant, score = await asyncio.to_thread(retrieve, req.query)
    return {
        "query": req.query,
        "relevant": relevant,
        "best_score": score,
        "sources": sources,
        "chunks": [c.strip()[:300] for c in context.split("\n\n---\n\n")],
    }
