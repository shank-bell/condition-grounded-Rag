"""Follow-up resolution: a question that leans on the chat is rewritten to stand alone before Stage 1; everything else is left alone."""
from types import SimpleNamespace

import pytest

from cgrag.config import get_settings
from cgrag.pipeline.followup import Standalone, needs_chat, resolve, resolve_followup, restore_names, transcript
from cgrag.pipeline.generate import ELABORATE, generate
from cgrag.pipeline.run import Pipeline

CHAT = [
    {"role": "user", "content": "How well do models perform on Kannada NLI?"},
    {"role": "assistant", "content": "On IndicXNLI, mBERT reaches 58.6 and MuRIL 74.0 [1]."},
]


class StubLLM:
    def __init__(self, answer=None, error=None, more_detail=False):
        self.answer, self.error, self.more_detail, self.prompts = answer, error, more_detail, []

    def structured(self, prompt, model_cls, **kwargs):
        self.prompts.append(prompt)
        if self.error:
            raise self.error
        return Standalone(question=self.answer, more_detail=self.more_detail), None


@pytest.mark.parametrize("question", ["Can you detail More On it????", "And for Hindi?", "Why?", "What about the same on XNLI, and why do they differ?"])
def test_a_question_that_points_back_may_need_the_chat(question):
    assert needs_chat(question)


@pytest.mark.parametrize("question", ["What F1 does BERT-large reach on SQuAD v1.1?", "Compare mBERT and XLM-R on XNLI for Hindi with zero-shot transfer"])
def test_a_question_that_stands_alone_does_not(question):
    assert not needs_chat(question)


def test_the_transcript_keeps_the_last_turns_and_cuts_long_ones():
    long_chat = [{"role": "user", "content": f"question {i}"} for i in range(6)] + [{"role": "assistant", "content": "x" * 900}]
    lines = transcript(long_chat).splitlines()
    assert len(lines) == 4 and lines[-1].startswith("Assistant: ") and len(lines[-1]) == len("Assistant: ") + 600
    assert lines[0] == "User: question 3"
    assert transcript([{"role": "user", "content": "  "}, {"role": "assistant"}]) == ""


def test_a_follow_up_is_rewritten_with_the_chat_in_the_prompt():
    llm = StubLLM("Can you give more detail on how models perform on Kannada NLI?")
    assert resolve_followup("Can you detail More On it????", CHAT, llm) == "Can you give more detail on how models perform on Kannada NLI?"
    assert "How well do models perform on Kannada NLI?" in llm.prompts[0] and "Last message: Can you detail More On it????" in llm.prompts[0]


def test_without_a_chat_the_model_is_not_asked():
    llm = StubLLM(error=AssertionError("must not be called"))
    assert resolve_followup("Can you detail More On it????", None, llm) == "Can you detail More On it????"
    assert resolve_followup("Can you detail More On it????", [], llm) == "Can you detail More On it????"
    assert llm.prompts == []


@pytest.mark.parametrize("bad", ["", "   ", "line one\nline two", "x" * 401])
def test_an_unusable_rewrite_leaves_the_question_unchanged(bad):
    assert resolve_followup("And for Hindi?", CHAT, StubLLM(bad)) == "And for Hindi?"


def test_a_model_error_leaves_the_question_unchanged():
    assert resolve_followup("And for Hindi?", CHAT, StubLLM(error=ValueError("not json"))) == "And for Hindi?"
    assert resolve_followup("And for Hindi?", CHAT, StubLLM(error=TimeoutError("slow"))) == "And for Hindi?"


def test_quotes_around_the_rewrite_are_removed():
    assert resolve_followup("And for Hindi?", CHAT, StubLLM('"How well do models perform on Hindi NLI?"')) == "How well do models perform on Hindi NLI?"


def test_the_wish_for_more_detail_is_read_with_the_rewrite():
    resolved = resolve("Can you detail More On it????", CHAT, StubLLM("Can you give more detail on Kannada NLI?", more_detail=True))
    assert resolved.question == "Can you give more detail on Kannada NLI?" and resolved.more_detail
    assert not resolve("And for Hindi?", CHAT, StubLLM("How well do models perform on Hindi NLI?")).more_detail


@pytest.mark.parametrize("bad", ["", "line one\nline two", "x" * 401])
def test_an_unusable_rewrite_drops_the_wish_for_more_detail(bad):
    resolved = resolve("Can you detail More On it????", CHAT, StubLLM(bad, more_detail=True))
    assert resolved == ("Can you detail More On it????", False)
    assert resolve("Can you detail More On it????", CHAT, StubLLM(error=ValueError("not json"))).more_detail is False
    assert resolve("Can you detail More On it????", [], StubLLM(error=AssertionError("must not be called"))).more_detail is False


def test_a_name_the_model_misspelled_is_put_back_as_the_chat_wrote_it():
    chat = "Assistant: IndicBERT+Samanantar reaches 74.7 and XLM-R 71.5 on IndicXNLI."
    assert restore_names("What did IndicBERT+Samanyar score on IndicXNLI?", chat, "Detail it") == "What did IndicBERT+Samanantar score on IndicXNLI?"
    assert restore_names("How do XLM-R and IndicBERT+Samanyar compare?", chat, "") == "How do XLM-R and IndicBERT+Samanantar compare?"


def test_numbers_new_names_and_correct_names_are_left_alone():
    chat = "Assistant: IndicBERT+Samanantar reaches 74.7 and XLM-R 71.5 on IndicXNLI."
    same = "How does IndicBERT+Samanantar at 74.8 compare to RoBERTa on SQuAD v2.0?"
    assert restore_names(same, chat, "") == same
    assert restore_names("Which of them is best?", chat, "") == "Which of them is best?"


def test_the_rewrite_goes_through_the_name_check():
    llm = StubLLM("What did IndicBERT+Samanyar score on IndicXNLI?")
    chat = CHAT + [{"role": "assistant", "content": "IndicBERT+Samanantar reaches 74.7 on IndicXNLI."}]
    assert resolve("And for it?", chat, llm).question == "What did IndicBERT+Samanantar score on IndicXNLI?"


class Writer:
    def __init__(self):
        self.messages = None

    def chat(self, messages, **kwargs):
        self.messages = messages
        return SimpleNamespace(text="Answer.")


def test_a_request_for_more_detail_tells_the_writer_not_to_repeat_and_allows_more_words():
    plain, more = Writer(), Writer()
    generate("What F1 does BERT-large reach?", [], plain, history=CHAT)
    generate("Can you give more detail on Kannada NLI?", [], more, history=CHAT, elaborate=True)
    assert "at most 180 words" in plain.messages[0]["content"] and ELABORATE not in plain.messages[-1]["content"]
    assert "at most 300 words" in more.messages[0]["content"] and ELABORATE in more.messages[-1]["content"]
    assert more.messages[1:-1] == CHAT


class Stop(Exception):
    pass


def pipeline(llm, **features):
    settings = get_settings()
    settings = settings.model_copy(update={"features": settings.features.model_copy(update=features)})
    seen = []

    def analyze(question):
        seen.append(question)
        raise Stop

    p = Pipeline.__new__(Pipeline)
    p.cfg = settings
    p.vectors = SimpleNamespace(count=lambda: 1)
    p.refine_llm = llm
    p.understand = SimpleNamespace(analyze=analyze)
    return p, seen


def test_stage_1_receives_the_rewritten_question_and_the_page_is_told():
    llm = StubLLM("Can you give more detail on how models perform on Kannada NLI?")
    p, seen = pipeline(llm)
    events = []
    with pytest.raises(Stop):
        p.run("Can you detail More On it????", CHAT, progress=events.append)
    assert seen == ["Can you give more detail on how models perform on Kannada NLI?"]
    steps = [(e["id"], e["status"]) for e in events if e["type"] == "step"]
    assert steps[:2] == [("1a_followup", "running"), ("1a_followup", "done")]
    assert any(e["type"] == "note" and e["stage"] == 1 and e["text"].startswith("follow-up read as: Can you give more detail") for e in events)


def test_the_page_is_told_when_a_follow_up_asks_for_more_detail():
    llm = StubLLM("Can you give more detail on how models perform on Kannada NLI?", more_detail=True)
    p, seen = pipeline(llm)
    events = []
    with pytest.raises(Stop):
        p.run("Can you detail More On it????", CHAT, progress=events.append)
    assert any(e["type"] == "note" and e["stage"] == 1 and e["text"].startswith("asks for more detail") for e in events)
    p, seen = pipeline(llm)
    events = []
    with pytest.raises(Stop):
        p.run("Can you detail More On it????", CHAT[:1], progress=events.append)
    assert not any(e["type"] == "note" and e["text"].startswith("asks for more detail") for e in events)


def test_a_question_that_stands_alone_is_skipped_with_a_reason():
    llm = StubLLM(error=AssertionError("must not be called"))
    p, seen = pipeline(llm)
    events = []
    with pytest.raises(Stop):
        p.run("What F1 does BERT-large reach on SQuAD v1.1?", CHAT, progress=events.append)
    assert seen == ["What F1 does BERT-large reach on SQuAD v1.1?"]
    skipped = [e for e in events if e["type"] == "step" and e["status"] == "skipped"]
    assert [e["id"] for e in skipped] == ["1a_followup"] and "stands on its own" in skipped[0]["reason"]


def test_the_first_question_of_a_chat_and_the_switch_leave_the_step_out():
    for history, features in ((None, {}), ([], {}), (CHAT, {"followup_rewrite": False})):
        p, seen = pipeline(StubLLM(error=AssertionError("must not be called")), **features)
        events = []
        with pytest.raises(Stop):
            p.run("Can you detail More On it????", history, progress=events.append)
        assert seen == ["Can you detail More On it????"]
        assert not [e for e in events if e.get("id") == "1a_followup"]
