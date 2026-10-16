import re

STOP = set("""a an the and or of to in on for is are was were be been with by at as it its this that
these those from what which who whom how when where why do does did can could should would will i you
we our your my me us they them their there here about into than then so if not no yes any all per may
have has had get gets""".split())


def tokenize(text: str) -> list[str]:
    toks = re.findall(r"[a-z0-9]+", text.lower())
    return [t[:-1] if len(t) > 3 and t.endswith("s") else t for t in toks if t not in STOP]
