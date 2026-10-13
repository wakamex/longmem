#!/usr/bin/env python3
"""ColBERT search that can return the matching part of each message instead of the whole of it.

MaxSim scores a message by summing, over the query's tokens, the best similarity to any token of
the message. The same table of similarities says how much of that score any part of the message
keeps: its coverage. `Index.search` ranks messages by the full score, and `excerpts` then picks
sentences greedily by coverage gained per character, either until each message keeps a set share
of its score or until one character budget for the whole result is spent. Coverage is a sum of
maxima (facility location), so it is monotone and submodular and greedy selection is near optimal.

Messages are embedded at up to MAX_TOKENS tokens, so text anywhere in a long message can match.
"""
import re

import torch

MODEL = "lightonai/ColBERT-Zero"
MAX_TOKENS = 8192
SENTENCE = re.compile(r"[^\n]+?(?:[.!?]+(?=\s|$)|\n|$)")
# Query tokens that match something in every sentence and so say nothing about where the answer is.
STOP = set("a an and are as at be by did do does for from had has have how i in is it me my of on or so that the their them "
           "they this to was we were what when where which who why with you your am been can could would should about "
           "any some there then than into our us if but not no".split())


def load():
    from pylate import models
    model = models.ColBERT(model_name_or_path=MODEL, device="cuda", document_length=MAX_TOKENS)
    model.half()
    return model


class Index:
    """The token vectors of one chat's messages, with the characters each vector stands for."""

    def __init__(self, model, texts, tag_chars):
        """`texts[k][:tag_chars[k]]` is a role tag that is ranked with the message but never excerpted."""
        self.model, self.texts, self.tags = model, texts, tag_chars
        prompt = model.prompts["document"]
        emb = model.encode(texts, is_query=False, prompt_name="document", batch_size=8, show_progress_bar=False, convert_to_tensor=False)
        self.emb = [torch.as_tensor(e).cuda() for e in emb]
        skip, tok = set(model.skiplist), model.tokenizer
        self.spans = []
        for text, tag, e in zip(texts, tag_chars, self.emb):
            # sentence-transformers strips the text before it tokenizes
            enc = tok((prompt + text).strip(), truncation=True, max_length=model.document_length - 1, return_offsets_mapping=True)
            ids = enc["input_ids"][:1] + [model.document_prefix_id] + enc["input_ids"][1:]
            offs = enc["offset_mapping"][:1] + [(0, 0)] + enc["offset_mapping"][1:]
            kept = [(a - len(prompt), b - len(prompt)) for i, (a, b) in zip(ids, offs) if i not in skip]
            if len(kept) != len(e):
                raise ValueError(f"{len(kept)} token spans for {len(e)} vectors")
            # tokens of the prompt, the role tag and the special tokens carry no text of the message
            self.spans.append([s if s[0] >= tag and s[1] > s[0] else None for s in kept])

    def query(self, text):
        """The query's vectors, and which of them count toward coverage."""
        m = self.model
        q = torch.as_tensor(m.encode([text], is_query=True, prompt_name="query", show_progress_bar=False)[0]).cuda()
        prompt = m.prompts["query"]
        enc = m.tokenizer((prompt + text).strip(), truncation=True, max_length=m.query_length - 1, return_offsets_mapping=True)
        offs = enc["offset_mapping"][:1] + [(0, 0)] + enc["offset_mapping"][1:]
        full = prompt + text
        keep = [b > a >= len(prompt) and full[a:b].strip().lower() not in STOP and any(c.isalnum() for c in full[a:b]) for a, b in offs]
        if len(keep) != len(q):
            raise ValueError(f"{len(keep)} query tokens for {len(q)} vectors")
        keep = torch.tensor(keep, device="cuda")
        return q, keep if keep.any() else torch.ones_like(keep)

    def search(self, text, top=10):
        """Indexes of the best messages, best first, with the query for `excerpts`."""
        q, keep = self.query(text)
        scores = torch.stack([(q @ e.T).max(-1).values.sum() for e in self.emb])
        return scores.float().topk(min(top, len(self.emb))).indices.tolist(), (q, keep)

    def _sentences(self, k, query):
        """Sentences of message k as (start, end, similarity table of its tokens), in order."""
        q, keep = query
        sim = (q[keep] @ self.emb[k].T).float().clamp(min=0)  # query tokens by message tokens
        text, tag = self.texts[k], self.tags[k]
        cuts = [(tag + s.start(), tag + s.end()) for s in SENTENCE.finditer(text[tag:]) if s.group(0).strip()]
        cols = [[] for _ in cuts]
        j = 0
        for t, span in enumerate(self.spans[k]):
            if span is None:
                continue
            while j + 1 < len(cuts) and span[0] >= cuts[j][1]:
                j += 1
            cols[j].append(t)
        zero = torch.zeros(sim.shape[0], device=sim.device)
        return [(a, b, sim[:, c].max(-1).values if c else zero) for (a, b), c in zip(cuts, cols)]

    def excerpts(self, hits, query, share=None, budget=None, neighbors=0):
        """The text to show for each hit: every sentence needed to keep `share` of the message's
        coverage, or the sentences that fit a total of `budget` characters over all hits."""
        sents = {k: self._sentences(k, query) for k in hits}
        have = {k: torch.zeros_like(sents[k][0][2]) if sents[k] else None for k in hits}
        full = {k: torch.stack([s[2] for s in sents[k]]).max(0).values.sum().item() if sents[k] else 0.0 for k in hits}
        chosen = {k: set() for k in hits}

        def best(k):
            """(gain per character, sentence) of the best sentence of message k not yet chosen."""
            cands = [((torch.maximum(have[k], v).sum() - have[k].sum()).item() / (b - a), n)
                     for n, (a, b, v) in enumerate(sents[k]) if n not in chosen[k]]
            return max(cands, key=lambda c: c[0]) if cands else (0.0, None)

        def take(k, n):
            chosen[k].add(n)
            have[k] = torch.maximum(have[k], sents[k][n][2])
            return sents[k][n][1] - sents[k][n][0]

        if share is not None:
            for k in hits:
                while sents[k] and have[k].sum().item() < share * full[k]:
                    gain, n = best(k)
                    if n is None or gain <= 0:
                        break
                    take(k, n)
        else:
            spent = sum(take(k, best(k)[1]) for k in hits if sents[k])  # every hit shows its best sentence
            while spent < budget:
                gain, k, n = max(((g, k, n) for k in hits if sents[k] for g, n in [best(k)]), key=lambda x: x[0], default=(0.0, None, None))
                if n is None or gain <= 0:
                    break
                spent += take(k, n)
        for k in hits:  # with neighbors, each chosen sentence brings the sentences next to it
            chosen[k] = {n + d for n in chosen[k] for d in range(-neighbors, neighbors + 1) if 0 <= n + d < len(sents[k])}
        out = {}
        for k in hits:
            text, parts, last = self.texts[k], [], self.tags[k]
            for n in sorted(chosen[k]):
                a, b, _ = sents[k][n]
                if text[last:a].strip():
                    parts.append("[...]")
                parts.append(text[a:b].strip())
                last = b
            if text[last:].strip():
                parts.append("[...]")
            out[k] = " ".join(parts) if chosen[k] else text[self.tags[k]:]
        return out
