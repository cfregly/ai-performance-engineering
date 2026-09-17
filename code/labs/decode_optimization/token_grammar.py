"""Byte-DFA token masking with an exact uncached path and state-mask cache.

Tokens may span multiple grammar edges or split UTF-8 characters. This implements
regular/finite-language constrained decoding; it is not a general CFG parser or
a replacement for XGrammar. The caller supplies the tokenizer's actual bytes.
"""

from types import MappingProxyType


class TokenGrammar:
    def __init__(self, transitions, accepting, vocabulary, *, eos_token_id, cache=True):
        self.transitions = tuple(MappingProxyType(dict(edges)) for edges in transitions)
        self.accepting = frozenset(accepting)
        self.vocabulary = tuple(vocabulary)
        self.eos_token_id = eos_token_id
        self.cache = cache
        self._masks = {}
        states = set(range(len(self.transitions)))
        if not states or not self.accepting or not self.accepting <= states:
            raise ValueError("grammar needs states and valid accepting states")
        if type(eos_token_id) is not int or eos_token_id not in range(len(self.vocabulary)):
            raise ValueError("eos_token_id is outside the vocabulary")
        for edges in self.transitions:
            if any(
                type(byte) is not int or byte not in range(256) or target not in states
                for byte, target in edges.items()
            ):
                raise ValueError("transitions must map byte values to valid states")
        if any(
            not isinstance(token, bytes) or (not token and i != eos_token_id)
            for i, token in enumerate(self.vocabulary)
        ):
            raise ValueError("vocabulary entries must be nonempty bytes except EOS")
        # Do not admit tokens that lead into a state from which acceptance is
        # impossible, even when the supplied DFA includes nonproductive edges.
        live = set(self.accepting)
        while True:
            extended = live | {
                s
                for s, edges in enumerate(self.transitions)
                if any(t in live for t in edges.values())
            }
            if extended == live:
                break
            live = extended
        self.live_states = frozenset(live)

    @classmethod
    def literals(cls, alternatives, vocabulary, *, eos_token_id, cache=True):
        transitions = [{}]
        accepting = set()
        for alternative in alternatives:
            if not isinstance(alternative, bytes):
                raise ValueError("literal alternatives must be bytes")
            state = 0
            for byte in alternative:
                if byte not in transitions[state]:
                    transitions[state][byte] = len(transitions)
                    transitions.append({})
                state = transitions[state][byte]
            accepting.add(state)
        return cls(transitions, accepting, vocabulary, eos_token_id=eos_token_id, cache=cache)

    def _destination(self, state, token_id):
        if token_id == self.eos_token_id:
            return -1 if state in self.accepting else None
        for byte in self.vocabulary[token_id]:
            state = self.transitions[state].get(byte)
            if state is None:
                return None
        return state if state in self.live_states else None

    def allowed_mask(self, state):
        if type(state) is not int or state < -1 or state >= len(self.transitions):
            raise ValueError("invalid grammar state")
        if state == -1:
            return (False,) * len(self.vocabulary)
        if self.cache and state in self._masks:
            return self._masks[state]
        mask = tuple(
            self._destination(state, token_id) is not None
            for token_id in range(len(self.vocabulary))
        )
        if self.cache:
            self._masks[state] = mask
        return mask

    def advance(self, state, token_id):
        if type(token_id) is not int or token_id not in range(len(self.vocabulary)):
            raise ValueError("invalid token ID")
        if not self.allowed_mask(state)[token_id]:
            raise ValueError("token violates the grammar or follows EOS")
        return self._destination(state, token_id)

    def precompile(self):
        if not self.cache:
            raise ValueError("precompile requires the cached backend")
        for state in range(len(self.transitions)):
            self.allowed_mask(state)

    def mask_logits(self, logits, state):
        """Apply the grammar to real PyTorch logits without changing allowed odds."""
        import torch

        if (
            logits.ndim != 1
            or logits.numel() != len(self.vocabulary)
            or not logits.is_floating_point()
        ):
            raise ValueError("logits must be a floating vector matching the vocabulary")
        allowed = self.allowed_mask(state)
        mask = torch.tensor(allowed, device=logits.device, dtype=torch.bool)
        if not any(allowed):
            raise ValueError("no valid tokens in this state")
        return logits.masked_fill(~mask, -torch.inf)
