"""Virtual Token Counter admission with linear and heap selection backends.

Counter lift, immediate input-token charging and observed output-token charging
follow Algorithm 2 of https://arxiv.org/abs/2401.00588. The model runner remains
responsible for memory admission, continuous batching and token generation.
"""

import heapq
import math
from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True)
class Request:
    request_id: int
    client_id: int
    input_tokens: int

    def __post_init__(self):
        if any(
            type(v) is not int or v < 0
            for v in (self.request_id, self.client_id, self.input_tokens)
        ):
            raise ValueError("request/client IDs and input_tokens must be nonnegative integers")


class VirtualTokenScheduler:
    def __init__(
        self, *, backend: str = "heap", input_weight: float = 1.0, output_weight: float = 2.0
    ):
        if backend not in {"linear", "heap"}:
            raise ValueError("backend must be linear or heap")
        if any(not math.isfinite(w) or w <= 0 for w in (input_weight, output_weight)):
            raise ValueError("token weights must be finite and positive")
        self.backend = backend
        self.input_weight = input_weight
        self.output_weight = output_weight
        self.counters = {}
        self.queues = {}
        self.running = {}
        self.pending_ids = set()
        self.heap = []
        self.versions = {}
        self.sequence = 0
        self.last_departed_client = None

    def _publish(self, client):
        version = self.versions.get(client, 0) + 1
        self.versions[client] = version
        if self.backend == "heap" and client in self.queues:
            heapq.heappush(
                self.heap, (self.counters[client], self.queues[client][0][0], client, version)
            )
            # Lazy invalidation must not accumulate an unbounded history when a
            # running client's token feedback updates a still-queued client.
            if len(self.heap) > 4 * len(self.queues) + 64:
                self.heap = [
                    (self.counters[c], q[0][0], c, self.versions[c]) for c, q in self.queues.items()
                ]
                heapq.heapify(self.heap)

    def _peek_client(self):
        if not self.queues:
            raise IndexError("no pending requests")
        if self.backend == "linear":
            return min(self.queues, key=lambda c: (self.counters[c], self.queues[c][0][0], c))
        while self.heap:
            _, _, client, version = self.heap[0]
            if client in self.queues and self.versions[client] == version:
                return client
            heapq.heappop(self.heap)
        raise RuntimeError("heap and pending queues disagree")

    def enqueue(self, request: Request):
        if request.request_id in self.pending_ids or request.request_id in self.running:
            raise ValueError("duplicate active request ID")
        client = request.client_id
        if client not in self.queues:
            if self.queues:
                floor = self.counters[self._peek_client()]
            elif self.last_departed_client is not None:
                floor = self.counters[self.last_departed_client]
            else:
                floor = 0.0
            self.counters[client] = max(self.counters.get(client, 0.0), floor)
            self.queues[client] = deque()
        self.queues[client].append((self.sequence, request))
        self.sequence += 1
        self.pending_ids.add(request.request_id)
        self._publish(client)

    def peek(self) -> Request:
        """Inspect the next request before the caller checks available KV memory."""
        return self.queues[self._peek_client()][0][1]

    def admit(self) -> Request:
        client = self._peek_client()
        _, request = self.queues[client].popleft()
        self.pending_ids.remove(request.request_id)
        self.running[request.request_id] = request
        self.counters[client] += self.input_weight * request.input_tokens
        if not self.queues[client]:
            del self.queues[client]
            self.last_departed_client = client
        self._publish(client)
        return request

    def record_output(self, request_id: int, tokens: int = 1):
        """Charge tokens actually produced; output lengths are not predicted."""
        if type(tokens) is not int or tokens < 0:
            raise ValueError("observed output tokens must be a nonnegative integer")
        request = self.running[request_id]
        self.counters[request.client_id] += self.output_weight * tokens
        self._publish(request.client_id)

    def complete(self, request_id: int):
        del self.running[request_id]

    def __len__(self):
        return len(self.pending_ids)


def dispatch_trace(requests, observed_outputs, *, backend="heap"):
    """Run actual policy operations with explicit post-admission feedback.

    This measures scheduler decisions, not model execution or request latency.
    """
    scheduler = VirtualTokenScheduler(backend=backend)
    for request in requests:
        scheduler.enqueue(request)
    order = []
    while scheduler:
        request = scheduler.admit()
        order.append(request.request_id)
        scheduler.record_output(request.request_id, observed_outputs[request.request_id])
        scheduler.complete(request.request_id)
    return tuple(order)
