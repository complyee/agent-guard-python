"""The two I/O shells around core.authorize: one sync (httpx.Client), one async
(httpx.AsyncClient). They perform operations and nothing else."""

from __future__ import annotations

import asyncio
import time
from typing import Any, Awaitable, Callable, Optional

import httpx

from .config import GuardConfig
from .core import AcrRequestError, Call, FetchJwks, Flow, Sleep, gateway_request, gateway_result, should_retry


def _json(res: httpx.Response) -> Any:
    try:
        return res.json()
    except ValueError:
        return {}


class SyncDriver:
    def __init__(self, config: GuardConfig, client: Optional[httpx.Client], sleep: Callable[[float], None]):
        self.config = config
        self._client = client
        self.sleep = sleep

    @property
    def client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(timeout=self.config.timeout_s)
        return self._client

    def call(self, op: Call) -> dict:
        url, headers, body = gateway_request(self.config, op)
        attempt = 0
        while True:
            try:
                res = self.client.post(url, headers=headers, json=body)
            except httpx.HTTPError as error:
                if should_retry(None, attempt, op):
                    attempt += 1
                    continue
                raise AcrRequestError(f"Agent Control Room unreachable: {error}") from None
            if should_retry(res.status_code, attempt, op):
                attempt += 1
                continue
            return gateway_result(res.status_code, _json(res))

    def perform(self, op: Any) -> Any:
        if isinstance(op, Call):
            return self.call(op)
        if isinstance(op, Sleep):
            return self.sleep(op.seconds)
        if isinstance(op, FetchJwks):
            res = self.client.get(op.url, headers={"accept": "application/json"})
            if res.status_code != 200:
                raise RuntimeError(f"JWK Set answered {res.status_code}")
            return _json(res)
        raise TypeError(f"unknown operation {op!r}")

    def run(self, flow: Flow) -> Any:
        value: Any = None
        error: Optional[BaseException] = None
        while True:
            try:
                op = flow.throw(error) if error else flow.send(value)
            except StopIteration as done:
                return done.value
            try:
                value, error = self.perform(op), None
            except Exception as e:  # handed back to the flow, which decides what it means
                value, error = None, e


class AsyncDriver:
    def __init__(self, config: GuardConfig, client: Optional[httpx.AsyncClient], sleep: Callable[[float], Awaitable[None]]):
        self.config = config
        self._client = client
        self.sleep = sleep

    async def call(self, client: httpx.AsyncClient, op: Call) -> dict:
        url, headers, body = gateway_request(self.config, op)
        attempt = 0
        while True:
            try:
                res = await client.post(url, headers=headers, json=body)
            except httpx.HTTPError as error:
                if should_retry(None, attempt, op):
                    attempt += 1
                    continue
                raise AcrRequestError(f"Agent Control Room unreachable: {error}") from None
            if should_retry(res.status_code, attempt, op):
                attempt += 1
                continue
            return gateway_result(res.status_code, _json(res))

    async def perform(self, client: httpx.AsyncClient, op: Any) -> Any:
        if isinstance(op, Call):
            return await self.call(client, op)
        if isinstance(op, Sleep):
            return await self.sleep(op.seconds)
        if isinstance(op, FetchJwks):
            res = await client.get(op.url, headers={"accept": "application/json"})
            if res.status_code != 200:
                raise RuntimeError(f"JWK Set answered {res.status_code}")
            return _json(res)
        raise TypeError(f"unknown operation {op!r}")

    async def run(self, flow: Flow) -> Any:
        # An injected client is reused; otherwise one short-lived client per run,
        # which keeps the guard safe across event loops.
        if self._client is not None:
            return await self._drive(self._client, flow)
        async with httpx.AsyncClient(timeout=self.config.timeout_s) as client:
            return await self._drive(client, flow)

    async def _drive(self, client: httpx.AsyncClient, flow: Flow) -> Any:
        value: Any = None
        error: Optional[BaseException] = None
        while True:
            try:
                op = flow.throw(error) if error else flow.send(value)
            except StopIteration as done:
                return done.value
            try:
                value, error = await self.perform(client, op), None
            except Exception as e:
                value, error = None, e


default_sync_sleep = time.sleep
default_async_sleep = asyncio.sleep
