from __future__ import annotations

import argparse
import json
import os
import re
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any


ZHIPU_SEARCH_URL = "https://open.bigmodel.cn/api/paas/v4/web_search"
ZHIPU_READER_URL = "https://open.bigmodel.cn/api/paas/v4/reader"
DEFAULT_ZHIPU_API_KEY = ""


@dataclass
class SearchItem:
    source_id: str
    title: str
    url: str
    publish_date: str
    content: str
    engine: str


def _clean_text(text: str) -> str:
    text = str(text or "")
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n", text)
    return text.strip()


class FastWebSearch:
    def __init__(self, args: argparse.Namespace, deepseek_api_key: str):
        self.args = args
        self.deepseek_api_key = deepseek_api_key
        self.zhipu_api_key = (
            getattr(args, "zhipu_api_key", "")
            or os.environ.get("ZHIPU_API_KEY", "")
            or DEFAULT_ZHIPU_API_KEY
        )

    def enabled(self) -> bool:
        return bool(getattr(self.args, "ai_web_search", True))

    def now(self) -> datetime:
        return datetime.now()

    def date_context(self) -> str:
        now = self.now()
        today = now.date()
        return (
            f"current_datetime={now.strftime('%Y-%m-%d %H:%M:%S')}, "
            f"today={today.isoformat()}, "
            f"yesterday={(today - timedelta(days=1)).isoformat()}, "
            f"tomorrow={(today + timedelta(days=1)).isoformat()}, "
            "timezone=Asia/Shanghai"
        )

    def post_json(self, url: str, payload: dict[str, Any], bearer: str, timeout: float) -> dict[str, Any]:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=body,
            headers={"Authorization": f"Bearer {bearer}", "Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            text = resp.read().decode("utf-8")
        data = json.loads(text)
        return data if isinstance(data, dict) else {}

    def deepseek(self, messages: list[dict[str, str]], max_tokens: int = 900) -> str:
        payload = {
            "model": getattr(self.args, "deepseek_model", "deepseek-v4-flash"),
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": max_tokens,
            "stream": False,
            "thinking": {"type": "disabled"},
        }
        url = str(getattr(self.args, "deepseek_base_url", "https://api.deepseek.com")).rstrip("/") + "/chat/completions"
        data = self.post_json(
            url,
            payload,
            self.deepseek_api_key,
            float(getattr(self.args, "ai_web_deepseek_timeout", 20.0)),
        )
        choices = data.get("choices") or []
        if not choices:
            return ""
        message = choices[0].get("message") or {}
        return str(message.get("content") or "").strip()

    def extract_json_object(self, text: str) -> dict:
        raw = str(text or "").strip()
        if raw.startswith("```"):
            raw = raw.strip("`").strip()
            if raw.lower().startswith("json"):
                raw = raw[4:].strip()
        try:
            data = json.loads(raw)
            return data if isinstance(data, dict) else {}
        except json.JSONDecodeError:
            pass
        start = raw.find("{")
        end = raw.rfind("}")
        if start >= 0 and end > start:
            try:
                data = json.loads(raw[start : end + 1])
                return data if isinstance(data, dict) else {}
            except json.JSONDecodeError:
                return {}
        return {}

    def expand_query(self, query: str) -> str:
        variants: list[str] = []
        for year, month, day in re.findall(r"(20\d{2})年(\d{1,2})月(\d{1,2})日?", query):
            variants.append(f"{year}-{int(month):02d}-{int(day):02d}")
        for year, month, day in re.findall(r"(20\d{2})-(\d{1,2})-(\d{1,2})", query):
            variants.append(f"{year}年{int(month)}月{int(day)}日")
        for variant in variants:
            if variant not in query:
                query = f"{query} {variant}"
        return query.strip()

    def relevance_terms(self, plan: dict[str, Any], question: str) -> set[str]:
        text = " ".join(
            [
                question,
                str(plan.get("search_query") or ""),
                " ".join(str(item) for item in plan.get("must_include") or []),
            ]
        ).lower()
        terms: set[str] = set()
        for token in re.findall(r"[a-z0-9][a-z0-9_.-]{1,}", text):
            terms.add(token)
        for block in re.findall(r"[\u4e00-\u9fff]{2,}", text):
            if len(block) <= 4:
                terms.add(block)
            else:
                for size in (2, 3, 4):
                    for index in range(0, len(block) - size + 1):
                        terms.add(block[index : index + size])
        return {term for term in terms if term not in {"2026", "2025", "2024"}}

    def source_score(self, item: SearchItem, terms: set[str], plan: dict[str, Any], question: str) -> float:
        haystack = f"{item.title} {item.publish_date} {item.content}".lower()
        score = 0.0
        for term in terms:
            if term in haystack:
                score += min(2.0, 0.35 + len(term) * 0.12)
        dates = set(re.findall(r"20\d{2}[-年/.]\d{1,2}[-月/.]\d{1,2}", question + " " + str(plan.get("search_query") or "")))
        for raw_date in dates:
            normalized = raw_date.replace("年", "-").replace("月", "-").replace("日", "")
            parts = [part for part in re.split(r"[-/.]", normalized) if part]
            variants = {raw_date}
            if len(parts) == 3:
                year, month, day = parts
                variants.add(f"{year}-{int(month):02d}-{int(day):02d}")
                variants.add(f"{year}年{int(month)}月{int(day)}日")
                variants.add(f"{int(month)}月{int(day)}日")
            if any(variant in haystack for variant in variants):
                score += 4.0
        if item.url:
            score += 0.2
        return score

    def search_one(self, query: str, engine: str, recency: str) -> list[dict[str, Any]]:
        payload = {
            "search_query": query,
            "search_engine": engine,
            "search_intent": False,
            "count": int(getattr(self.args, "ai_web_search_count", 5)),
            "search_recency_filter": recency,
            "content_size": str(getattr(self.args, "ai_web_content_size", "medium")),
        }
        data = self.post_json(
            str(getattr(self.args, "zhipu_search_url", ZHIPU_SEARCH_URL)),
            payload,
            self.zhipu_api_key,
            float(getattr(self.args, "ai_web_search_timeout", 10.0)),
        )
        results = data.get("search_result") or []
        return results if isinstance(results, list) else []

    def search(self, question: str, plan: dict[str, Any]) -> list[SearchItem]:
        query = self.expand_query(str(plan.get("search_query") or question).strip())
        engines = [
            item.strip()
            for item in str(getattr(self.args, "ai_web_search_engines", "search_pro,search_pro_sogou,search_pro_quark")).split(",")
            if item.strip()
        ] or ["search_pro"]
        recency = str(plan.get("search_recency") or plan.get("recency") or "noLimit")
        if recency not in {"oneDay", "oneWeek", "oneMonth", "oneYear", "noLimit"}:
            recency = "noLimit"

        raw_by_engine: dict[str, list[dict[str, Any]]] = {engine: [] for engine in engines}
        workers = min(len(engines), int(getattr(self.args, "ai_web_max_parallel_searches", 3)))
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            futures = {pool.submit(self.search_one, query, engine, recency): engine for engine in engines}
            for future in as_completed(futures):
                engine = futures[future]
                try:
                    raw_by_engine[engine] = [result for result in future.result() if isinstance(result, dict)]
                except Exception as exc:
                    print(f"[ai-web] search failed engine={engine} error={exc}", flush=True)

        all_items: list[SearchItem] = []
        seen: set[str] = set()
        for engine in engines:
            for result in raw_by_engine.get(engine, []):
                title = str(result.get("title") or "").strip()
                url = str(result.get("link") or result.get("url") or "").strip()
                content = _clean_text(str(result.get("content") or result.get("snippet") or ""))
                key = url or title
                if not key or key in seen or not content:
                    continue
                seen.add(key)
                all_items.append(
                    SearchItem(
                        source_id="",
                        title=title,
                        url=url,
                        publish_date=str(result.get("publish_date") or result.get("date") or ""),
                        content=content,
                        engine=engine,
                    )
                )

        terms = self.relevance_terms(plan, question)
        all_items.sort(key=lambda item: self.source_score(item, terms, plan, question), reverse=True)
        items = all_items[: int(getattr(self.args, "ai_web_max_sources", 10))]
        for index, item in enumerate(items, 1):
            item.source_id = f"S{index}"
        return items

    def sources_payload(self, sources: list[SearchItem]) -> list[dict[str, str]]:
        limit = int(getattr(self.args, "ai_web_max_snippet_chars", 700))
        return [
            {
                "id": item.source_id,
                "title": item.title[:140],
                "url": item.url,
                "publish_date": item.publish_date,
                "engine": item.engine,
                "snippet": item.content[:limit],
            }
            for item in sources
        ]

    def answer_from_search(self, question: str, plan: dict[str, Any], sources: list[SearchItem]) -> dict[str, Any]:
        prompt = f"""
Answer a Chinese voice-assistant question using the supplied web search snippets.
Return strict JSON only.

Context: {self.date_context()}
Question: {question}
Plan: {json.dumps(plan, ensure_ascii=False)}
Search snippets:
{json.dumps(self.sources_payload(sources), ensure_ascii=False)}

Rules:
- Directly answer in Chinese, suitable for speech playback.
- Use only local context and the snippets. Do not add unsupported facts.
- Every factual sentence that uses web snippets must cite source ids like [S1].
- Prefer snippets that directly contain the requested entity, date, and value.
- Ignore snippets that are fresh but only tangentially related to the question.
- If snippets are relevant but not enough and one source URL could clarify the
  answer, set need_reader=true and choose exactly one reader_url.
- If no source can support the answer, say evidence is insufficient rather than guessing.
- Resolve relative wording into concrete dates when the context makes it clear.

Schema:
{{"answer":"...","confidence":"high|medium|low","need_reader":false,"reader_url":"","used_sources":["S1"],"reason":"short reason"}}
"""
        content = self.deepseek(
            [
                {"role": "system", "content": "Return strict JSON only. No markdown."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=int(getattr(self.args, "ai_web_answer_tokens", 900)),
        )
        data = self.extract_json_object(content)
        if not str(data.get("answer") or "").strip():
            data["answer"] = content.strip()
        return self.ensure_citation(data)

    def read_page(self, url: str) -> dict[str, str]:
        payload = {
            "url": url,
            "timeout": int(getattr(self.args, "ai_web_reader_page_timeout", 8.0)),
            "return_format": "text",
            "retain_images": False,
            "no_cache": False,
        }
        data = self.post_json(
            str(getattr(self.args, "zhipu_reader_url", ZHIPU_READER_URL)),
            payload,
            self.zhipu_api_key,
            float(getattr(self.args, "ai_web_reader_timeout", 10.0)),
        )
        result = data.get("reader_result") or {}
        if not isinstance(result, dict):
            result = {}
        return {
            "title": str(result.get("title") or ""),
            "content": _clean_text(str(result.get("content") or ""))[: int(getattr(self.args, "ai_web_max_reader_chars", 2200))],
        }

    def answer_with_reader(
        self,
        question: str,
        plan: dict[str, Any],
        sources: list[SearchItem],
        first_answer: dict[str, Any],
        page: dict[str, str],
    ) -> dict[str, Any]:
        prompt = f"""
Use the search snippets and one read page to produce the final Chinese answer.
Return strict JSON only.

Context: {self.date_context()}
Question: {question}
Plan: {json.dumps(plan, ensure_ascii=False)}
Initial answer attempt: {json.dumps(first_answer, ensure_ascii=False)}
Search snippets:
{json.dumps(self.sources_payload(sources), ensure_ascii=False)}
Read page:
{json.dumps(page, ensure_ascii=False)}

Rules:
- Directly answer in Chinese, suitable for speech playback.
- Use only local context, snippets, and the read page.
- Cite source ids when using snippets, and cite [R1] when using the read page.
- If evidence is still insufficient, say so plainly.

Schema:
{{"answer":"...","confidence":"high|medium|low","used_sources":["S1","R1"],"reason":"short reason"}}
"""
        content = self.deepseek(
            [
                {"role": "system", "content": "Return strict JSON only. No markdown."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=int(getattr(self.args, "ai_web_answer_tokens", 900)),
        )
        data = self.extract_json_object(content)
        if not str(data.get("answer") or "").strip():
            data["answer"] = content.strip()
        return self.ensure_citation(data)

    def ensure_citation(self, answer: dict[str, Any]) -> dict[str, Any]:
        text = str(answer.get("answer") or "").strip()
        if not text or "[S" in text or "[R" in text:
            return answer
        used = answer.get("used_sources") or []
        if isinstance(used, list) and used:
            first = str(used[0]).strip()
            if first:
                answer = dict(answer)
                answer["answer"] = f"{text} [{first}]"
        return answer

    def answer(self, question: str, plan: dict[str, Any] | None = None) -> str:
        if not self.enabled():
            raise RuntimeError("AI web search is disabled")
        if not self.zhipu_api_key:
            raise RuntimeError("missing ZHIPU_API_KEY")
        if not self.deepseek_api_key:
            raise RuntimeError("missing DeepSeek API key")

        plan = dict(plan or {})
        if not str(plan.get("search_query") or "").strip():
            plan["search_query"] = question
        start = time.perf_counter()
        sources = self.search(question, plan)
        search_elapsed = time.perf_counter() - start
        first_answer = self.answer_from_search(question, plan, sources)
        final_answer = first_answer
        reader_url = str(first_answer.get("reader_url") or "").strip()
        if (
            bool(getattr(self.args, "ai_web_allow_reader", True))
            and bool(first_answer.get("need_reader"))
            and reader_url.startswith(("http://", "https://"))
        ):
            try:
                page = self.read_page(reader_url)
                if page.get("content"):
                    final_answer = self.answer_with_reader(question, plan, sources, first_answer, page)
            except Exception as exc:
                print(f"[ai-web] reader failed url={reader_url} error={exc}", flush=True)

        answer = str(final_answer.get("answer") or "").strip()
        titles = " | ".join(source.title[:40] for source in sources[:3])
        print(
            f"[ai-web] query={plan.get('search_query')} sources={len(sources)} "
            f"search_sec={search_elapsed:.2f} answer={answer[:80]} sources={titles}",
            flush=True,
        )
        return answer
