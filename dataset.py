"""Dataset loading for OpenThoughts."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional

from datasets import load_dataset


@dataclass
class DatasetSample:
    sample_id: str
    question: str
    answer: str = ""
    source: str = "unknown"
    domain: str = ""
    dataset_system: str = ""
    test_cases: Any = None
    starter_code: Any = None
    raw: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.raw:
            self.raw = _build_raw_payload(
                sample_id=self.sample_id,
                question=self.question,
                answer=self.answer,
                source=self.source,
                domain=self.domain,
                dataset_system=self.dataset_system,
                test_cases=self.test_cases,
                starter_code=self.starter_code,
            )

    def prompt_context(self) -> Dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "question": self.question,
            "dataset_system": self.dataset_system,
        }

    def to_dict(self) -> Dict[str, Any]:
        return _build_raw_payload(
            sample_id=self.sample_id,
            question=self.question,
            answer=self.answer,
            source=self.source,
            domain=self.domain,
            dataset_system=self.dataset_system,
            test_cases=self.test_cases,
            starter_code=self.starter_code,
        )

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "DatasetSample":
        return cls(
            sample_id=str(payload.get("sample_id") or payload.get("id") or ""),
            question=_coerce_text(payload.get("question")),
            answer=_coerce_text(payload.get("answer")),
            source=_coerce_text(payload.get("source")) or "unknown",
            domain=_coerce_text(payload.get("domain")),
            dataset_system=_coerce_text(payload.get("dataset_system")),
            test_cases=payload.get("test_cases"),
            starter_code=payload.get("starter_code"),
        )


class OpenThoughtsDataset:
    def __init__(
        self,
        samples_per_source: int,
        seed: int = 42,
        data_path: Optional[str] = None,
        source: str = "open-thoughts/OpenThoughts-114k",
        split: str = "train",
        streaming: bool = True,
        subset: str = "metadata",
    ) -> None:
        if samples_per_source <= 0:
            raise ValueError("samples_per_source must be a positive integer.")

        self.samples_per_source = samples_per_source
        self.seed = seed
        self.data_path = Path(data_path).expanduser() if data_path else None
        self.source = source
        self.split = split
        self.streaming = streaming
        self.subset = subset
        self.samples = self._load()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> DatasetSample:
        return self.samples[index]

    def __iter__(self) -> Iterator[DatasetSample]:
        return iter(self.samples)

    def _load(self) -> List[DatasetSample]:
        if self.data_path:
            if self.data_path.exists():
                return self._load_from_path(self.data_path)
            if self.data_path.suffix.lower() == ".json":
                samples = self._sample_remote_dataset()
                self._write_cache(self.data_path, samples)
                return samples
            raise FileNotFoundError(f"Dataset file not found: {self.data_path}")

        return self._sample_remote_dataset()

    def _load_from_path(self, path: Path) -> List[DatasetSample]:
        if path.suffix.lower() == ".json":
            return self._load_cache(path)
        if path.suffix.lower() == ".jsonl":
            return self._sample_records(_iter_jsonl_records(path))
        raise ValueError(f"Unsupported dataset file type: {path.suffix or '<none>'}")

    def _load_cache(self, path: Path) -> List[DatasetSample]:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError("Cached dataset JSON must contain a list of samples.")
        samples = [DatasetSample.from_dict(item) for item in payload if isinstance(item, dict)]
        if not samples:
            raise ValueError("The cached dataset file is empty.")
        return _sort_samples(samples)

    def _sample_remote_dataset(self) -> List[DatasetSample]:
        dataset_args = [self.source]
        if self.subset not in (None, "", "default"):
            dataset_args.append(self.subset)

        dataset = load_dataset(
            *dataset_args,
            split=self.split,
            streaming=self.streaming,
        )
        return self._sample_records(dataset)

    def _sample_records(self, records: Iterable[Dict[str, Any]]) -> List[DatasetSample]:
        samples_by_source: Dict[str, List[DatasetSample]] = {}
        counts_by_source: Dict[str, int] = {}
        rng_by_source: Dict[str, random.Random] = {}

        for index, record in enumerate(records):
            sample = normalize_sample(record, index)
            counts_by_source[sample.source] = counts_by_source.get(sample.source, 0) + 1
            bucket = samples_by_source.setdefault(sample.source, [])

            if len(bucket) < self.samples_per_source:
                bucket.append(sample)
                continue

            rng = rng_by_source.setdefault(
                sample.source,
                random.Random(f"{self.seed}:{sample.source}"),
            )
            replace_index = rng.randrange(counts_by_source[sample.source])
            if replace_index < self.samples_per_source:
                bucket[replace_index] = sample

        samples = [sample for bucket in samples_by_source.values() for sample in bucket]
        if not samples:
            raise ValueError("The dataset is empty.")
        return _sort_samples(samples)

    def _write_cache(self, path: Path, samples: List[DatasetSample]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        serialized = [sample.to_dict() for sample in samples]
        path.write_text(
            json.dumps(serialized, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def load_samples(
    samples_per_source: int,
    seed: int = 42,
    data_path: Optional[str] = None,
    source: str = "open-thoughts/OpenThoughts-114k",
    split: str = "train",
    streaming: bool = True,
    subset: str = "metadata",
) -> List[DatasetSample]:
    dataset = OpenThoughtsDataset(
        samples_per_source=samples_per_source,
        seed=seed,
        data_path=data_path,
        source=source,
        split=split,
        streaming=streaming,
        subset=subset,
    )
    return dataset.samples


def normalize_sample(record: Dict[str, Any], index: int) -> DatasetSample:
    sample_id = _coerce_text(
        _pick_value(record, "sample_id", "id")
    ) or str(index)
    question = _extract_question(record)
    answer = _extract_answer(record)
    source = _coerce_text(_pick_value(record, "source")) or "unknown"
    domain = _coerce_text(_pick_value(record, "domain"))
    dataset_system = _coerce_text(_pick_value(record, "system", "system_prompt"))
    test_cases = _pick_value(record, "test_cases")
    starter_code = _pick_value(record, "starter_code")

    return DatasetSample(
        sample_id=sample_id,
        question=question,
        answer=answer,
        source=source,
        domain=domain,
        dataset_system=dataset_system,
        test_cases=test_cases,
        starter_code=starter_code,
    )


def _sort_samples(samples: List[DatasetSample]) -> List[DatasetSample]:
    return sorted(samples, key=lambda sample: (sample.source, sample.sample_id))


def _iter_jsonl_records(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                yield json.loads(stripped)


def _build_raw_payload(
    *,
    sample_id: str,
    question: str,
    answer: str,
    source: str,
    domain: str,
    dataset_system: str,
    test_cases: Any,
    starter_code: Any,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "sample_id": sample_id,
        "question": question,
        "answer": answer,
        "source": source,
        "domain": domain,
        "dataset_system": dataset_system,
    }
    if test_cases is not None:
        payload["test_cases"] = test_cases
    if starter_code is not None:
        payload["starter_code"] = starter_code
    return payload


def _pick_value(record: Dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in record and record[key] not in (None, ""):
            return record[key]

    metadata = record.get("metadata")
    if isinstance(metadata, dict):
        for key in keys:
            if key in metadata and metadata[key] not in (None, ""):
                return metadata[key]

    return None


def _extract_question(record: Dict[str, Any]) -> str:
    for key in ("question", "problem", "prompt", "instruction"):
        value = _pick_value(record, key)
        text = _coerce_text(value)
        if text:
            return text

    conversations = record.get("conversations") or record.get("messages") or []
    if isinstance(conversations, list):
        for message in conversations:
            role = _message_role(message)
            text = _message_text(message)
            if role in ("user", "human") and text:
                return text
        for message in conversations:
            text = _message_text(message)
            if text:
                return text

    raise ValueError("Unable to extract the dataset question from the record.")


def _extract_answer(record: Dict[str, Any]) -> str:
    for key in ("answer", "ground_truth_solution", "deepseek_solution", "solution"):
        text = _coerce_text(_pick_value(record, key))
        if text:
            return text

    conversations = record.get("conversations") or record.get("messages") or []
    if isinstance(conversations, list):
        for message in conversations:
            role = _message_role(message)
            text = _message_text(message)
            if role in ("assistant", "model", "gpt", "bot") and text:
                return text

    return ""


def _message_role(message: Dict[str, Any]) -> str:
    if not isinstance(message, dict):
        return ""
    role = message.get("role") or message.get("from") or message.get("speaker")
    return str(role or "").strip().lower()


def _message_text(message: Dict[str, Any]) -> str:
    if not isinstance(message, dict):
        return ""
    content = message.get("content")
    if isinstance(content, list):
        chunks = []
        for block in content:
            if isinstance(block, dict):
                text = block.get("text")
                if isinstance(text, str) and text.strip():
                    chunks.append(text.strip())
        if chunks:
            return "\n".join(chunks)
    for key in ("value", "content", "text"):
        value = message.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _coerce_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value).strip()


# if __name__ == "__main__":
#     # Example usage
#     ds = OpenThoughtsDataset(samples_per_source=100, 
#                              seed=42,
#                              data_path="data/openthoughts_sample.json")
#     print("total samples:", len(ds))
#     print(ds[1].question)