"""Simple dataset loader."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from datasets import load_dataset


DATASETS = [
    # {
    #     "name": "OpenThoughts",
    #     "path": "open-thoughts/OpenThoughts-114k",
    #     "subset": "metadata",
    #     "split": "train",
    # },
    {
        "name": "MATH500",
        "path": "HuggingFaceH4/MATH-500",
        "split": "test",
    },
    {
        "name": "JEEBench",
        "path": "daman1209arora/jeebench",
        "split": "test",
    },
    {
        "name": "LiveCodeBench",
        "path": "livecodebench/code_generation_lite",
        "split": "test",
        "load_kwargs": {
            "trust_remote_code": True,
            "version_tag": "release_v2",
        },
    },
]

TRAIN_RATIO = 0.6


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

    def to_dict(self) -> Dict[str, Any]:
        item = {
            "sample_id": self.sample_id,
            "question": self.question,
            "answer": self.answer,
            "source": self.source,
            "domain": self.domain,
            "dataset_system": self.dataset_system,
        }

        if self.test_cases is not None:
            item["test_cases"] = self.test_cases

        if self.starter_code is not None:
            item["starter_code"] = self.starter_code

        return item

    @classmethod
    def from_dict(cls, item: Dict[str, Any]) -> "DatasetSample":
        return cls(
            sample_id=str(item.get("sample_id") or ""),
            question=str(item.get("question") or ""),
            answer=str(item.get("answer") or ""),
            source=str(item.get("source") or "unknown"),
            domain=str(item.get("domain") or ""),
            dataset_system=str(item.get("dataset_system") or ""),
            test_cases=item.get("test_cases"),
            starter_code=item.get("starter_code"),
        )


class OpenThoughtsDataset:
    def __init__(
        self,
        samples_per_source: int,
        seed: int = 42,
        data_path: Optional[str] = None,
    ) -> None:
        # samples_per_source only controls OpenThoughts.
        self.open_thoughts_samples = samples_per_source
        self.seed = seed
        self.data_path = Path(data_path) if data_path else None
        self.samples = self.load()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> DatasetSample:
        return self.samples[index]

    def __iter__(self) -> Iterator[DatasetSample]:
        return iter(self.samples)

    def load(self) -> List[DatasetSample]:
        if self.data_path and self.data_path.exists():
            return load_json(self.data_path)

        split_data = build_train_test_json(
            open_thoughts_samples=self.open_thoughts_samples,
            seed=self.seed,
        )

        if self.data_path:
            name = self.data_path.stem.lower()

            if "train" in name:
                samples = split_data["train"]
            elif "test" in name:
                samples = split_data["test"]
            else:
                samples = split_data["train"] + split_data["test"]

            write_json(self.data_path, samples)
            return samples

        return split_data["train"] + split_data["test"]


def load_samples(
    samples_per_source: int,
    seed: int = 42,
    data_path: Optional[str] = None,
) -> List[DatasetSample]:
    return OpenThoughtsDataset(
        samples_per_source=samples_per_source,
        seed=seed,
        data_path=data_path,
    ).samples


def build_train_test_json(seed: int, spec) -> Dict[str, List[DatasetSample]]:
    train_samples = []
    test_samples = []

    dataset_name = spec["name"]
    print(f"Loading {dataset_name}...")

    dataset_args = [spec["path"]]
    if spec.get("subset"):
        dataset_args.append(spec["subset"])

    dataset = load_dataset(
        *dataset_args,
        split=spec["split"],
        **spec.get("load_kwargs", {}),
    )

    samples = []

    for index, record in enumerate(dataset):

        record = dict(record)

        try:
            samples.append(normalize_sample(record, index))
        except Exception as e:
            print(f"Skip bad sample from {dataset_name}: {e}")

    random.seed(seed)
    random.shuffle(samples)

    split_index = int(len(samples) * TRAIN_RATIO)

    train_samples.extend(samples[:split_index])
    test_samples.extend(samples[split_index:])

    print(f"{dataset_name}: total={len(samples)}, train={split_index}, test={len(samples) - split_index}")

    train_samples.sort(key=lambda x: (x.source, x.sample_id))
    test_samples.sort(key=lambda x: (x.source, x.sample_id))

    return {
        "train": train_samples,
        "test": test_samples,
    }


def normalize_sample(record: Dict[str, Any], index: int) -> DatasetSample:
    source = str(get_value(record, "source") or "unknown").strip()

    sample_id = str(
        get_value(record, "sample_id")
        or get_value(record, "id")
        or get_value(record, "question_id")
        or get_value(record, "problem_id")
        or index
    ).strip()

    if not sample_id.startswith(f"{source}:"):
        sample_id = f"{sample_id}"

    return DatasetSample(
        sample_id=sample_id,
        question=extract_question(record),
        answer=extract_answer(record),
        source=source,
        domain=str(get_value(record, "domain") or "").strip(),
        dataset_system=str(
            get_value(record, "system")
            or get_value(record, "system_prompt")
            or ""
        ).strip(),
        test_cases=(
            get_value(record, "test_cases")
            or get_value(record, "public_test_cases")
            or get_value(record, "private_test_cases")
        ),
        starter_code=get_value(record, "starter_code"),
    )


def extract_question(record: Dict[str, Any]) -> str:
    for key in ["question", "problem", "prompt", "instruction", "question_content"]:
        text = str(get_value(record, key) or "").strip()
        if text:
            title = str(
                get_value(record, "question_title")
                or get_value(record, "title")
                or ""
            ).strip()

            options = str(
                get_value(record, "options")
                or get_value(record, "choices")
                or ""
            ).strip()

            parts = []
            if title:
                parts.append(title)
            parts.append(text)
            if options:
                parts.append(f"Options: {options}")

            return "\n\n".join(parts)

    messages = record.get("conversations") or record.get("messages") or []

    for message in messages:
        role = str(message.get("role") or message.get("from") or "").lower()
        text = str(message.get("content") or message.get("value") or "").strip()

        if role in ["user", "human"] and text:
            return text

    raise ValueError("Cannot find question.")


def extract_answer(record: Dict[str, Any]) -> str:
    for key in [
        "answer",
        "ground_truth_solution",
        "deepseek_solution",
        "solution",
        "output",
        "canonical_solution",
    ]:
        text = str(get_value(record, key) or "").strip()
        if text:
            return text

    messages = record.get("conversations") or record.get("messages") or []

    for message in messages:
        role = str(message.get("role") or message.get("from") or "").lower()
        text = str(message.get("content") or message.get("value") or "").strip()

        if role in ["assistant", "model", "gpt", "bot"] and text:
            return text

    return ""


def get_value(record: Dict[str, Any], key: str) -> Any:
    if key in record and record[key] not in [None, ""]:
        return record[key]

    metadata = record.get("metadata")
    if isinstance(metadata, dict) and key in metadata and metadata[key] not in [None, ""]:
        return metadata[key]

    return None


def load_json(path: Path) -> List[DatasetSample]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [DatasetSample.from_dict(item) for item in data]


def write_json(path: Path, samples: List[DatasetSample]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    data = [sample.to_dict() for sample in samples]

    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


# def main() -> None:
    
#     for spec in DATASETS:
#         dataset_name = spec["name"]
        
#         data_dir = Path(f"data/{dataset_name}")
#         data_dir.mkdir(parents=True, exist_ok=True)

#         open_thoughts_samples = 100
#         seed = 42

#         train_path = data_dir / "train.json"
#         test_path = data_dir / "test.json"

#         split_data = build_train_test_json(
#             open_thoughts_samples=open_thoughts_samples,
#             seed=seed,
#             spec=spec
#         )

#         write_json(train_path, split_data["train"])
#         write_json(test_path, split_data["test"])

#         json_files = sorted(data_dir.glob("*.json"))

#         print("\nAll json files:")
#         for path in json_files:
#             print(path)


# if __name__ == "__main__":
#     main()


class ReasoningDataset:
    def __init__(
        self,
        data_path: Optional[str] = None,
    ) -> None:
        self.data_path = Path(data_path) if data_path else None
        self.samples = self.load()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> DatasetSample:
        return self.samples[index]

    def __iter__(self) -> Iterator[DatasetSample]:
        return iter(self.samples)

    def load(self) -> List[DatasetSample]:
        if self.data_path is None:
            raise ValueError("data_path must be provided.")

        if not self.data_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {self.data_path}")

        if self.data_path.suffix != ".json":
            raise ValueError(f"Only .json dataset files are supported: {self.data_path}")

        return load_json(self.data_path)