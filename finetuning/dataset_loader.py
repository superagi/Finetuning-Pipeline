from datasets import load_dataset


def format_example(example):
    return {"text": f"[INST] {example['instruction']} [/INST] {example['answer']}"}


class DatasetLoader:
    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name
        self.dataset_loaded = False
        self._load_dataset()

    def _load_dataset(self):
        # Load training split (you can process it here)
        self.train_dataset = load_dataset(
            self.dataset_name, split="train[:20%]", use_auth_token="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")
        self.train_dataset = self.train_dataset.map(format_example)
        self.train_dataset = self.train_dataset.remove_columns(
            ['instruction', 'answer'])
        self.data = self.train_dataset.train_test_split(test_size=0.01)
        # Set the dataset flag
        self.dataset_loaded = True

    def get_dataset(self):
        assert self.dataset_loaded, \
            "Dataset not loaded. Please run load_dataset() first."
        return self.data['train'], self.data['test']
