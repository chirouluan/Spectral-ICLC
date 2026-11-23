"""LUMIR classification adapters, independent of the retired reproduction scripts."""
import numpy as np
from .lumir_reference import SinglePrompt, AdaptivePrompt, SOURCE_SHA256, format_examples


def lumir_messages(x, labels, query, candidates, adaptive=False):
    filename = "Mul_Generate.py" if adaptive else "Generate_single.py"
    obj = AdaptivePrompt() if adaptive else SinglePrompt()
    train = [{"x": row.round(6).tolist(), "label": str(float(label))} for row, label in zip(x, labels)]
    test = [{"x": query.round(6).tolist()}]
    messages = [{"role": "system", "content": obj.SYSTEM_PROMPT},
                {"role": "user", "content": f"Allowed labels: {[str(float(v)) for v in candidates]}\nFew-shot examples:\n"
                 f"{format_examples(train)}\n\n{obj.format_eval_prompt(test)}"}]
    return messages, SOURCE_SHA256[filename]


def parse_lumir(text, candidates):
    import json
    values = json.loads(text)
    if not isinstance(values, list) or len(values) != 1:
        raise ValueError("LUMIR requires exactly one predicted label for one query")
    from .llm import parse_label
    return parse_label(json.dumps({"label": values[0]}), candidates)


class NeuralClassifier:
    """Original LUMIR architectures; explicit validation-only early stopping."""
    def __init__(self, name, seed, config, checkpoint):
        self.name, self.seed, self.config, self.checkpoint = name, seed, config, checkpoint

    def fit(self, x, y, validation):
        import torch
        import torch.nn as nn
        from sklearn.preprocessing import StandardScaler
        from torch.utils.data import DataLoader, TensorDataset
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        torch.set_num_threads(2)
        torch.use_deterministic_algorithms(True)
        self.device = torch.device("cpu")  # conservative deterministic default
        self.classes_ = np.unique(y)
        self.scaler = StandardScaler().fit(x)
        vx, vy = validation
        if not len(vx):
            raise ValueError("Neural models require seen-class validation for early stopping")
        lookup = {label: i for i, label in enumerate(self.classes_)}
        # In the unseen-concentration task, validation also contains held-out
        # labels. They are excluded from checkpoint selection, as a closed-set
        # classifier cannot learn or select against labels absent from Train.
        keep = np.asarray([label in lookup for label in vy], dtype=bool)
        vx, vy = np.asarray(vx)[keep], np.asarray(vy)[keep]
        if not len(vx):
            raise ValueError("Validation contains no classes represented in Train")
        from .architectures import CNN1DClassifier, TransformerClassifier
        self.source_hash = SOURCE_SHA256["other_models.py"]
        cls = CNN1DClassifier if self.name == "cnn1d" else TransformerClassifier
        self.model = cls(x.shape[1], len(self.classes_)).to(self.device)
        tx = torch.tensor(self.scaler.transform(x), dtype=torch.float32)
        ty = torch.tensor([lookup[v] for v in y], dtype=torch.long)
        valx = torch.tensor(self.scaler.transform(vx), dtype=torch.float32)
        valy = torch.tensor([lookup[v] for v in vy], dtype=torch.long)
        batch = min(self.config.get("batch_size", 8), len(tx))
        if batch < 2:
            raise ValueError("At least two reference samples are required")
        generator = torch.Generator().manual_seed(self.seed)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.config.get("learning_rate", .001),
                               weight_decay=self.config.get("weight_decay", 0.0))
        use_loss = self.config.get("selection_metric", "accuracy") == "loss"
        best, stale, history, best_epoch = (float("inf") if use_loss else -1.), 0, [], None
        steps = 0
        for epoch in range(self.config.get("epochs", 50)):
            self.model.train()
            # Merge a singleton remainder with the preceding batch. BatchNorm in
            # CNN1D can otherwise fail for short feature sequences.
            order = torch.randperm(len(tx), generator=generator).tolist()
            batches = [order[i:i+batch] for i in range(0, len(order), batch)]
            if len(batches) > 1 and len(batches[-1]) == 1:
                batches[-2].extend(batches.pop())
            total_loss = 0.0
            for indices in batches:
                bx, by = tx[indices], ty[indices]
                opt.zero_grad()
                loss = nn.functional.cross_entropy(self.model(bx), by)
                loss.backward()
                opt.step()
                steps += 1
                total_loss += float(loss.detach()) * len(indices)
            self.model.eval()
            with torch.no_grad():
                outputs = torch.cat([self.model(bx) for bx in valx.split(batch)])
                acc = float((outputs.argmax(1) == valy).float().mean())
                val_loss = float(nn.functional.cross_entropy(outputs, valy))
            history.append({"epoch": epoch + 1, "train_loss": total_loss / len(tx),
                            "validation_loss": val_loss, "validation_accuracy": acc,
                            "optimizer_steps": steps})
            score = val_loss if use_loss else acc
            delta = self.config.get("min_delta", 0.0)
            improved = score < best - delta if use_loss else score > best + delta
            if improved:
                best, stale = score, 0
                best_epoch = epoch
                best_accuracy = acc
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), self.checkpoint)
            else:
                stale += 1
            if epoch + 1 >= self.config.get("min_epochs", 0) and stale >= self.config.get("patience", 10):
                break
        self.model.load_state_dict(torch.load(self.checkpoint, map_location="cpu", weights_only=True))
        self.model.eval()
        self.history = history
        self.best_epoch = int(best_epoch)
        self.best_validation_accuracy = float(best_accuracy)
        self.best_validation_loss = float(best_val_loss)
        self.optimizer_steps = steps
        self.actual_batch_size = batch
        self.epochs_ran = len(history)
        return self

    def predict(self, x):
        import torch
        with torch.no_grad():
            pred = self.model(torch.tensor(self.scaler.transform(x), dtype=torch.float32)).argmax(1).numpy()
        return self.classes_[pred]
