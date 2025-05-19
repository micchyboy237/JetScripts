import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from model import load_translation_model
import logging
from stratified_sampler import StratifiedSampler, StratifiedData
from dataset import ProcessedData, load_translation_data
import mplcursors
import numpy as np
import json
import os
from pathlib import Path
import time


class DatasetEvaluator:
    def __init__(
        self,
        model,
        tokenizer,
        threshold=0.1,
        max_data_points=float('inf'),
        batch_size=32,
        sample_size=100,
        verbose=False,
        reports_dir=None,
        checkpoints_dir=None,
        eval_interval=25
    ):

        self.model = model
        self.tokenizer = tokenizer
        self.threshold = threshold
        self.max_data_points = max_data_points
        self.batch_size = batch_size
        self.sample_size = sample_size
        self.verbose = verbose
        self.reports_dir = reports_dir
        self.checkpoints_dir = checkpoints_dir
        self.eval_interval = eval_interval

        self.best_loss = float('inf')
        self.best_checkpoint_path = None
        self.last_checkpoint_path = None

        logging.basicConfig(level=logging.INFO if verbose else logging.WARNING)

    def evaluate_and_filter_weak_data(self, dataset, custom_loss_function=None):
        logging.info(
            "Starting dataset evaluation and filtering weak performing data...")
        filtered_data, detailed_report = [], []
        total_data_points = min(self.max_data_points, len(dataset))

        # Customize the progress bar format
        pbar_format = "{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
        pbar = tqdm(total=total_data_points, desc="Evaluating",
                    leave=False, bar_format=pbar_format)

        total_loss = 0.0
        processed_data_points = 0

        try:
            for step in range(0, total_data_points, self.batch_size):
                batch_data = dataset[step:step + self.batch_size]
                # Adjust batch_data to fit model input
                adapted_batch_data = [(item.source, item.target)
                                      for item in batch_data]
                losses = self._process_batch(
                    adapted_batch_data, custom_loss_function)
                self._update_reports(batch_data, losses,
                                     filtered_data, detailed_report, pbar, step)

                batch_loss_sum = losses.sum().item()
                total_loss += batch_loss_sum
                processed_data_points += len(batch_data)
                average_loss = total_loss / processed_data_points
                last_loss = losses[-1].item()
                batch_num = (step // self.batch_size) + 1

                pbar.set_description(
                    f"Batch {batch_num}; Step {step}; (Last Loss: {last_loss:.4f}, Avg Loss: {average_loss:.4f})")

                if batch_num % self.eval_interval == 0:
                    # Save the model checkpoint
                    self._save_model_checkpoint(batch_num, average_loss)

        except Exception as e:
            logging.error(f"Error during evaluation: {e}")
        finally:
            pbar.close()

        logging.info("Evaluation completed...")
        logging.info(f"Weak Data to Train: {len(filtered_data)}")

        if self.reports_dir:
            self._save_detailed_report(detailed_report)

        return detailed_report

    def _save_model_checkpoint(self, batch_num, loss):
        logging.info("Saving model checkpoint...")

        save_checkpoints_dir_path = Path(self.checkpoints_dir)
        save_checkpoints_dir_path.mkdir(exist_ok=True, parents=True)

        # Remove the old last checkpoint if it exists
        if self.last_checkpoint_path and os.path.exists(self.last_checkpoint_path):
            os.remove(self.last_checkpoint_path)
            logging.info(
                f"Removed old last checkpoint: {self.last_checkpoint_path}")

        # Save the last checkpoint
        last_checkpoint_name = f"eval_weak_last_batch={batch_num}_loss={loss:.4f}.ckpt"
        self.last_checkpoint_path = os.path.join(
            self.checkpoints_dir, last_checkpoint_name)
        torch.save(self.model.state_dict(), self.last_checkpoint_path)
        logging.info(
            f"Last model checkpoint saved: {self.last_checkpoint_path}")

        # Save the best checkpoint if current loss is lower than the best loss
        if loss < self.best_loss:
            # Remove the old best checkpoint if it exists
            if self.best_checkpoint_path and os.path.exists(self.best_checkpoint_path):
                os.remove(self.best_checkpoint_path)
                logging.info(
                    f"Removed old best checkpoint: {self.best_checkpoint_path}")

            self.best_loss = loss
            best_checkpoint_name = f"eval_weak_best_batch={batch_num}_loss={loss:.4f}.ckpt"
            self.best_checkpoint_path = os.path.join(
                self.checkpoints_dir, best_checkpoint_name)
            torch.save(self.model.state_dict(), self.best_checkpoint_path)
            logging.info(
                f"Best model checkpoint saved: {self.best_checkpoint_path}")

    def _save_detailed_report(self, detailed_report):
        logging.info("Saving evaluated filtered data report...")

        save_reports_dir_path = Path(self.reports_dir)
        save_reports_dir_path.mkdir(exist_ok=True, parents=True)

        report_name = f"eval_weak_{time.strftime('%Y%m%d-%H%M%S')}.json"

        with open(os.path.join(save_reports_dir_path, report_name), 'w', encoding='utf-8') as f:
            json.dump(detailed_report, f, indent=4)

    def _process_batch(self, batch_data, custom_loss_function):
        inputs = self.tokenizer([src for src, _ in batch_data],
                                return_tensors='pt', padding=True, truncation=True, max_length=512)
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer([tgt for _, tgt in batch_data],
                                    return_tensors='pt', padding=True, truncation=True, max_length=512)
        decoder_input_ids = self._shift_tokens_right(labels['input_ids'])

        with torch.no_grad():
            outputs = self.model(
                input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'], decoder_input_ids=decoder_input_ids)
            losses = custom_loss_function(outputs, labels['input_ids'], self.tokenizer) if custom_loss_function else self._calculate_loss(
                outputs, labels['input_ids'])

        # Ensure losses are iterable
        if losses.dim() == 0:  # Checking if losses is a 0-d tensor (scalar)
            # Convert scalar to a 1D tensor with one element
            losses = losses.view(1)

        return losses

    def _update_reports(self, batch_data, losses, filtered_data, detailed_report, pbar, batch_start_index):
        for idx, (data_item, loss) in enumerate(zip(batch_data, losses)):
            global_index = batch_start_index + idx
            loss_value = loss.item()
            is_filtered = loss_value > self.threshold

            if is_filtered:
                filtered_data.append(data_item)
                detailed_report.append({
                    'index': global_index,
                    'source': data_item.source,
                    'target': data_item.target,
                    'loss': loss_value,
                    'filtered': is_filtered,
                    'score': data_item.score  # Include the score from StratifiedData
                })

            pbar.update(1)

    def _calculate_loss(self, outputs, labels):
        criterion = torch.nn.CrossEntropyLoss()
        return criterion(outputs.logits.view(-1, self.model.config.vocab_size), labels.view(-1))

    def _shift_tokens_right(self, input_ids):
        shifted_input_ids = input_ids.clone()
        shifted_input_ids[:, :-1] = input_ids[:, 1:]
        shifted_input_ids[:, -1] = self.tokenizer.pad_token_id
        return shifted_input_ids


def generate_report_scores(detailed_report):
    # Function to process and print out the detailed report with scores
    # This can be formatted as per your requirement
    results_with_scores = []
    for item in detailed_report:
        # Use the score directly from the StratifiedData instance
        score = item['score'] if 'score' in item else None
        results_with_scores.append({**item, 'score': score})

        # Display with losses and scores
        print(
            f"Index: {int(item['index'])}, Source: {item['source']}, Target: {item['target']}, Loss: {item['loss']}, Filtered: {item['filtered']}, Score: {score}")

    return results_with_scores


def plot_loss_values(detailed_report_with_scores, threshold):
    # Extract loss values, scores, and whether they were filtered
    losses = [item['loss'] for item in detailed_report_with_scores]
    filtered = [item['filtered'] for item in detailed_report_with_scores]
    scores = [item['score'] for item in detailed_report_with_scores]

    # Create a scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(range(len(losses)), losses, c=[
                'red' if f else 'green' for f in filtered], label='Loss')
    plt.scatter(range(len(scores)), scores, c='blue', label='Score',
                alpha=0.5)  # Adding scores as blue dots
    plt.axhline(y=threshold, color='blue', linestyle='--',
                label=f'Threshold: {threshold}')

    # Labeling
    plt.title("Loss Values and Scores of Translations")
    plt.xlabel("Data Point Index")
    plt.ylabel("Loss / Score")
    plt.legend()
    plt.grid(True)

    # Display the plot en-tl sentences on hover
    # Include the loss and score
    cursor = mplcursors.cursor(hover=True)
    cursor.connect("add", lambda sel: sel.annotation.set_text(
        f"Loss: {losses[int(sel.index)]}\nScore: {scores[int(sel.index)]}\nSource: {detailed_report_with_scores[int(sel.index)]['source']}\nTarget: {detailed_report_with_scores[int(sel.index)]['target']}"))

    # Add text about average loss and average difference between scores and losses
    avg_loss = np.mean(losses)
    avg_score = np.mean(scores)
    avg_diff = np.mean(np.abs(np.array(scores) - np.array(losses)))
    plt.text(0.5, 0.95, f"Average Loss: {avg_loss:.4f}\nAverage Score: {avg_score:.4f}\nAverage Difference: {avg_diff:.4f}",
             horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)

    # Display the plot
    plt.show()


def main():
    save_reports_dir = "instruction_generator/translation/reports"

    directory = "server/static/datasets/translations"
    includes = ["diwa*.json", "jmbt*.json", "alpaca*.json"]
    excludes = ["Ramos*.json", "filwords_wiki*.json"]
    model_name = "Helsinki-NLP/opus-mt-en-tl"
    # checkpoint_path = None
    checkpoint_path = "server/static/models/translation/best.translation.ckpt"

    batch_size = 32
    threshold = 0.3

    # Load the dataset
    dataset = load_translation_data(directory, includes, excludes)

    # Initialize StratifiedSampler and DatasetEvaluator
    sampler = StratifiedSampler(dataset)
    stratified_samples = sampler.get_samples()

    # Initialize the Marian model and tokenizer for English to Tagalog translation
    model, tokenizer = load_translation_model(model_name, checkpoint_path)

    evaluator = DatasetEvaluator(
        model,
        tokenizer,
        threshold=threshold,
        batch_size=batch_size,
        reports_dir=save_reports_dir)

    # Evaluate and filter the stratified sample
    detailed_report = evaluator.evaluate_and_filter_weak_data(
        stratified_samples, custom_loss_function=None)

    detailed_report_with_scores = generate_report_scores(detailed_report)
    plot_loss_values(detailed_report_with_scores, evaluator.threshold)


if __name__ == "__main__":
    main()
