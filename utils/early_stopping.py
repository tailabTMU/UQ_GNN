import torch
class EarlyStopping:
    def __init__(self, best_model_path, patience=10, min_delta=0):
        """
        Args:
            best_model_path (str): Path to saved best model.
            patience (int): How many epochs to wait after the last significant increase.
            min_delta (float): Minimum change in the monitored metric to qualify as a significant increase.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False
        self.best_model_path = best_model_path

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss:  # Improvement is always OK
            self.best_loss = val_loss
            self.counter = 0
            torch.save(model.state_dict(), self.best_model_path)
        elif val_loss > self.best_loss + self.min_delta:  # Significant increase
            self.counter += 1
        else:
            self.counter = 0

        if self.counter >= self.patience:
            self.early_stop = True
