"""
Training logger.

Logs metrics in table-like format.
"""

class TrainingLogger:

    def __init__(self):
        self.history = []

    def log(self, epoch, train_loss, val_loss):
        self.history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss
        })

    def summary(self):
        for row in self.history:
            print(row)