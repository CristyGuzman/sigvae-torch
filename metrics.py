from sklearn.metrics import average_precision_score, roc_auc_score
import numpy as np
import torch
import copy

class Metrics(object):
    def __init__(self):
        self.n_samples = 0
        self.metrics_agg = {"roc": None, "ap": None}
        self._should_call_reset = False  # a guard to avoid stupid mistakes

    def reset(self):
        """
        Reset all metrics.
        """
        self.metrics_agg = {"roc": None, "ap": None}
        self.n_samples = 0
        self._should_call_reset = False  # now it's again safe to compute new values

    def compute(self, predictions, targets):
        roc = roc_auc_score(targets, predictions)
        ap = average_precision_score(targets, predictions)
        metrics = {"roc": roc, "ap": ap}
        return metrics

    def aggregate(self, new_metrics):
        assert isinstance(new_metrics, dict)
        assert list(new_metrics.keys()) == list(self.metrics_agg.keys())
        # sum over the batch dimension
        for m in new_metrics:
            if self.metrics_agg[m] is None:
                self.metrics_agg[m] = np.sum(new_metrics[m], axis=0)
            else:
                self.metrics_agg[m] += np.sum(new_metrics[m], axis=0)

        # keep track of the total number of samples processed
        print(f"New metrics are: {new_metrics}")
        #batch_size = new_metrics[list(new_metrics.keys())[0]].shape[0] # doesnt have shape bc it's a single number
        self.n_samples += 1 ## harcoded to 1, bc batch creates a single graph

    def compute_and_aggregate(self, predictions,targets):
        if isinstance(predictions, torch.Tensor):
            ps = predictions.detach().cpu().numpy()
            ts = targets.detach().cpu().numpy()
        else:
            ps = predictions
            ts = targets
        print(f"predictions and targets are: {(ps, ts)}")
        new_metrics = self.compute(ps, ts)
        self.aggregate(new_metrics)

    def get_final_metrics(self):
        """
        Finalize and return the metrics - this should only be called once all the data has been processed.
        :return: A dictionary of the final aggregated metrics per time step.
        """
        self._should_call_reset = True  # make sure to call `reset` before new values are computed
        assert self.n_samples > 0

        for m in self.metrics_agg:
            self.metrics_agg[m] = self.metrics_agg[m] / self.n_samples

        # return a copy of the metrics so that the class can be re-used again immediately
        return copy.deepcopy(self.metrics_agg)

