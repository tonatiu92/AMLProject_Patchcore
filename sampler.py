import torch
import numpy as np



class Sampler():
    STARTING_POINT = 10
    def __init__(self,percentage: float,device: torch.device,):
        self.percentage = percentage
        self.device = device
    
    def _compute_batchwise_differences(self,A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """Computes batchwise Euclidean distances using PyTorch."""
        a_times_a = A.unsqueeze(1).bmm(A.unsqueeze(2)).reshape(-1, 1)
        b_times_b = B.unsqueeze(1).bmm(B.unsqueeze(2)).reshape(1, -1)
        a_times_b = A.mm(B.T)

        return (-2 * a_times_b + a_times_a + b_times_b).clamp(0, None).sqrt()

    def run(self, features: torch.Tensor, projection:int = 1024 ):
        """Subsamples features using Greedy Coreset.

        Args:
            features: [N x D]
        """
        if self.percentage == 1:
            #if percentage is 1 we want the entire memory bank
            return features
        
        #Applying Johnson-Lindenstrauss theorem to reduce dimensionalities through random linear projections  
        if features.shape[1] != projection:
            mapper = torch.nn.Linear(features.shape[1], projection, bias=False)
            _ = mapper.to(self.device)
            features = features.to(self.device)
        sample_indices = self._compute_greedy_coreset_indices(features)
        features = features[sample_indices]
        return features

    def _compute_greedy_coreset_indices(self, features: torch.Tensor) -> np.ndarray:
        number_of_starting_points = np.clip(
            self.STARTING_POINT, None, len(features)
        ) #define the number of starting points
        
        #among this point choose a random starting point
        start_points = np.random.choice(
            len(features), number_of_starting_points, replace=False
        ).tolist()

        #compute the matrix of distances
        approximate_distance_matrix = self._compute_batchwise_differences(
            features, features[start_points]
        )
        approximate_coreset_anchor_distances = torch.mean(
            approximate_distance_matrix, axis=-1
        ).reshape(-1, 1)
        coreset_indices = []
        num_coreset_samples = int(len(features) * self.percentage)

        with torch.no_grad():
            for _ in range(num_coreset_samples):
                select_idx = torch.argmax(approximate_coreset_anchor_distances).item()
                coreset_indices.append(select_idx)
                coreset_select_distance = self._compute_batchwise_differences(
                    features, features[select_idx : select_idx + 1]  # noqa: E203
                )
                approximate_coreset_anchor_distances = torch.cat(
                    [approximate_coreset_anchor_distances, coreset_select_distance],
                    dim=-1,
                )
                approximate_coreset_anchor_distances = torch.min(
                    approximate_coreset_anchor_distances, dim=1
                ).values.reshape(-1, 1)

        return np.array(coreset_indices)