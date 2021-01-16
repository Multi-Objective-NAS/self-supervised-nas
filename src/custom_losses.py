from pytorch_metric_learning import losses


class PenalizedMarginLoss(losses.MarginLoss):
    def __init__(self, penalty, **kwargs):
        super().__init__(**kwargs)
        self.penalty = penalty


    def compute_loss(self, embeddings, labels, indices_tuple):
        indices_tuple = lmu.convert_to_triplets(indices_tuple, labels, self.triplets_per_anchor)
        anchor_idx, positive_idx, negative_idx = indices_tuple
        if len(anchor_idx) == 0:
            return self.zero_losses()
        anchors, positives, negatives = embeddings[anchor_idx], embeddings[positive_idx], embeddings[negative_idx]

        beta = self.beta if len(self.beta) == 1 else self.beta[labels[anchor_idx]]
        beta = beta.to(embeddings.device)
        
        d_ap = torch.nn.functional.pairwise_distance(positives, anchors, p=2)
        d_an = torch.nn.functional.pairwise_distance(negatives, anchors, p=2)

        pos_loss = torch.nn.functional.relu(d_ap - beta + self.margin)
        neg_loss = torch.nn.functional.relu(beta - d_an + self.margin)

        penalize_pos_loss = torch.nn.functional.relu(-(d_ap - beta + self.margin)) * self.penalty
        penalize_neg_loss = torch.nn.functional.relu(-(beta - d_an + self.margin)) * self.penalty

        num_pos_pairs = (pos_loss > 0.0).nonzero().size(0)
        num_neg_pairs = (neg_loss > 0.0).nonzero().size(0)

        divisor_summands = {"num_pos_pairs": num_pos_pairs, "num_neg_pairs": num_neg_pairs}

        margin_loss = pos_loss + neg_loss + penalize_pos_loss + penalize_neg_loss

        loss_dict = {"margin_loss": {"losses": margin_loss, "indices": indices_tuple, "reduction_type": "triplet", "divisor_summands": divisor_summands}, 
                    "beta_reg_loss": self.compute_reg_loss(beta, anchor_idx, divisor_summands)}

        return loss_dict
