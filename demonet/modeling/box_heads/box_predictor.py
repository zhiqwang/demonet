import torch
import torchvision.ops

from ..priors_generator.anchor_utils import decode


class BoxPredictor(object):
    """At test time, Detect is the final layer of SSD. Decode location preds,
    apply non-maximum suppression to location predictions based on conf
    scores and threshold to a top_k number of output predictions for both
    confidence score and locations.
    """
    def __init__(
        self, num_classes, bkg_label, top_k,
        conf_thresh, nms_thresh,
    ):
        self.num_classes = num_classes
        self.background_label = bkg_label
        self.top_k = top_k
        # Parameters used in nms.
        self.nms_thresh = nms_thresh
        if nms_thresh <= 0:
            raise ValueError('nms_threshold must be non negative.')
        self.conf_thresh = conf_thresh
        self.variance = [0.1, 0.2]
        self.width = 1
        self.height = 1

    def __call__(self, loc_data, conf_data, priors):
        """
        Args:
            loc_data (tensor): [batch_size, num_priors x 4] predicted locations.
            conf_data (tensor): [batch_size x num_priors, num_classes] class predictions.
            priors (tensor): [num_priors, 4] real boxes corresponding all the priors.
        """
        batch_size = loc_data.shape[0]
        num_priors = priors.shape[0]
        # conf_preds: batch_size x num_priors x num_classes
        conf_preds = conf_data.view(batch_size, num_priors, self.num_classes)
        device = conf_data.device
        priors = priors.to(device)
        # output: label, score, xmin, ymin, xmax, ymax
        output = torch.zeros(batch_size, self.top_k, 6)

        # Decode predictions into bboxes.
        for i in range(batch_size):
            decoded_boxes = decode(loc_data[i], priors, self.variance)  # num_priors x 4
            # For each class, perform nms
            conf_scores = conf_preds[i].clone()  # num_priors x num_classes

            num_classes = conf_scores.shape[1]

            decoded_boxes = decoded_boxes.view(num_priors, 1, 4)
            decoded_boxes = decoded_boxes.expand(num_priors, num_classes, 4)
            labels = torch.arange(num_classes, device=device)
            labels = labels.view(1, num_classes).expand_as(conf_scores)

            # remove predictions with the background label
            decoded_boxes = decoded_boxes[:, 1:]
            conf_scores = conf_scores[:, 1:]
            labels = labels[:, 1:]

            # batch everything, by making every class prediction be a separate instance
            decoded_boxes = decoded_boxes.reshape(-1, 4)
            conf_scores = conf_scores.reshape(-1)
            labels = labels.reshape(-1)

            # remove low scoring decoded_boxes
            indices = torch.nonzero(conf_scores > self.conf_thresh).squeeze(1)
            decoded_boxes = decoded_boxes[indices]
            conf_scores = conf_scores[indices]
            labels = labels[indices]

            decoded_boxes[:, 0::2] *= self.width
            decoded_boxes[:, 1::2] *= self.height

            keep = torchvision.ops.boxes.batched_nms(
                decoded_boxes, conf_scores, labels, self.nms_thresh,
            )

            # keep only topk scoring predictions
            keep = keep[:self.top_k]

            decoded_boxes = decoded_boxes[keep]
            conf_scores = conf_scores[keep][:, None]
            labels = labels[keep][:, None].float()

            output[i, :len(keep)] = torch.cat((labels, conf_scores, decoded_boxes), 1)
        return output
