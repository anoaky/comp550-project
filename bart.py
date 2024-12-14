import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import BartForSequenceClassification
from sbf_common import SBFModel, SBFPreprocessed

class SBFBART(SBFModel):
    def __init__(self, 
                 /,
                 **kwargs):
        model = BartForSequenceClassification.from_pretrained('facebook/bart-large-mnli')
        super().__init__(model, **kwargs)