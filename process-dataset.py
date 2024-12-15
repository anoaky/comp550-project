import torch
from transformers import PreTrainedTokenizer, BartTokenizer
from datasets import Dataset, load_dataset
from argparse import ArgumentParser

MAX_LENGTH = 256
HF_DS = 'allenai/social_bias_frames'

tok_kwargs = {
    'padding': 'max_length',
    'max_length': MAX_LENGTH,
    'truncation': True,
    'return_tensors': 'pt',
    'return_attention_mask': True,
    'add_special_tokens': True,
}

def get_dataset(split: str, feature: str, tokenizer: PreTrainedTokenizer):
    def remove_blanks(row):
        if len(row[feature]) == 0:
            row[feature] = '0.0'
            # for *some* reason, there are blank entries, even though
            # that should correspond to '0.0'
        return row
    def binarize_feature(ds: Dataset):
        posts = ds.unique('post')
        responses = dict.fromkeys(posts, [])
        def gather(x):
            responses[x['post']].append(x[feature])
        def binarize(x):
            post = tokenizer(x['post'], **tok_kwargs)
            mean_response = torch.tensor(responses[x['post']], dtype=torch.float32).mean().softmax(0).round().item()
            return {
                'input_ids': post.input_ids.view(-1),
                'attention_mask': post.attention_mask.view(-1),
                'labels': mean_response,
            }
        ds.map(gather, desc='Gathering annotator responses...')
        return ds.map(binarize, remove_columns=['post', feature], desc='Binarizing selected feature...')
    ds = load_dataset(HF_DS, split=split, trust_remote_code=True)
    ds = ds.select_columns(['post', feature]).map(remove_blanks)
    ds = ds.class_encode_column(feature)
    ds = binarize_feature(ds)
    return ds

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-f', '--feature', required=True, choices=['offensiveYN'], type=str)
    args = parser.parse_args()
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
    train_set = get_dataset('train', args.feature, tokenizer)
    train_set.push_to_hub('anoaky/sbf-bart-tokenized',
                          config_name=args.feature,
                          split='train')