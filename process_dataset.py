from datasets import Dataset, load_dataset
from argparse import ArgumentParser

MAX_LENGTH = 256
HF_DS = 'allenai/social_bias_frames'

def sanity_check(df, ds, feature):
    for i in range(len(df)):
        assert df.at[i, feature] == ds[i][feature]

def get_dataset(split: str, feature: str):
    ds = load_dataset(HF_DS, split=split, trust_remote_code=True)
    ds = ds.select_columns(['post', feature]).map(lambda x: {'post': x['post'], feature: float(x[feature]) if len(x[feature]) > 0 else 0.0})
    df = ds.to_pandas()
    df = df.groupby(['post']).mean().reset_index()
    ds = Dataset.from_pandas(df)
    sanity_check(df, ds, feature)
    return ds

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-f', '--feature', required=True, choices=['offensiveYN', 'sexYN', 'intentYN', 'speakerMinorityYN'], type=str)
    parser.add_argument('-s', '--split', required=True, choices=['train', 'validation', 'test'], type=str)
    args = parser.parse_args()
    new_ds = get_dataset(args.split, args.feature)
    new_ds.save_to_disk(f'data/{args.feature}/{args.split}')
    new_ds = Dataset.load_from_disk(f'data/{args.feature}/{args.split}')
    new_ds.push_to_hub('anoaky/sbf-collated',
                          config_name=args.feature,
                          split=args.split)