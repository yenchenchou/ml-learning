from src import (
    DataCleaner,
    DataFetcher,
    DataSpliter,
    EnvInit,
    FeatEngineer,
    MovieLensDataset,
)

env_init = EnvInit()
data_fetch = DataFetcher()
data_clean = DataCleaner()
data_split = DataSpliter()
data_feat = FeatEngineer()

if __name__ == "__main__":
    device = env_init.available_device()
    seed = env_init.fix_seed(12345)
    df = data_fetch()
    df = data_clean(df)
    df_train, df_eval, df_test = data_split(df, method="leave-k-out", k=1)
    del df
    data_feat.get_components(df_train)
    neg_sampling_ratio = 4
    train_users, train_items, train_labels = data_feat.neg_sampling(
        df_train, ratio=neg_sampling_ratio
    )
    eval_users, eval_items, eval_labels = data_feat.neg_sampling(
        df_eval, ratio=neg_sampling_ratio
    )
    train_dataset = MovieLensDataset(
        users=train_users, items=train_items, labels=train_labels
    )
    eval_dataset = MovieLensDataset(
        users=eval_users, items=eval_items, labels=eval_labels
    )
    print("Done")
