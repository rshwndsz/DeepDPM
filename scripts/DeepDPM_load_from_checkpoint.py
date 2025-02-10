from argparse import ArgumentParser, Namespace

import torch
from src.clustering_models.clusternet_modules.clusternetasmodel import ClusterNetModel
from src.datasets import CustomDataset


def main(args):
    # Load model from checkpoint
    cp_path = args.checkpoint_path
    cp_state = torch.load(cp_path)
    hparams = Namespace()
    for key, value in cp_state["hyper_parameters"].items():
        setattr(hparams, key, value)
    model = ClusterNetModel.load_from_checkpoint(
        checkpoint_path=args.checkpoint_path,
        input_dim=args.data_dim,
        init_k=cp_state["state_dict"]["cluster_net.class_fc2.weight"].shape[0],
        hparams=hparams,
    )
    model.eval()

    # Inference
    dataset_obj = CustomDataset(args)
    _, val_loader = dataset_obj.get_loaders()
    cluster_assignments = []
    assert val_loader is not None
    for data, label in val_loader:
        soft_assign = model(data)
        hard_assign = soft_assign.argmax(-1)
        cluster_assignments.append(hard_assign)
    print(cluster_assignments)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--data-dim", type=int, default=27)
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--transform-input-data", type=str, default="as_is")
    parser.add_argument("--use-labels-for-eval", action="store_true")

    args = parser.parse_args()
    main(args)
