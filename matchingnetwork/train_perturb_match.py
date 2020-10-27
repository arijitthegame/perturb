import os
import torch
from tqdm import tqdm
import logging

from torchmeta.datasets.helpers import omniglot
from torchmeta.utils.data import BatchMetaDataLoader
from torchmeta.utils.matching import matching_log_probas, matching_loss

from model_match import MLP
import os
import torch
from tqdm import tqdm
import logging
import makedata
import preprocessing
import scanpy as sc
from dataloader import perturbdataloader
from torch.utils.data import DataLoader

from torchmeta.utils.gradient_based import gradient_update_parameters

from utils import get_accuracy
torch.manual_seed(0)
logger = logging.getLogger(__name__)


def train(args):
    logger.warning('This script is an example to showcase the extensions and '
                   'data-loading features of Torchmeta, and as such has been '
                   'very lightly tested.')

    perturb_mock, sgRNA_list_mock = makedata.json_to_perturb_data(
        path="/home/member/xywang/WORKSPACE/MaryGUO/one-shot/MOCK_MON_crispr_combine/crispr_analysis")

    total = sc.read_h5ad("/home/member/xywang/WORKSPACE/MaryGUO/one-shot/mock_one_perturbed.h5ad")
    trainset, testset = preprocessing.make_total_data(total, sgRNA_list_mock)

    TrainSet = perturbdataloader(trainset, ways=args.num_ways, support_shots=args.num_shots, query_shots=15)
    TrainLoader = DataLoader(TrainSet, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)


    model = MLP(args.embedding_size)
    model.to(device=args.device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Training loop

    with tqdm(TrainLoader, total=args.num_batches) as pbar:
        for batch_idx, (inputs_support, inputs_query, target_support, target_query) in enumerate(pbar):
            model.zero_grad()

            inputs_support = inputs_support.to(device=args.device)
            target_support = target_support.to(device=args.device)
            train_embeddings = model(inputs_support)

            inputs_query = inputs_query.to(device=args.device)
            target_query = target_query.to(device=args.device)
            test_embeddings = model(inputs_query)

            loss = matching_loss(train_embeddings,
                                 target_support,
                                 test_embeddings,
                                 target_query,
                                 args.num_ways)

            loss.backward()
            optimizer.step()

            with torch.no_grad():
                # calculate the accuracy
                log_probas = matching_log_probas(train_embeddings,
                                                 target_support,
                                                 test_embeddings,
                                                 args.num_ways)
                test_predictions = torch.argmax(log_probas, dim=1)
                accuracy = torch.mean((test_predictions == target_query).float())
                pbar.set_postfix(accuracy='{0:.4f}'.format(accuracy.item()))

            if batch_idx >= args.num_batches:
                break

    # Save model
    if args.output_folder is not None:
        filename = os.path.join(args.output_folder, 'matching_network_omniglot_'
            '{0}shot_{1}way.pt'.format(args.num_shots, args.num_ways))
        with open(filename, 'wb') as f:
            state_dict = model.state_dict()
            torch.save(state_dict, f)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser('Matching Networks')

    parser.add_argument('--folder', type=str,
                        default="/home/member/xywang/WORKSPACE/MaryGUO/pytorch-meta",
                        help='Path to the folder the data is downloaded to.')
    parser.add_argument('--num-shots', type=int, default=5,
        help='Number of examples per class (k in "k-shot", default: 5).')
    parser.add_argument('--num-ways', type=int, default=5,
        help='Number of classes per task (N in "N-way", default: 5).')

    parser.add_argument('--embedding-size', type=int, default=64,
        help='Dimension of the embedding/latent space (default: 64).')
    parser.add_argument('--hidden-size', type=int, default=64,
        help='Number of channels for each convolutional layer (default: 64).')

    parser.add_argument('--output-folder', type=str, default=None,
        help='Path to the output folder for saving the model (optional).')
    parser.add_argument('--batch-size', type=int, default=16,
        help='Number of tasks in a mini-batch of tasks (default: 16).')
    parser.add_argument('--num-batches', type=int, default=100000,
        help='Number of batches the prototypical network is trained over (default: 100).')
    parser.add_argument('--num-workers', type=int, default=1,
        help='Number of workers for data loading (default: 1).')
    parser.add_argument('--download', action='store_true',
        help='Download the Omniglot dataset in the data folder.')
    parser.add_argument('--use-cuda', default= True,
                        help='Use CUDA if available.')
    parser.add_argument('--gpu', type=int, default= 9)

    args = parser.parse_args()
    args.device = torch.device('cuda' if args.use_cuda
                                         and torch.cuda.is_available() else 'cpu')

    is_cuda = torch.cuda.is_available()
    if is_cuda:
        torch.cuda.set_device(args.gpu)

    train(args)
