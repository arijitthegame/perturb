import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
import logging
import makedata
import preprocessing
import scanpy as sc
from dataloader import perturbdataloader,perturbdataloader_test
from torch.utils.data import DataLoader

from torchmeta.utils.gradient_based import gradient_update_parameters
from model import MLP
from utils import get_accuracy
import helpfuntions
import numpy as np
torch.manual_seed(0)
np.random.seed(0)
logger = logging.getLogger(__name__)


def train(args):

    perturb_mock, sgRNA_list_mock = makedata.json_to_perturb_data(path = "/home/member/xywang/WORKSPACE/MaryGUO/one-shot/MOCK_MON_crispr_combine/crispr_analysis")

    total = sc.read_h5ad("/home/member/xywang/WORKSPACE/MaryGUO/one-shot/mock_one_perturbed.h5ad")
    trainset, testset = preprocessing.make_total_data(total,sgRNA_list_mock)

    TrainSet = perturbdataloader(trainset, ways = args.num_ways, support_shots = args.num_shots, query_shots = 15)
    TrainLoader = DataLoader(TrainSet, batch_size=args.batch_size_train, shuffle=False,num_workers=args.num_workers)

    model = MLP(out_features = args.num_ways)

    model.to(device=args.device)
    model.train()
    meta_optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Training loop
    with tqdm(TrainLoader, total=args.num_batches) as pbar:
        for batch_idx, (inputs_support, inputs_query, target_support, target_query) in enumerate(pbar):
            model.zero_grad()

            inputs_support = inputs_support.to(device=args.device)
            target_support = target_support.to(device=args.device)

            inputs_query = inputs_query.to(device=args.device)
            target_query = target_query.to(device=args.device)

            outer_loss = torch.tensor(0., device=args.device)
            accuracy = torch.tensor(0., device=args.device)
            for task_idx, (train_input, train_target, test_input,
                           test_target) in enumerate(zip(inputs_support, target_support,inputs_query, target_query)):

                train_logit = model(train_input)
                inner_loss = F.cross_entropy(train_logit, train_target)

                model.zero_grad()
                params = gradient_update_parameters(model,
                                                    inner_loss,
                                                    step_size=args.step_size,
                                                    first_order=args.first_order)

                test_logit = model(test_input, params=params)
                outer_loss += F.cross_entropy(test_logit, test_target)

                with torch.no_grad():
                    accuracy += get_accuracy(test_logit, test_target)

            outer_loss.div_(args.batch_size_train)
            accuracy.div_(args.batch_size_train)

            outer_loss.backward()
            meta_optimizer.step()

            pbar.set_postfix(accuracy='{0:.4f}'.format(accuracy.item()))
            if batch_idx >= args.num_batches or accuracy.item() > 0.95:
                break

    # Save model
    if args.output_folder is not None:
        filename = os.path.join(args.output_folder, 'maml_omniglot_'
                                                    '{0}shot_{1}way.th'.format(args.num_shots, args.num_ways))
        with open(filename, 'wb') as f:
            state_dict = model.state_dict()
            torch.save(state_dict, f)

    # start test
    test_support, test_query, test_target_support, test_target_query \
        = helpfuntions.sample_once(testset,support_shot=args.num_shots, shuffle=False,plus = len(trainset))
    test_query = torch.from_numpy(test_query).to(device=args.device)
    test_target_query = torch.from_numpy(test_target_query).to(device=args.device)

    TrainSet = perturbdataloader_test(test_support, test_target_support)
    TrainLoader = DataLoader(TrainSet, args.batch_size_test)

    meta_optimizer.zero_grad()
    inner_losses = []
    accuracy_test = []

    for epoch in range(args.num_epoch):
        model.to(device=args.device)
        model.train()

        for _, (inputs_support,target_support) in enumerate(TrainLoader):

            inputs_support = inputs_support.to(device=args.device)
            target_support = target_support.to(device=args.device)

            train_logit = model(inputs_support)
            loss = F.cross_entropy(train_logit, target_support)
            inner_losses.append(loss)
            loss.backward()
            meta_optimizer.step()
            meta_optimizer.zero_grad()

            test_logit = model(test_query)
            with torch.no_grad():
                accuracy = get_accuracy(test_logit, test_target_query)
                accuracy_test.append(accuracy)



        if (epoch + 1) % 3 == 0:
            print('Epoch [{}/{}], Loss: {:.4f},accuray: {:.4f}'.format(epoch + 1, args.num_epoch, loss,accuracy))



if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser('Model-Agnostic Meta-Learning (MAML)')

    parser.add_argument('--folder', type=str,
                        default="/home/member/xywang/WORKSPACE/MaryGUO/pytorch-meta",
                        help='Path to the folder the data is downloaded to.')
    parser.add_argument('--num-shots', type=int, default=5,
                        help='Number of examples per class (k in "k-shot", default: 5).')
    parser.add_argument('--num-ways', type=int, default=10,
                        help='Number of classes per task (N in "N-way", default: 10).')

    parser.add_argument('--first-order', action='store_true',
                        help='Use the first-order approximation of MAML.')
    parser.add_argument('--step-size', type=float, default=0.4,
                        help='Step-size for the gradient step for adaptation (default: 0.4).')
    parser.add_argument('--hidden-size', type=int, default=64,
                        help='Number of channels for each convolutional layer (default: 64).')


    parser.add_argument('--output-folder', type=str, default= "/home/member/xywang/WORKSPACE/MaryGUO/pytorch-meta/examples/maml/output",
                        help='Path to the output folder for saving the model (optional).')
    parser.add_argument('--batch-size-train', type=int, default= 16,
                        help='Number of tasks in a mini-batch of tasks in train (default: 16).')
    parser.add_argument('--batch-size-test', type=int, default= 5,
                        help='Number of tasks in a mini-batch of tasks in test (default: 5).')
    parser.add_argument('--num-epoch', type=int, default= 30,
                        help='Number of epochs of the fine-tune step (default: 100).') #
    parser.add_argument('--num-batches', type=int, default= 3000,
                        help='Number of batches the model is trained over (default: 100).') #
    parser.add_argument('--num-workers', type=int, default=1,
                        help='Number of workers for data loading (default: 1).')
    parser.add_argument('--download', default=False,
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
