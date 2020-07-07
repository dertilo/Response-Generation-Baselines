import argparse
import json
import math
import random
from model import Transformer
import sys

from collections import Counter
from tqdm import tqdm


def str2bool(v):
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


# fmt:off
parser = argparse.ArgumentParser(description='Topical-Chat Training Script')

parser.add_argument('--seed', type=int, default=42, metavar='S', help='random seed (default: 42)')

parser.add_argument('--epoch', nargs='+',type=int, default=[20])
parser.add_argument('--batch_size', type=int, default=8, metavar='N')
parser.add_argument('--use_attn', type=str2bool, const=True, nargs='?', default=False)

parser.add_argument('--emb_size', type=int, default=300)
parser.add_argument('--hid_size', type=int, default=300)
parser.add_argument('--num_layers', type=int, default=1)
parser.add_argument('--dropout', type=float, default=0.2)
parser.add_argument('--lr', type=float, default=0.005)
parser.add_argument('--l2_norm', type=float, default=0.00001)
parser.add_argument('--clip', type=float, default=5.0, help='clip the gradient by norm')

parser.add_argument('--seq2seq', type=str2bool, const=True, nargs='?', default=False)

parser.add_argument('--use_knowledge', type=str2bool, const=True, nargs='?', default=True)

parser.add_argument('--data_path', type=str, default='processed_output/')
parser.add_argument('--data_size', type=float, default=-1.0)
parser.add_argument('--save_path', type=str, default='save/')
# fmt:on
args = parser.parse_args()

if not args.data_path.endswith("/"):
    args.data_path = args.data_path + "/"

if not args.save_path.endswith("/"):
    args.save_path = args.save_path + "/"


def load_data(split):
    src = [l.strip() for l in open(args.data_path + split + ".src").readlines()]
    tgt = [l.strip() for l in open(args.data_path + split + ".tgt").readlines()]
    fct = [l.strip() for l in open(args.data_path + split + ".fct").readlines()]
    return list(zip(src, tgt, fct))


# Build vocabulary
i2w = [w.strip() for w in open(args.save_path + "vocab.txt").readlines()]
w2i = {w: i for i, w in enumerate(i2w)}


def run_validation(epoch, dataset_name: str):
    dataset = load_data(dataset_name)
    print("Number of %s instances: %d" % (dataset_name, len(dataset)))

    model = Transformer(
        i2w=i2w, use_knowledge=args.use_knowledge, args=args, test=True
    ).cuda()
    model.load("{0}model_{1}.bin".format(args.save_path, epoch))
    model.transformer.eval()
    # Iterate over batches
    num_batches = math.ceil(len(dataset) / args.batch_size)
    cum_loss = 0
    cum_words = 0
    predicted_sentences = []
    indices = list(range(len(dataset)))
    for batch in tqdm(range(num_batches)):
        # Prepare batch
        batch_indices = indices[batch * args.batch_size : (batch + 1) * args.batch_size]
        batch_rows = [dataset[i] for i in batch_indices]

        # Encode batch. If facts are being used, they'll be prepended to the input
        input_seq, input_lens, target_seq, target_lens = model.prep_batch(batch_rows)

        # Decode batch
        predicted_sentences += model.decode(input_seq, input_lens)

        # Evaluate batch
        cum_loss += model.eval_ppl(input_seq, input_lens, target_seq, target_lens)
        cum_words += (target_seq != w2i["_pad"]).sum().item()

        # Log epoch
    ppl = math.exp(cum_loss / cum_words)
    print("{} Epoch: {} PPL: {}".format(dataset_name, epoch, ppl))
    # Save predictions
    open(
        "{0}{1}_epoch_{2}.pred".format(args.save_path, dataset_name, str(epoch)), "w+"
    ).writelines([l + "\n" for l in predicted_sentences])


for ep in args.epoch:
    # run_validation(ep, "valid_freq")
    # run_validation(ep, "valid_rare")
    run_validation(ep, "test_freq")
    run_validation(ep, "test_rare")
