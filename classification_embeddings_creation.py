"""
1) embedding extractions of interactive RNA pairs
2) classification using these embeddings

baseline: 
python classification_embeddings_creation.py --out_path 'classification/db_FINAL_STRAT/classification_baseline_training.p' --test_data_path 'classification/db_FINAL_STRAT/train_instances' --ckpt_path 'model.pt' --tokenizer_path 'tokenizer'

python classification_embeddings_creation.py --out_path 'classification/db_FINAL_STRAT/classification_baseline_test.p' --test_data_path 'classification/db_FINAL_STRAT/test_instances' --ckpt_path 'model.pt' --tokenizer_path 'tokenizer'

aug:
python classification_embeddings_creation.py --out_path 'classification/db_FINAL_STRAT/classification_aug_training.p' --test_data_path 'classification/db_FINAL_STRAT/train_instances' --ckpt_path 'model.pt' --tokenizer_path 'tokenizer'

python classification_embeddings_creation.py --out_path 'classification/db_FINAL_STRAT/classification_aug_test.p' --test_data_path 'classification/db_FINAL_STRAT/test_instances' --ckpt_path 'model.pt' --tokenizer_path 'tokenizer'

aug types:
python classification_embeddings_creation.py --out_path 'classification/db_FINAL_STRAT/classification_aug_types_training.p' --test_data_path 'classification/db_FINAL_STRAT/train_instances' --ckpt_path 'model.pt' --tokenizer_path 'tokenizer'

python classification_embeddings_creation.py --out_path 'classification/db_FINAL_STRAT/classification_aug_types_test.p' --test_data_path 'classification/db_FINAL_STRAT/test_instances' --ckpt_path 'model.pt' --tokenizer_path 'tokenizer'


types rna-fm comp

python classification_embeddings_creation.py --out_path 'classification/db_RNA-FM-comparison/classification_aug_types_training.p' --test_data_path 'classification/db_RNA-FM-comparison/train_instances-filter1022' --ckpt_path 'model.pt' --tokenizer_path 'tokenizer'

python classification_embeddings_creation.py --out_path 'classification/db_RNA-FM-comparison/classification_aug_types_test.p' --test_data_path 'classification/db_RNA-FM-comparison/test_instances-filter1022' --ckpt_path 'model.pt' --tokenizer_path 'tokenizer'



rna-fm mirna-lncrna
python classification_embeddings_creation.py --out_path 'classification/single_interactions_RNA_FM/classification_mirna-lncrna_aug_types_training.p' --test_data_path 'classification/single_interactions_RNA_FM/fm-single-interactions/train_instances-mirna_lncrna-filter1022' --ckpt_path 'model.pt' --tokenizer_path 'tokenizer'

python classification_embeddings_creation.py --out_path 'classification/single_interactions_RNA_FM/classification_mirna-lncrna_aug_types_test.p' --test_data_path 'classification/single_interactions_RNA_FM/fm-single-interactions/test_instances-mirna_lncrna-filter1022' --ckpt_path 'model.pt' --tokenizer_path 'tokenizer'

rna-fm mirna-mirna
python classification_embeddings_creation.py --out_path 'classification/single_interactions_RNA_FM/classification_mirna-mirna_aug_types_training.p' --test_data_path 'classification/single_interactions_RNA_FM/fm-single-interactions/train_instances-mirna_mirna-filter1022' --ckpt_path 'model.pt' --tokenizer_path 'tokenizer'

python classification_embeddings_creation.py --out_path 'classification/single_interactions_RNA_FM/classification_mirna-mirna_aug_types_test.p' --test_data_path 'classification/single_interactions_RNA_FM/fm-single-interactions/test_instances-mirna_mirna-filter1022' --ckpt_path 'model.pt' --tokenizer_path 'tokenizer'

rna-fm mirna-snorna
python classification_embeddings_creation.py --out_path 'classification/single_interactions_RNA_FM/classification_mirna-snorna_aug_types_training.p' --test_data_path 'classification/single_interactions_RNA_FM/fm-single-interactions/train_instances-mirna_snorna-filter1022' --ckpt_path 'model.pt' --tokenizer_path 'tokenizer'

python classification_embeddings_creation.py --out_path 'classification/single_interactions_RNA_FM/classification_mirna-snorna_aug_types_test.p' --test_data_path 'classification/single_interactions_RNA_FM/fm-single-interactions/test_instances-mirna_snorna-filter1022' --ckpt_path 'model.pt' --tokenizer_path 'tokenizer'

"""
special_cases_check = True
augmented_positives = True
augmented_negatives = True
use_rna_type_for_negatives = False

# Keep track: always true
keep_type_track = True
testing = True

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import pickle
from contextlib import nullcontext
import torch
import re
from model import GPTConfig, GPT
from tqdm import tqdm
import random
import numpy as np
from transformers import AutoTokenizer
from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode
import argparse
import itertools
import random
from itertools import combinations
from multiprocessing import Pool, Manager, cpu_count

parser = argparse.ArgumentParser()
parser.add_argument("--init_from", type=str, default="resume", help="Directory of raw data & output files")
parser.add_argument("--out_path", type=str, required=True)
parser.add_argument("--test_data_path", type=str, required=True)
parser.add_argument("--strategy",type=str, required=False,default='top_k',help="should be in ['greedy_search', 'sampling', 'top_k', 'beam_search']")
parser.add_argument("--temperature",type=float, required=False,default=1.0,help="1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions")
parser.add_argument("--top_k",type=int, required=False,default=1,help="retain only the top_k most likely tokens, clamp others to have 0 probability")
parser.add_argument("--ckpt_path",type=str, required=True,help="path to a checkpoint/model")
parser.add_argument("--tokenizer_path",type=str, required=True,help="path to a tokenizer directory")
parser.add_argument("--repetition_penalty",type=float, required=False,default=1.0)

args = parser.parse_args()
init_from = args.init_from
out_path = args.out_path
test_data_path = args.test_data_path + '-25.02.txt'
test_data_path_no_augmentation = args.test_data_path + '-25.02-noaugmentation.txt'
strategy = args.strategy
temperature = args.temperature
top_k = args.top_k
ckpt_path = args.ckpt_path
tokenizer_path = args.tokenizer_path
repetition_penalty = args.repetition_penalty

# -----------------------------------------------------------------------------
seed = 1337
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = 'float32'
# dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
compile = False # use PyTorch 2.0 to compile the model to be faster
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
torch.manual_seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# model
if init_from == 'resume':
    # init from a model saved in a specific directory
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
elif init_from.startswith('gpt2'):
    # init from a given GPT-2 model
    model = GPT.from_pretrained(init_from, dict(dropout=0.0))

model.eval()
model.to(device)
if compile:
    model = torch.compile(model) # requires PyTorch 2.0 (optional)
encode = tokenizer.encode
decode = tokenizer.decode


# functions: ----------------------------------------------------------------------------------------
def worker_task_guided(elements, true_pairs_set, true_pairs_sublist, target_size, progress_queue, 
                       true_pairs_dict, rna_probabilities, keep_type_track, checking_types):
    """
    Worker function to generate unique random pairs.
    Args:
        elements (list): List of elements to sample from.
        true_pairs_set (set): Set of true pairs to exclude.
        true_pairs_sublist (list): Sublist of true pairs assigned to the worker.
        target_size (int): Number of pairs to generate.
        progress_queue (Queue): Queue for tracking progress.
    Returns:
        list: A list of sampled pairs.
    """
    sampled_pairs = []
    sampled_pairs_with_keys = []
    print('checking types: ', checking_types)
    print('keep_type_track', True if keep_type_track else False)
    for true_pair in true_pairs_sublist:
        counter = 0
        trial = 0
        while counter < target_size:
            if checking_types:
                type_seq1 = true_pair[1][0]
                #print('type_seq1', type_seq1)
                if type_seq1 in rna_probabilities:
                    probabilities = rna_probabilities[type_seq1][0]
                    #print('probabilities', probabilities)
                    labels = rna_probabilities[type_seq1][1]
                    #print('labels', labels)
                else:
                    print(f"Warning: {type_seq1} not found in rna_probabilities")
                    break  # Skip this iteration
                # Sample an index based on the probability distribution
                sampled_index = np.random.choice(len(probabilities), p=probabilities)
                # Retrieve the corresponding label
                sample_seq2_type = labels[sampled_index]
                #print('sample_seq2_type', sample_seq2_type)
                if sample_seq2_type == 'not classified':
                    sample_seq2_type = 'ncRNA'
                if sample_seq2_type == 'sncRNA':
                    break
                #print('true_pairs_dict', sample_seq2_type in true_pairs_dict)
                list_to_sample = true_pairs_dict[sample_seq2_type]
                if not list_to_sample:
                    print(f"Warning: No elements found for {sample_seq2_type}")
                    break  # Skip this iteration
                sampled_element = random.choice(list_to_sample)
                #print('sampled_element', len(sampled_element))
                pair = (true_pair[0][0], sampled_element) # true_pair: ((seq1, seq2), (ty1, ty2))
                if keep_type_track:
                    key_pair = (type_seq1, sample_seq2_type)
            else:
                sampled_element = random.choice(elements)
                if keep_type_track:
                    pair = (true_pair[0][0], sampled_element)
                    key_pair = (true_pair[1][0], keep_type_track[sampled_element])
                else:
                    pair = (true_pair[0], sampled_element)
                
            pair_check = frozenset(pair)
            if pair not in sampled_pairs:
                if pair_check not in true_pairs_set:
                    trial = 0
                    if keep_type_track: 
                        sampled_pairs_with_keys.append((pair, key_pair))
                    sampled_pairs.append(pair)
                    counter += 1
                    if counter >= target_size:
                        break
            trial += 1
            #print('counter', counter)
            if trial >= 10:
                print('trial done for the true couple')
                break
    progress_queue.put(1)  # Signal progress
    if testing: 
        return sampled_pairs_with_keys
    return sampled_pairs

def sample_combinations_guided_parallel(elements, true_pairs, special_cases, 
                                        rna_probabilities, true_pairs_keys, 
                                        true_pairs_dict, keep_type_track=False,
                                        sampling_types=False,
                                        sample_quantity=20):
    """
    Parallelized sampling of combinations, excluding true pairs.
    Args:
        elements (list): List of elements to form combinations from.
        true_pairs (list): List of true interacting pairs (to exclude).
        special_cases (list): List of special cases to be excluded
        sample_quantity (int): Number of how many negatives to find for each true_pairs[i]
        rna_probabilities 
        true_pairs_keys 
        true_pairs_dict
        keep_type_track if true return with the negatives, their keys (RNA types)
    Returns:
        list: A list of sampled combinations.
    """
    n_true = len(true_pairs)
    print('Negative sample size: ', n_true * sample_quantity)
    # Convert true pairs to a set for quick exclusion
    true_pairs_set = set(frozenset(pair) for pair in true_pairs)
    print('true_pairs_set: ', len(true_pairs_set))
    #print('true_pairs_set: ', true_pairs_set)
    special_cases_set = set(frozenset(pair) for pair in special_cases)
    #print('special_cases_set', special_cases_set)
    print('special_cases_set', len(special_cases_set))
    true_pairs_set = true_pairs_set.union(special_cases_set) # join true pairs and special cases
    #print('true_pairs_set union', true_pairs_set)
    print('true_pairs_set union', len(true_pairs_set))
    # Determine number of workers and task size per worker
    num_workers = 10
    print('num_workers: ', num_workers)
    task_size = n_true // num_workers

    for key in true_pairs_dict.keys():
        print('key', key)
        print('len list', len(true_pairs_dict[key]))

    true_pairs_with_type = []
    if keep_type_track:
        index = 0
        for pair in true_pairs:
            types = true_pairs_keys[index]
            true_pairs_with_type.append((pair, types)) # lista di coppie di due coppie: [((seq1, seq2), (type1, type2)), ...]
            index += 1
        true_pairs = true_pairs_with_type

    print('true_pairs_with_type: ', len(true_pairs_with_type))

    # Split true_pairs into chunks for each worker
    true_pairs_chunks = [
        true_pairs[i * task_size:(i + 1) * task_size] if i < num_workers - 1 else true_pairs[i * task_size:]
        for i in range(num_workers)
    ]
    print('true_pairs_chunks: ', len(true_pairs_chunks))

    # Setup for multiprocessing
    with Manager() as manager:
        progress_queue = manager.Queue()
        with Pool(processes=num_workers) as pool:
            # Create a shared progress bar
            with tqdm(total=num_workers, desc="Sampling Pairs") as pbar:
                # Map worker tasks
                results = [
                    pool.apply_async(
                        worker_task_guided,
                        args=(elements, true_pairs_set, chunk, sample_quantity, 
                              progress_queue, true_pairs_dict, rna_probabilities, keep_type_track, sampling_types)
                    )
                    for chunk in true_pairs_chunks
                ]

                # Update progress bar in the main process
                completed = 0
                while completed < num_workers:
                    progress_queue.get()  # Block until a progress signal is received
                    completed += 1
                    pbar.update(1)

                # Combine results from all workers
                sampled_pairs = []
                for result in results:
                    sampled_pairs.extend(result.get())

    return sampled_pairs

def combined_pooling(embeddings):
    # Max pooling along the sequence length dimension (dim=1)
    max_pooled, _ = embeddings.max(dim=1)  # Result: (1, 1024)
    # Average pooling along the sequence length dimension (dim=1)
    avg_pooled = embeddings.mean(dim=1)  # Result: (1, 1024)
    # Concatenating max-pooled and avg-pooled results along the feature dimension (dim=-1)
    combined_embeddings = torch.cat([max_pooled, avg_pooled], dim=-1)  # Result: (1, 2048)
    # Print shapes to verify
    #print('embeddings: ', embeddings)
    #print("Max-pooled shape:", max_pooled.shape)
    #print("Max-pooled shape:", max_pooled)
    #print("Avg-pooled shape:", avg_pooled.shape)
    #print("Avg-pooled shape:", avg_pooled)
    #print("Combined embeddings shape:", combined_embeddings.shape)
    return combined_embeddings

def extract_keys(pairs):
    tag_dictionary = {
        'miRNA' : ['B', 'D'], #OK sncRNA
        'lncRNA' : ['Y'], #OK lncRNA
        'pseudo' : ['B', 'K'], #OK sncRNA
        'rRNA' : ['B', 'M'], #OK sncRNA
        'not classified' : [], # OK
        'snoRNA' : ['B', 'N'], #OK sncRNA
        'circRNA' : ['Y', 'R'], #OK lncRNA
        'scRNA' : ['B', 'S'], #OK sncRNA
        'ncRNA' : [], # OK
        'snRNA' : ['B', 'V'], # OK sncRNA
        'ribozyme' : ['B', 'W'], #OK sncRNA
        'scaRNA' : ['B', 'X'], #OK sncRNA
        'tRF' : ['B', 'Ċ'], #OK sncRNA
        'piRNA' : ['B', 'Ġ'], #OK sncRNA
        'sncRNA' : ['B'] # OK sncRNA
    }
    # Create a new dictionary with frozensets as keys:
    flipped_dictionary = {}
    for rna_type, tags in tag_dictionary.items():
        # Convert the list of tags into a frozenset:
        tags_fs = frozenset(tags)
        # Add or update the flipped dictionary:
        if tags_fs in flipped_dictionary:
            flipped_dictionary[tags_fs].append(rna_type)
        else:
            flipped_dictionary[tags_fs] = [rna_type]
    # Display the flipped dictionary:
    #for fs_key, rna_types in flipped_dictionary.items():
    #    print(f"{fs_key}: {rna_types}")
    pairs_nokeys = []
    keys_list = []
    keys_dict = {}
    for pair in pairs:
        ligand = pair[0]
        target = pair[1]
        flag = False
        key1 = []
        if ligand:
            while flag == False and len(ligand) > 1:
                if ligand[0] not in ['A', 'U', 'G', 'C']:
                    key1.append(ligand[0])
                    ligand = ligand[1:] # we cut the keyword in ligand here
                else:
                    flag = True
        if len(ligand) <= 1:
            continue
        flag = False
        key2 = []
        if target:
            while flag == False and len(target) > 1:
                if target[0] not in ['A', 'U', 'G', 'C']:
                    key2.append(target[0])
                    target = target[1:] # we cut the keyword in target here
                else:
                    flag = True
        if len(target) <= 1:
            continue

        pair_filtered = (ligand, target)
        pairs_nokeys.append(pair_filtered)

        if not key1:
            rna_type1 = 'ncRNA'
        else:
            fs_key1 = frozenset(key1)
            # Look up the corresponding RNA type for each extracted key.
            # If the key is not found, we default to None.
            rna_type1 = flipped_dictionary.get(fs_key1, [None])[0]

        if not key2:
            rna_type2 = 'ncRNA'
        else:
            fs_key2 = frozenset(key2)
            rna_type2 = flipped_dictionary.get(fs_key2, [None])[0]

        if rna_type1 in keys_dict:
            keys_dict[rna_type1].append(ligand)
        else:
            keys_dict[rna_type1] = [ligand]

        if rna_type2 in keys_dict:
            keys_dict[rna_type2].append(target)
        else:
            keys_dict[rna_type2] = [target]
        
        # Append a tuple (rna_type1, rna_type2) to keys_list.
        keys_list.append((rna_type1, rna_type2))
        if rna_type1 == [None]:
            print('something wrong 1')
        if rna_type2 == [None]:
            print('something wrong 2')
        # removing dict duplicates
        for key in keys_dict.keys():
            keys_dict[key] = list(set(keys_dict[key]))
    return pairs_nokeys, keys_list, keys_dict

def load_rna_probs():
    # classification/db_RNA-FM-comparison/rna_couples_probabilities_filter1022.pkl
    # classification/db_FINAL_STRAT/rna_couples_probabilities.pkl
    with open('classification/db_RNA-FM-comparison/rna_couples_probabilities_filter1022.pkl', 'rb') as pickle_file:
        loaded_rna_probabilities = pickle.load(pickle_file)
    return loaded_rna_probabilities

def extract_negatives_keys(negatives_with_keys):
    # original list: [((seq1, seq2), (type1, type2)), ...]
    keys_list_tuples = []
    negatives_list_tuples = []
    for element in negatives_with_keys:
        negatives_list_tuples.append(element[0])
        keys_list_tuples.append(element[1])
    return negatives_list_tuples, keys_list_tuples
# functions END  ----------------------------------------------------------------------------------------

# Here we load the testing database
with open(test_data_path, 'r') as file:
    #true_pairs = [line.strip().split('X') for line in file.readlines()]
    true_pairs = [re.split(r'(?<!B)X', line.strip()) for line in file]

true_pairs = [[pair[0], pair[1]] for pair in true_pairs]
print('true_pairs len: ', len(true_pairs))
with open(test_data_path_no_augmentation, 'r') as file:
    #true_pairs_noaug = [line.strip().split('X') for line in file.readlines()]
    true_pairs_noaug = [re.split(r'(?<!B)X', line.strip()) for line in file]
true_pairs_no_aug = [[pair[0], pair[1]] for pair in true_pairs_noaug]
print('true_pairs_no_aug len: ', len(true_pairs_no_aug))

if augmented_positives:
    print('We are using augmented positives')
else:
    print('We are NOT using augmented positives')

if augmented_negatives:
    print('We are creating negatives starting from augmented positives')
else:
    print('We are creating negatives starting from NOT augmented positives')

rna_probabilities = load_rna_probs()

#------------------------------------------
# TESTING ONLY
#print('---------------WE ARE TESTING------------------')
#true_pairs = true_pairs[:1000]
#true_pairs_no_aug = true_pairs_no_aug[:1000]
#print('true_pairs: ', true_pairs)

print('len(true_pairs): ', len(true_pairs))
print('len(true_pairs_no_aug): ', len(true_pairs_no_aug))

true_pairs, true_pairs_keys, true_pairs_dict = extract_keys(true_pairs)

true_pairs_no_aug, true_pairs_no_aug_keys, true_pairs_no_aug_dict = extract_keys(true_pairs_no_aug)

print('len(true_pairs): ', len(true_pairs))
print('len(true_pairs_no_aug): ', len(true_pairs_no_aug))

# Negatives creation:
print('Negative CREATION ---------------------')
true_pairs_no_aug_special_cases = []
for pair in true_pairs_no_aug:
    element_1 = pair[0]
    element_1_flipped = element_1[::-1]# FLIP the sequence
    element_2 = pair[1]
    element_2_flipped = element_2[::-1]# FLIP the sequence
    true_pairs_no_aug_special_cases.append((element_1, element_2_flipped))
    true_pairs_no_aug_special_cases.append((element_2, element_1_flipped))
    true_pairs_no_aug_special_cases.append((element_1, element_1_flipped))
    true_pairs_no_aug_special_cases.append((element_2, element_2_flipped))
    true_pairs_no_aug_special_cases.append((element_1, element_1))
    true_pairs_no_aug_special_cases.append((element_2, element_2))

if special_cases_check:
    print('Checking for special cases')
else:
    true_pairs_no_aug_special_cases = []
    print('No special cases usage/check')

# Step 1: Extract all unique elements
elements = list(set([item for pair in true_pairs for item in pair]))
print('Unique all_elements: ', len(elements))
elements_not_aug = list(set([item for pair in true_pairs_no_aug for item in pair]))
print('Unique all_elements_no_aug: ', len(elements_not_aug))

if augmented_negatives:
    elements_to_sample = elements
    true_pairs_to_sample = true_pairs
    print('Using augmented positives to construct negatives')
else:
    elements_to_sample = elements_not_aug
    true_pairs_to_sample = true_pairs_no_aug
    true_pairs_keys = true_pairs_no_aug_keys
    true_pairs_dict = true_pairs_no_aug_dict
    print('Using NOT augmented positives to construct negatives')

if keep_type_track:
    elements_to_sample_dict = {}
    for sequence in elements:
        # search for RNA type
        for key in true_pairs_dict.keys():
            if sequence in true_pairs_dict[key]:
                elements_to_sample_dict[sequence] = key
    keep_type_track = elements_to_sample_dict  

# Step 2: Find non-interacting pairs, Sample 10 negatives for each positive pair
non_interacting_pairs = sample_combinations_guided_parallel(elements_to_sample, true_pairs_to_sample, 
                                                            true_pairs_no_aug_special_cases, rna_probabilities, 
                                                            true_pairs_keys, true_pairs_dict, keep_type_track=keep_type_track,
                                                            sampling_types=use_rna_type_for_negatives)
#if keep_type_track:
print('non_interacting_pairs: ', len(non_interacting_pairs))

# Remove duplicates while treating (A, B) as different from (B, A)
non_interacting_pairs = list(set(non_interacting_pairs))
print('non_interacting_pairs, without duplicates caused by parallel computation: ', len(non_interacting_pairs))

if testing:
    if keep_type_track:
        non_interacting_pairs, negatives_pairs_keys = extract_negatives_keys(non_interacting_pairs)

if augmented_positives:
    print('Using augmented positives as positives')
else:
    true_pairs = true_pairs_no_aug
    print('Using NOT augmented positives as positives')

# DOPO DEVO AVERE TRUE_PAIRS e NON_INTERACTIVE_PAIRS come positivi e negativi.
name_file = out_path.split('/')[-1][:-2]
# Step 3: Save non-interacting pairs to a pickle file
output_file = f"non_interacting_pairs{name_file}.pkl"
with open(output_file, "wb") as f:
    pickle.dump(non_interacting_pairs, f)
print(f"Non-interacting pairs saved to {output_file}")

output_file = f"interacting_pairs{name_file}.pkl"
with open(output_file, "wb") as f:
    pickle.dump(true_pairs, f)
print(f"interacting pairs saved to {output_file}")

output_file = f"elements_unique{name_file}.pkl"
with open(output_file, "wb") as f:
    pickle.dump(elements, f)
print(f"Unique elements saved to {output_file}")


#----------------------------------------

# for every test instance we have interacting molechule: A, than ligand: 'X' + B natural true molechule interacting with A

# First, we extract all the embeddings of all the unique sequences:
unique_seq_embeddings = {}
for element in elements:
    sequence = element
    # TRUE LOGITS
    target_ids = encode("".join(sequence))
    x_target = (torch.tensor(target_ids, dtype=torch.long, device=device)[None, ...])
    with torch.no_grad():
        with ctx:
            logits_target = combined_pooling(model.extract_embeddings(x_target))
            logits_target = logits_target.detach()
            # ADD IN THE DICTIONARY
            unique_seq_embeddings[sequence] = logits_target

len_true_pairs = len(true_pairs)
output_full = []

# check that keys has same len of true data
if keep_type_track:
    assert len_true_pairs == len(true_pairs_keys)

# DATASET: 1 positive, from this #n) negatives.
index = 0
with tqdm(total=len_true_pairs, desc="Extracting positive embeddings") as pbar:
    for target, ligand in true_pairs:
        pbar.update(1)
        output_in_file = {'true':None}
        # TRUE LOGITS
        logits_target = unique_seq_embeddings[target]
        logits_ligand = unique_seq_embeddings[ligand]
        if keep_type_track and testing: 
            output_in_file['true'] = ([logits_target, logits_ligand], true_pairs_keys[index])
            index += 1
        else:
            output_in_file['true'] = [logits_target, logits_ligand]
        output_full.append(output_in_file)

print('output_in_file - only positives: ', len(output_full))

# negative pairs
index = 0
len_neg_pairs = len(non_interacting_pairs)
with tqdm(total=len_neg_pairs, desc="Extracting negative embeddings") as pbar:
    for target, ligand in non_interacting_pairs:
        pbar.update(1)
        output_in_file = {'negative':None}
        # negative LOGITS
        logits_target = unique_seq_embeddings[target]
        logits_ligand = unique_seq_embeddings[ligand]
        if keep_type_track and testing: 
            output_in_file['negative'] = ([logits_target, logits_ligand], negatives_pairs_keys[index])
            index += 1
        else:
            output_in_file['negative'] = [logits_target, logits_ligand]
        output_full.append(output_in_file)
    print('output_in_file - positives AND negatives: ', len(output_full))
   
with open(out_path, 'wb') as f:
    pickle.dump(output_full, f)



