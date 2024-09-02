#!/usr/bin/python
#-*- coding: utf-8 -*-

import sys, time, os, argparse
import yaml
import numpy
import torch
import glob
import zipfile
import warnings
import datetime
from tuneThreshold import *
from SpeakerNet import *
from DatasetLoader import *
import torch.distributed as dist
import torch.multiprocessing as mp
import shutil
import matplotlib.pyplot as plt
from utils import score2class2, score2class4, score2class6
warnings.simplefilter("ignore")
from torch.utils.tensorboard import SummaryWriter

MODEL_LIST = ['ecapa_tdnn', 'ecapa_tdnn_sv', 'hubert_large_ll60k', 'xls_r_300m', 'unispeech_sat', "wavlm_base_plus", "wavlm_large", "wavlm_large_sv"]

## ===== ===== ===== ===== ===== ===== ===== =====
## Parse arguments
## ===== ===== ===== ===== ===== ===== ===== =====

parser = argparse.ArgumentParser(description = "SpeakerNet")

## Data loader
parser.add_argument('--max_frames',     type=int,   default=300,    help='Input length to the network for training')
parser.add_argument('--eval_frames',    type=int,   default=400,    help='Input length to the network for testing 0 uses the whole files')
parser.add_argument('--batch_size',     type=int,   default=128,    help='Batch size, number of speakers per batch')
parser.add_argument('--nDataLoaderThread', type=int, default=8,     help='Number of loader threads')
parser.add_argument('--augment',        type=bool,  default=False,  help='Augment input')
parser.add_argument('--seed',           type=int,   default=10,     help='Seed for the random number generator')
parser.add_argument('--max_label',      type=int,   default=6,     help='Maximum value of labels')

## Training details
parser.add_argument('--test_interval',  type=int,   default=1,     help='Test and save every [test_interval] epochs')
parser.add_argument('--max_epoch',      type=int,   default=10,    help='Maximum number of epochs')
parser.add_argument('--trainfunc',      type=str,   default='mse',  help='Loss function')

## Optimizer and Scheduler
parser.add_argument('--optimizer',      type=str,   default="adam", help='sgd or adam')
parser.add_argument('--lr',             type=float, default=1e-3,  help='Learning rate')
parser.add_argument("--lr_decay",       type=float, default=0.95,   help='Learning rate decay every [test_interval] epochs')
parser.add_argument('--weight_decay',   type=float, default=1e-3,      help='Weight decay in the optimizer')

parser.add_argument('--scheduler',      type=str,   default="steplr", help='Learning rate scheduler')
parser.add_argument('--step_size',      type=int,   default=1000,     help='step size for iterative scheduler')
parser.add_argument('--first_cycle_steps',  type=int,   default=2500, help='First cycle step size')
parser.add_argument('--cycle_mult',         type=float, default=1.0, help='Cycle steps magnification')
parser.add_argument('--max_lr',             type=float, default=0.1, help="First cycle's max learning rate")
parser.add_argument('--min_lr',             type=float, default=0.001, help='Min learning rate')
parser.add_argument('--warmup_steps',       type=int,   default=0, help='Linear warmup step size')
parser.add_argument('--gamma',              type=float, default=1.0, help='Decrease rate of max learning rate by cycle')

## Load and save
parser.add_argument('--initial_model',  type=str,   default="",     help='Initial model weights')
parser.add_argument('--save_path',      type=str,   default="exps/exp1", help='Path for model and logs')

## Training and test data
parser.add_argument('--train_list',     type=str,   default="data/voxsim_train_list_raw.txt",  help='Train list')
parser.add_argument('--test_list',      type=str,   default="data/voxsim_test_list.txt",   help='Evaluation list')
parser.add_argument('--train_path',     type=str,   default="data/voxceleb1", help='Absolute path to the train set')
parser.add_argument('--test_path',      type=str,   default="data/voxceleb1", help='Absolute path to the test set')
parser.add_argument('--musan_path',     type=str,   default="data/musan_split", help='Absolute path to the test set')
parser.add_argument('--rir_path',       type=str,   default="data/simulated_rirs", help='Absolute path to the test set')

## Model definition
parser.add_argument('--n_mels',         type=int,   default=80,     help='Number of mel filterbanks')
parser.add_argument('--mlp',            type=int,   default=None,    nargs='*', help='use mlp')
parser.add_argument('--model',          type=str,   default='wavlm_large', help='Input feature type for ECAPA-TDNN')
parser.add_argument('--update_extract', type=bool,  default=False,  help='Whether to update extractor model or not')
parser.add_argument('--emb_dim',        type=int,   default=256,    help='Output embedding size')

## For test only
parser.add_argument('--eval',           dest='eval', action='store_true', help='Eval only')
parser.add_argument('--nClasses',       type=int,   default=2,   help='num classes for calculating accuracy')
parser.add_argument('--threshold',      type=float,   default=0.5, help='thresholds for calculating accuracy')

## Distributed and mixed precision training
parser.add_argument('--port',           type=str,   default="8888", help='Port for distributed training, input as text')
parser.add_argument('--distributed',    dest='distributed', action='store_true', help='Enable distributed training')
parser.add_argument('--mixedprec',      dest='mixedprec',   action='store_true', help='Enable mixed precision training')

args = parser.parse_args()

## ===== ===== ===== ===== ===== ===== ===== =====
## Trainer script
## ===== ===== ===== ===== ===== ===== ===== =====

def main_worker(gpu, ngpus_per_node, args):
    
    args.gpu = gpu
    
    assert args.model in MODEL_LIST, 'The model_name should be in {}'.format(MODEL_LIST)
    
    ## Load models
    s = SpeakerNet(**vars(args))
    
    if args.distributed:
        os.environ['MASTER_ADDR']='localhost'
        os.environ['MASTER_PORT']=args.port

        dist.init_process_group(backend='nccl', world_size=ngpus_per_node, rank=args.gpu)

        torch.cuda.set_device(args.gpu)
        s.cuda(args.gpu)

        s = torch.nn.parallel.DistributedDataParallel(s, device_ids=[args.gpu], find_unused_parameters=True)

        print('Loaded the model on GPU {:d}'.format(args.gpu))

    else:
        s = WrappedModel(s).cuda(args.gpu)

    it = 1
    best_pearson = 0

    if args.gpu == 0:
        ## Write args to scorefile
        scorefile   = open(args.result_save_path+"/scores.txt", "a+")
            
    if args.eval == False:
        ## Initialise trainer and data loader
        train_dataset = train_dataset_loader(**vars(args))

        if args.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True, seed=args.seed, drop_last=True)
            drop_last = False
            shuffle = False
        else:
            train_sampler = None
            drop_last = True
            shuffle = True

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            num_workers=args.nDataLoaderThread,
            sampler=train_sampler,
            pin_memory=False,
            worker_init_fn=worker_init_fn,
            drop_last=drop_last,
            shuffle=shuffle,
        )

    trainer     = ModelTrainer(s, **vars(args))

    ## Load model weights
    modelfiles = glob.glob('%s/model0*.model'%args.model_save_path)
    modelfiles.sort()

    if(args.initial_model != ""):
        trainer.loadParameters(args.initial_model)
        print("Model {} loaded!".format(args.initial_model))
    elif len(modelfiles) >= 1:
        trainer.loadParameters(modelfiles[-1])
        print("Model {} loaded from previous state!".format(modelfiles[-1]))
        it = int(os.path.splitext(os.path.basename(modelfiles[-1]))[0][5:]) + 1
        for line in reversed(scorefile.readlines()):
            line_info = line.strip().split(',')
            if (it-1) == int(line_info[0].split()[-1]) and 'bestPearson' in line_info[-1]:
                best_pearson = float(line_info.split()[-1])
                break
    
    if trainer.lr_step == 'epoch':
        for ii in range(1,it):
            trainer.__scheduler__.step()
    
    ## Evaluation code - must run on single GPU
    if args.eval == True:

        pytorch_total_params = sum(p.numel() for p in s.module.parameters())

        print('Total parameters: ',pytorch_total_params)
        print('Test list',args.test_list)
        
        all_scores, all_labels, all_results = trainer.evaluateFromList(**vars(args))

        if args.gpu == 0:

            pearson = pearsonr(all_scores, all_labels)[0]
            spearman = spearmanr(all_scores, all_labels)[0]
            r2 = r2_score(all_labels, all_scores)
            rmse = mean_squared_error(all_labels, all_scores, squared=False)

            acc = 0
            pos_label = []
            pos_score = []
            neg_label = []
            neg_score = []
            for i in range(len(all_scores)):
                if abs(all_scores[i] - all_labels[i]) < args.threshold:
                    acc += 1
                    pos_label.append(all_labels[i])
                    pos_score.append(all_scores[i])
                else:
                    neg_label.append(all_labels[i])
                    neg_score.append(all_scores[i])
            accuracy = acc / len(all_labels)
            
            plt.scatter(pos_label, pos_score, s=10, facecolors='none', edgecolors='b')
            plt.scatter(neg_label, neg_score, s=10, facecolors='none', edgecolors='r')
            plt.plot([min(all_labels), max(all_labels)], [min(all_labels), max(all_labels)], 'r-')
            plt.xlabel('Label')
            plt.ylabel('Prediction')

            plt.savefig(os.path.join(args.save_path,'results.png'))

            with open(os.path.join(args.save_path, 'results.txt'), 'w') as f:
                f.write('wav1 wav2 label pred\n')
                for result in all_results:
                    f.write(result + '\n')
            
            avg_scores = sum(all_scores) / len(all_scores)
            print("Average score: ", avg_scores)
            
            scorefile.write("Model {} loaded!\n".format(args.initial_model))
            scorefile.write('{} Pearson {:2.5f} Spearman {:2.5f} R2 {:2.5f} RMSE {:2.5} Acc {:2.3}'.format(time.strftime("%Y-%m-%d %H:%M:%S"), pearson, spearman, r2, rmse, accuracy))

            print('\n',time.strftime("%Y-%m-%d %H:%M:%S"), 
                "Pearson {:2.5f}".format(pearson), 
                "Spearman {:2.5f}".format(spearman),
                "R2 {:2.5f}".format(r2), 
                "RMSE {:2.5f}".format(rmse),
                "Acc {:2.5f}".format(accuracy * 100))

        return

    shutil.copy2('trainSpeakerNet.py', args.save_path)
    shutil.copy2('SpeakerNet.py', args.save_path)
    shutil.copy2('DatasetLoader.py', args.save_path)
    shutil.copy2('optimizer/' + args.optimizer + '.py', args.save_path)
    shutil.copy2('scheduler/' + args.scheduler + '.py', args.save_path)
    shutil.copy2('models/SSL_ECAPA.py', args.save_path)

    ## Save training code and params
    if args.gpu == 0:
        pyfiles = glob.glob('./*.py')
        strtime = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

        zipf = zipfile.ZipFile(args.result_save_path+ '/run%s.zip'%strtime, 'w', zipfile.ZIP_DEFLATED)
        for file in pyfiles:
            zipf.write(file)
        zipf.close()

        with open(args.result_save_path + '/run%s.cmd'%strtime, 'w') as f:
            for items in vars(args):
                f.write('{} {}\n'.format(items, vars(args)[items]))
            f.write('Number of parameters: {}\n'.format(sum(p.numel() for p in s.module.parameters())))
    
    writer_path = os.path.join(args.save_path, 'tb_logs')
    writer = SummaryWriter(writer_path)


    for it in range(it,args.max_epoch+1):
        
        if args.distributed:
            train_sampler.set_epoch(it)

        clr = [x['lr'] for x in trainer.__optimizer__.param_groups]
        
        loss, pearson = trainer.train_network(train_loader, writer, verbose=(args.gpu == 0))

        if args.gpu == 0:
            writer.add_scalar('TLoss_epoch', loss, it)
            writer.add_scalar('TPearson', pearson, it)
            print('\n',time.strftime("%Y-%m-%d %H:%M:%S"), "Epoch {:d}, LR {:.9f}, TLOSS {:f}, Pearson {:1.3f}".format(it, max(clr), loss, pearson))
            scorefile.write("Epoch {:d}, LR {:f}, TLOSS {:f}, Pearson {:1.3f} \n".format(it, max(clr), loss, pearson))

        if it % args.test_interval == 0:
            
            all_scores, all_labels, _ = trainer.evaluateFromList(**vars(args))

            if args.gpu == 0:

                pearson = pearsonr(all_scores, all_labels)[0]
                r2 = r2_score(all_labels, all_scores)
                rmse = mean_squared_error(all_labels, all_scores, squared=False)
                
                if pearson > best_pearson:
                    best_pearson = pearson

                print('\n',time.strftime("%Y-%m-%d %H:%M:%S"), "Epoch {:d}, R2 {:1.3f}, RMSE {:1.3f}, Pearson {:1.3f}, bestPearson {:1.3f}".format(it, r2, rmse, pearson, best_pearson))
                scorefile.write("Epoch {:d}, R2 {:1.3f}, RMSE {:1.3f}, Pearson {:1.4f}, bestPearson {:1.4f}\n".format(it, r2, rmse, pearson, best_pearson))

                scorefile.flush()
                
                trainer.saveParameters(args.model_save_path+"/model%09d.model"%it)

                writer.add_scalar('VPearson_epoch', pearson, it)
                writer.add_scalar('VRMSE_epoch', rmse, it)

    if args.gpu == 0:
        scorefile.close()

    if args.gpu == 0:
        scorefile.close()

## ===== ===== ===== ===== ===== ===== ===== =====
## Main function
## ===== ===== ===== ===== ===== ===== ===== =====

def main():
    args.model_save_path     = args.save_path+"/model"
    args.result_save_path    = args.save_path+"/result"
    args.feat_save_path      = ""

    if os.path.exists(args.save_path):
        answer = input('Current dir exists, do you want to remove and refresh it?\n')
        if answer in ['yes','y','ok','1']:
            print('Dir removed !')
            shutil.rmtree(args.save_path)
            os.makedirs(args.save_path)
        else:
            print('Dir Not removed !')
    
    os.makedirs(args.model_save_path, exist_ok=True)
    os.makedirs(args.result_save_path, exist_ok=True)

    n_gpus = torch.cuda.device_count()

    print('Python Version:', sys.version)
    print('PyTorch Version:', torch.__version__)
    print('Number of GPUs:', torch.cuda.device_count())
    print('Save path:',args.save_path)

    if args.distributed:
        mp.spawn(main_worker, nprocs=n_gpus, args=(n_gpus, args))
    else:
        main_worker(0, None, args)


if __name__ == '__main__':
    main()