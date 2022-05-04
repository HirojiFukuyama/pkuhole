# -*- coding: utf-8 -*-
"""
This script is for training on the boya platform.
"""

# necessary imports
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch import optim
from tqdm import tqdm
from dataload import *
from lgg_model import *
import os
from argparse import ArgumentParser
import time
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# print("Is GPU available?", torch.cuda.is_available())
logger.info("Is GPU available? {}".format(torch.cuda.is_available()))

def main():
    # parse arguments
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, default='../dataset/hole-merge.txt')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lr_decay', type=float, default=0.98)
    parser.add_argument('--embed_size', type=int, default=512)
    parser.add_argument('--hidden_size', type=int, default=512)
    parser.add_argument('--n_layers', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--save_wv', type=bool, default=False)
    parser.add_argument('--input_word_count', type=int, default=30)
    parser.add_argument('--train_in_all', type=float, default=0.8)

    args = parser.parse_args()

    save_model = args.save_wv

    # generate dataset and dataloader
    BS = args.batch_size
    input_word_count = args.input_word_count
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # print("The device is: ", device)
    logger.info("The device is: {}".format(device))

    data_path = args.dataset
    full_dataset = MyDataset(data_path, input_word_count, save_word_model=False) # set the save option to True for the first time
    vocabulary_length = full_dataset.vocabulary_length
    # print("The vocabulary length is: ", vocabulary_length)
    logger.info("The vocabulary length is: {}".format(vocabulary_length))

    train_in_all = args.train_in_all # train_in_all is the proportion of the dataset that will be used for training

    train_size = int(train_in_all * len(full_dataset))
    val_size = len(full_dataset) - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    train_dataloader = DataLoader(train_dataset, batch_size=BS, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BS, shuffle=True)

    from datetime import datetime as dt

    now = str(dt.now())
    time_path = now[:10] + "_" + now[11:13] + "_" + now[14:16] + "_" + now[17:19]
    # print("The time is: ", time_path)
    logger.info("The time is: {}".format(time_path))
    if save_model:
        try:
            full_dataset.word_model.save('checkpoints/' + time_path + '_word_model')
        except:
            logger.info('Word model not saved!')

    # some components in the training process
    LR = args.lr # the learning rate of 0.001 is still too large, maybe needs lr_decay or batch_norm
    num_epoches = args.epochs
    net = LSTM_enhanced(vocabulary_length, args.embed_size, args.hidden_size, args.n_layers, dropout=args.dropout).to(device)
    optimizer = optim.Adam(net.parameters(), lr=LR) 
    criterion = nn.CrossEntropyLoss()

    # 2022/2/27 add a lr decay controller
    lr_decay_rate = args.lr_decay
    ctrl = torch.optim.lr_scheduler.ExponentialLR(optimizer, lr_decay_rate)

    # # print the hyperparameters
    # print("The hyperparameters are as follows:")
    # print("The dataset is:", data_path)
    # print("The learning rate is:", LR)
    # print("The number of epochs is:", num_epoches)
    # print("The batch size is:", BS)
    # print("The input word count is:", input_word_count)
    # print("The vocabulary length is:", vocabulary_length)
    # print("The ratio of train/val is:", int(train_in_all/(1-train_in_all)), ":", 1)
    # print("The network is:", net)
    # print("The device is:", device)
    # print("The optimizer is:", optimizer)
    # print("The criterion is:", criterion)
    # print("The lr decay schedule is exponential and the lr decay rate is:", lr_decay_rate)
    logger.info("The hyperparameters are as follows:")
    logger.info("The dataset is: {}".format(data_path))
    logger.info("The learning rate is: {}".format(LR))
    logger.info("The number of epochs is: {}".format(num_epoches))
    logger.info("The batch size is: {}".format(BS))
    logger.info("The input word count is: {}".format(input_word_count))
    logger.info("The vocabulary length is: {}".format(vocabulary_length))
    logger.info("The ratio of train/val is: {}:1".format(int(train_in_all/(1-train_in_all))))
    logger.info("The network is: {}".format(net))
    logger.info("The device is: {}".format(device))
    logger.info("The optimizer is: {}".format(optimizer))
    logger.info("The criterion is: {}".format(criterion))
    logger.info("The lr decay schedule is exponential and the lr decay rate is: {}".format(lr_decay_rate))

    # start training!
    # print("Start training!")
    # print('-' * 65)
    logger.info("Start training!")
    logger.info('-' * 65)
    for epoch in range(num_epoches):
        start_time = time.time()
        # train
        net.train()
        train_loss = 0
        for i, data in enumerate(train_dataloader):
            data = data.to(device)
            data = data.to(torch.long)
            label = data[:,1:]
            out = net(data)[:,:-1,:]
            out = torch.transpose(out, 2, 1)

            optimizer.zero_grad()
            loss = criterion(out, label)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()

        train_avg_loss = train_loss / len(train_dataloader)
        # print("Epoch: {}/{}".format(epoch+1, num_epoches), "train_loss: {:.4f}".format(train_avg_loss), "time: {:.4f}s".format(time.time() - start_time))
        logger.info("Epoch: {}/{} train_loss: {:.4f} time: {:.4f}s".format(epoch+1, num_epoches, train_avg_loss, time.time() - start_time))
        ctrl.step() # lr decay
        
        # validation
        net.eval()
        val_loss = 0
        with torch.no_grad():
            for i, data in enumerate(val_dataloader):
                data = data.to(device)
                data = data.to(torch.long)
                label = data[:,1:]
                out = net(data)[:,:-1,:]
                out = torch.transpose(out, 2, 1)
                loss = criterion(out, label)
                val_loss += loss.item()
        
        val_avg_loss = val_loss / len(val_dataloader)
        # print("Epoch: {}/{}".format(epoch+1, num_epoches), "val_loss:   {:.4f}".format(val_avg_loss), "time: {:.4f}s".format(time.time() - start_time))
        # print('-' * 65)
        logger.info("Epoch: {}/{} val_loss:   {:.4f} time: {:.4f}s".format(epoch+1, num_epoches, val_avg_loss, time.time() - start_time))

        if (epoch+1) % 5 == 0:
            try:
                torch.save(net, 'checkpoints/' + "Epoch_" + str(epoch+1)) # save once every 5 epochs
            except:
                logger.info('Checkpoint not saved!')

        logger.info('-' * 65)
        
    # print("Finish training!")
    logger.info("Finish training!")

    # if you want to save your language model...
    # model_name: input_word_count, hidden_size, n_layers, epochs, time_path
    # e.g. hole-merge_30_512_3_50_2022-05-04_03_17_17
    model_name = "hole-merge_" + str(input_word_count) + "_" + str(args.hidden_size) + "_" + str(args.n_layers) + "_" + str(args.epochs)
    model_name = model_name + '_' + time_path
    try:
        torch.save(net, "checkpoints/" + model_name)
    except:
        logger.info('Model not saved!')

    try:
        torch.save(net, "../model/" + model_name)
        # print("The model is saved as: ", model_name)
        logger.info("The final model is saved as: {}".format(model_name))
    except:
        # print("Final model not saved!")
        logger.info("Final model not saved!")

    # print("Now the learning rate is:", optimizer.param_groups[0]['lr'])
    # print("Now the train loss is:", train_avg_loss)
    # print("Now the val loss is:", val_avg_loss)
    logger.info("Now the learning rate is: {}".format(optimizer.param_groups[0]['lr']))
    logger.info("Now the train loss is: {}".format(train_avg_loss))
    logger.info("Now the val loss is: {}".format(val_avg_loss))


if __name__ == "__main__":
    main()