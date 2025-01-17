import logging
import torch
import datetime

from yacs.config import CfgNode
from roberta_model import RobertaISTS
from torch.utils.data import DataLoader
from torch.optim import Optimizer


def do_train(cfg: CfgNode, model: RobertaISTS, train_loader: DataLoader, optimizer: Optimizer, losses: list) -> None:
    output_dir = cfg.OUTPUT_DIR

    device_type = cfg.MODEL.DEVICE
    device = torch.device(device_type)
    epochs = cfg.SOLVER.MAX_EPOCHS
    log_period = cfg.SOLVER.LOG_PERIOD

    logger = logging.getLogger("model.train")
    logger.info("Start training")
    model.to(device)


    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        running_loss1 = 0.0
        running_loss2 = 0.0

        partial_loss1 = 0.0
        partial_loss2 = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, value, explanation = data[0].to(device), data[1].to(device), data[2].to(device)
            #inputs = inputs.to(device)
            # forward + backward + optimize
            out1, out2 = model(inputs[0])

            explanation = explanation.type(torch.LongTensor).to(device)
            loss1 = losses[0](out1, value)
            loss2 = losses[1](out2, explanation)
            loss = loss1 + loss2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            running_loss1 += loss1.item()
            running_loss2 += loss2.item()
            partial_loss1 += loss1.item()
            partial_loss2 += loss2.item()
            if i % log_period == log_period - 1:
                logger.info('EPOCH: [%d/%d] BATCHES [%d/%d] loss summed: %.3f loss MSE: %.3f loss NLLL: %.3f' %
                            (epoch + 1, epochs, i + 1, len(train_loader), (partial_loss1 + partial_loss2) / log_period,
                             partial_loss1 / log_period, partial_loss2 / log_period))
                partial_loss1 = 0.0
                partial_loss2 = 0.0

        logger.info('EPOCH: [%d] FINISHED loss summed: %.3f loss MSE: %.3f loss NLLL: %.3f' %
                    (epoch + 1, running_loss / len(train_loader), running_loss1 / len(train_loader),
                     running_loss2 / len(train_loader)))

        logger.info('Finished Training')
        logger.info('Saving model ...')
        output_filename = output_dir + '/' + datetime.datetime.now().strftime("%d%m%Y%H%M%S") + '_model.pt'
        torch.save(model.state_dict(), output_filename)
        logger.info('Model saved as :' + output_filename)

