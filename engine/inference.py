import logging
from scipy.stats import pearsonr
from sklearn.metrics import f1_score
import torch

from yacs.config import CfgNode
from roberta_model import RobertaISTS
from torch.utils.data import DataLoader


def inference(cfg: CfgNode, model: RobertaISTS, val_loader: DataLoader) -> None:
    device_type = cfg.MODEL.DEVICE
    device = torch.device(device_type)
    log_period = cfg.SOLVER.LOG_PERIOD
    logger = logging.getLogger("model.inference")
    logger.info("Start inferencing")

    gold_val = []
    gold_exp = []
    pred_val = []
    pred_exp = []
    model.eval()
    model.to(device)

    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            inputs, value = data[0].to(device), data[1].to(device)
            out_vector = model(inputs)
            _, predicted = torch.max(out_vector, 1)

            splitted = torch.split(value, 6, dim=1)
            val = splitted[0]
            exp = splitted[1]
            
            pred_split = torch.split(out_vector, 6, dim=1)
            p_val = pred_split[0]
            p_exp = pred_split[1]

            gold_val.append(torch.argmax(val).cpu())
            gold_exp.append(torch.argmax(exp).cpu())

           
            #gold_exp.append(explanation.item())
            pred_val.append(torch.argmax(p_val).cpu())
            pred_exp.append(torch.argmax(p_exp).cpu())

            #pred_exp.append(predicted.item())

            if i % log_period == log_period - 1:
                logger.info('Progress [%d/%d]' % (i + 1, len(val_loader)))\
                
                # break
                
    
   
    logger.info('| F1 score for explanations: %.3f' % f1_score(gold_exp, pred_exp, average='micro'))
    #logger.info('| Pearson for explanations: {:.3f}'.format(pearsonr(gold_exp, pred_exp)[0]))
    # round_to_whole = [round(num) for num in pred_val]
    logger.info('| F1 score for values: %.3f' % f1_score(gold_val, pred_val, average='micro'))
    # logger.info('| Pearson for values: {:.3f}'.format(pearsonr(gold_val, pred_val)[0]))

