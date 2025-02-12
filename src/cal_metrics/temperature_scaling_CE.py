import torch
from torch import nn, optim
from torch.nn import functional as F
import pickle
import torch.nn.functional as f
import cal_metrics.ECE as ECE

from torch.utils.data import TensorDataset, DataLoader

class Temperature(nn.Module):
    """
    A thin decorator, which wraps a model with temperature scaling
    model (nn.Module):
        A classification neural network
        NB: Output of the neural network should be the classification logits,
            NOT the softmax (or log softmax)!
    """
    def __init__(self):
        super(Temperature, self).__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits):
        return self.temperature_scale(logits)
    
    def return_changes(self, logits):
        return logits - self.temperature_scale(logits)

    def temperature_scale(self, logits):
        """
        Perform temperature scaling on logits
        """
        # Expand temperature to match the size of logits
        
        try:
            temperature = self.temperature.unsqueeze(1).expand(logits.size(0), logits.size(1))
        except:
            print(logits.shape)
            print(self.temperature.unsqueeze(1).shape)
            assert(0)
        return logits / temperature

    # This function probably should live outside of this class, but whatever
    def set_temperature(self, logits, labels):
        """
        Tune the tempearature of the model (using the validation set).
        We're going to set it to optimize NLL.
        valid_loader (DataLoader): validation set loader
        """
        self.cuda()
        CE_criterion = nn.CrossEntropyLoss().cuda()
        ece_criterion = _ECELoss(n_bins=25).cuda()

        # Calculate NLL and ECE before temperature scaling
        before_temperature_nll = CE_criterion(logits, labels).item()
        before_temperature_ece = ece_criterion(logits, labels).item()
        print('Before temperature - NLL: %.3f, ECE: %.3f' % (before_temperature_nll, before_temperature_ece))

        # Next: optimize the temperature w.r.t. NLL
        optimizer = optim.LBFGS([self.temperature], lr=0.01, max_iter=50)

        def eval():
            optimizer.zero_grad()
            loss = CE_criterion(self.temperature_scale(logits), labels)
            loss.backward()
            return loss
        optimizer.step(eval)

        # Calculate NLL and ECE after temperature scaling
        after_temperature_nll = CE_criterion(self.temperature_scale(logits), labels).item()
        after_temperature_ece = ece_criterion(self.temperature_scale(logits), labels).item()
        print('Optimal temperature: %.3f' % self.temperature.item())
        print('After temperature - NLL: %.3f, ECE: %.3f' % (after_temperature_nll, after_temperature_ece))

        return self

class Temperature_adaptive(nn.Module):
    """
    A thin decorator, which wraps a model with temperature scaling
    model (nn.Module):
        A classification neural network
        NB: Output of the neural network should be the classification logits,
            NOT the softmax (or log softmax)!
    """
    def __init__(self, input_dim=10, feat_dim = 5):
        super(Temperature_adaptive, self).__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1)
        self.input_dim = input_dim
        self.feat_dim = feat_dim

        self.embed_layer = nn.Sequential(
            nn.Linear(self.input_dim, self.feat_dim),
            nn.ReLU(),
            # nn.Linear(self.feat_dim, self.feat_dim),
            # nn.ReLU(),
        )
        self.embed_layer2 = nn.Sequential(
            nn.Linear(self.input_dim, self.feat_dim),
            nn.ReLU(),
            # nn.Linear(self.feat_dim, self.feat_dim),
            # nn.ReLU(),
        )
        self.correctness_layer = nn.Sequential(
            nn.Linear(self.feat_dim, 1),
            nn.Sigmoid()
        )

        self.temperature_layer = nn.Linear(self.feat_dim, 1)


    def forward(self, logits, temp_arg, return_temp=False, correctness=False):

        return self.temperature_scale(logits, temp_arg, return_temp=return_temp, correctness=correctness)
    
    def return_changes(self, logits):
        return logits - self.temperature_scale(logits)

    def temperature_scale(self, logits, temp_arg, return_temp=False, correctness=False):
        """
        Perform temperature scaling on logits
        """
        # Expand temperature to match the size of logits
        feat = self.embed_layer(temp_arg)
        feat2 = self.embed_layer2(temp_arg)
        temperature = torch.clamp(self.temperature_layer(feat), 0.1, 10)
        correctness_out = self.correctness_layer(feat2).squeeze(dim=-1)
        
        # corr_conf = 2*(correctness_out - 0.5).abs()
        # temperature = 1+ (temperature-1)*corr_conf
        
        # correctness_out = torch.cat((temperature, 1/temperature), dim=-1)
        temperature = temperature.expand(logits.size(0), logits.size(1)) 
        # correctness_out = F.softmax(logits/temperature, dim=1).max(dim=1)[0]
        
        cal_logits = logits / temperature
        if len(logits)>2000:
            corre_pred = correctness_out > 0.5
            corr_conf = 0.5+(correctness_out - 0.5).abs().detach()
            logit_conf = F.softmax(logits, dim=1).max(dim=1)[0]
            ambigu_idx = (((temperature[:, 0]>1) & (correctness_out > 0.5)) | ((temperature[:, 0]<1) & (correctness_out < 0.5)))
            # ambigu_idx = corr_conf<0.6
            # ambigu_idx = ambigu_idx | ambigu_idx2
            # print(logit_conf.mean())
            # print(corr_conf.mean())
            # assert(0)
            # ambigu_idx = corr_conf < 0.7
            # print(correctness_out[:10])
            print('ambigu ratio:', ambigu_idx.sum()/len(ambigu_idx))
            # assert(0)
            # cal_logits[ambigu_idx] = logits[ambigu_idx]

        # 将数据保存到pickle文件
        if return_temp and correctness:
            return cal_logits, temperature[:, 0], correctness_out
        elif return_temp:
            return cal_logits, temperature[:, 0]
        elif correctness:
            return cal_logits, correctness_out
        else:
            return cal_logits

    # This function probably should live outside of this class, but whatever
    def set_temperature(self, logits, temp_arg, labels):
        """
        Tune the tempearature of the model (using the validation set).
        We're going to set it to optimize NLL.
        valid_loader (DataLoader): validation set loader
        """
        CE_criterion = nn.CrossEntropyLoss(reduction='none').cuda()
        nll_criterion = nn.NLLLoss(reduction='none').cuda()
        mse_criterion = nn.MSELoss().cuda()
        ece_criterion = _ECELoss(n_bins=25).cuda()
        ece_criterion_conf = ECE._ECELoss(n_bins=25)
        mse_criterion = nn.MSELoss(reduction='none').cuda()  # mseloss使confidence和对错接近
        correctness_criterion = nn.BCELoss(reduction='none').cuda()  # 二分类问题使用二元交叉熵损失
        # correctness_criterion = nn.CrossEntropyLoss(reduction='none')  # 二分类问题使用二元交叉熵损失


        # Calculate NLL and ECE before temperature scaling
        before_temperature_nll = CE_criterion(logits, labels).mean().item()
        before_temperature_ece = ece_criterion(logits, labels).mean().item()
        print('Before temperature - NLL: %.3f, ECE: %.3f' % (before_temperature_nll, before_temperature_ece))

        # Next: optimize the temperature w.r.t. NLL

        # parameters = torch.cat([param.view(-1) for param in self.temperature_layer.parameters()])
        # optimizer = optim.LBFGS(self.parameters(), lr=0.001, max_iter=50)
        optimizer = optim.Adam(self.parameters(), lr=0.0001)
        
        # 将数据和标签封装成 TensorDataset
        dataset = TensorDataset(torch.cat((logits, temp_arg), dim=-1), labels)

        # 创建数据加载器
        batch_size = 1000
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


        self.cuda()

        for epoch in range(500):
            self.train()
            print_flag = True
            for batch_data, batch_labels in data_loader:
                # 清除梯度
                optimizer.zero_grad()
                # 前向传播
                batch_labels = batch_labels.cuda()
                b_logits = batch_data[:, :-self.input_dim].cuda()
                _, first_predicts = b_logits.max(dim=-1)
                logit_conf = F.softmax(b_logits, dim=1).max(dim=1)[0]

                #处理imbalance问题
                correctness = torch.eq(first_predicts, batch_labels)
                weight_correct = correctness.float().sum()/len(correctness)
                weights = torch.zeros(len(correctness)).cuda()
                # weights[correctness] = 1
                weights[correctness] = 1-weight_correct
                # weights[~correctness] = 1
                weights[~correctness] = weight_correct

                b_temp_arg = batch_data[:, -self.input_dim:].cuda()
                outputs, temperature, correctness_out = self(b_logits.cuda(), b_temp_arg.cuda(), return_temp=True, correctness=True)
                softmax_vector = F.softmax(outputs, dim=1)
                conf_out = softmax_vector.max(dim=1)[0]
                
                # logSoftmax_v = F.log_softmax(outputs, dim=1)
                # loss_cal_p = - logSoftmax_v[torch.arange(len(outputs)), batch_labels][correctness]
                # loss_cal_n = - torch.log(1-conf_out[~correctness]+ 1e-6)
                # loss_cal = loss_cal_p.mean()+ loss_cal_n.mean()
                
                # loss_cal = nll_criterion(logSoftmax_v, batch_labels)
                
                # if epoch>10:
                #     loss_cal = nll_criterion(logSoftmax_v, batch_labels)
                # else:
                #     neg_logSoftmax_v = torch.log(1-torch.exp(logSoftmax_v[~correctness]))
                #     pos_logSoftmax_v = logSoftmax_v[correctness]
                #     loss_cal = nll_criterion(torch.cat((neg_logSoftmax_v, pos_logSoftmax_v), dim=0), torch.cat((first_predicts[~correctness], batch_labels[correctness]), dim=0))

                loss_cal = CE_criterion(outputs, batch_labels)
                # loss_cal  = correctness_criterion(conf_out, correctness.float())
                # conf_label = correctness.float()
                # conf_label[~correctness] = 1/conf_out.shape[-1]
                loss_mse = mse_criterion(conf_out, correctness.float())
                # loss_mse = (conf_out-logit_conf)[~correctness].mean()-(conf_out-logit_conf)[correctness].mean()
                loss_correctness = correctness_criterion(correctness_out, correctness.float())
                loss_consist = mse_criterion(conf_out, correctness_out)
                # correctness_acc =  (correctness_out.max(-1)[1] == correctness.long()).sum()/len(correctness)
                correctness_acc =  ((correctness_out>0.5).long() == correctness.long()).sum()/len(correctness)
                # if epoch % 50 == 0 and print_flag:  
                #     wrongThigh_idx = correctness & (temperature>1)
                #     corretThigh_idx = ~correctness & (temperature>1)
                #     wrongTlow_idx = ~correctness & (temperature<1)
                #     corretTlow_idx = correctness & (temperature<1)
                #     print(correctness_out[wrongThigh_idx].mean())
                #     print(correctness_out[corretThigh_idx].mean())
                #     print(correctness_out[wrongTlow_idx].mean())
                #     print(correctness_out[corretTlow_idx].mean())
           
                # loss_temperature_regular = temperature + 1/temperature
                # loss_temperature = (1/temperature[~correctness]).mean() + temperature[correctness].mean() + (temperature[~correctness].mean() - temperature[correctness].mean()).abs()
                # loss_temperature = temperature[correctness_out>0.5].mean() + 1/temperature[~(correctness_out>0.5)].mean()
                alpha = correctness.sum()/len(correctness)
                loss = loss_cal

                # loss = loss*weights
                # loss = loss.sum()/weights.sum()
                loss = loss.mean()
                # loss += loss_temperature
            
                if epoch % 50 == 0 and print_flag:  
                    # print(epoch, loss_cal.mean().item(), loss_correctness.mean().item(), loss_temperature.float().mean().item())
                    print(epoch, loss_cal.mean().item(), loss_mse.mean().item(), loss_correctness.mean().item(), correctness_acc.item(), correctness.sum().item())
                    # print(epoch, loss_correctness.mean().item(), correctness_acc)
                    print_flag = False
                loss = loss.mean()
                loss.backward()
                optimizer.step()

        # Calculate NLL and ECE after temperature scaling
        self.eval()
        logits_cal, correctness_out = self.temperature_scale(logits, temp_arg, correctness=True)
        after_temperature_nll = CE_criterion(logits_cal, labels).mean().item()
        after_temperature_ece = ece_criterion(logits_cal, labels).mean().item()
        correctness = F.softmax(logits, dim=1).max(dim=1)[1].eq(labels)
        after_correctnessACC = correctness_criterion(correctness_out, correctness.float()).mean().item()
        print('Optimal temperature: %.3f' % self.temperature.item())
        print('After temperature - NLL: %.3f, ECE: %.3f' % (after_temperature_nll, after_temperature_ece))
        print('After temperature - NLL: %.3f, ECE: %.3f' % (after_temperature_nll, after_correctnessACC))

        return self

class _ECELoss(nn.Module):
    """
    Calculates the Expected Calibration Error of a model.
    (This isn't necessary for temperature scaling, just a cool metric).

    The input to this loss is the logits of a model, NOT the softmax scores.

    This divides the confidence outputs into equally-sized interval bins.
    In each bin, we compute the confidence gap:

    bin_gap = | avg_confidence_in_bin - accuracy_in_bin |

    We then return a weighted average of the gaps, based on the number
    of samples in each bin

    See: Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
    "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI.
    2015.
    """
    def __init__(self, n_bins=15):
        """
        n_bins (int): number of confidence interval bins
        """
        super(_ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels):
        softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece