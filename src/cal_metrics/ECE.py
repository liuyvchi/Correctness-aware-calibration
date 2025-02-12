import numpy as np
import torch
import matplotlib.pyplot as plt

def plot_weighted_histogram(average_accuracies, average_confidences, cal=False):
   # 假设你有每个confidence bin区间对应的样本的average confidence和average accuracy数据
    confidence_bins = np.arange(0, 1.00, 0.04)  # confidence bin的区间

    # 绘制直方图
    plt.plot(confidence_bins, average_confidences, color='blue', marker='o', label='Average Confidence')
    plt.plot(confidence_bins, average_accuracies, color='orange', marker='x', label='Average Accuracy')

    plt.title('Average Confidence and Accuracy by Confidence Bins')
    plt.xlabel('Confidence Bins')
    plt.ylabel('Values')
    plt.legend()  # 显示图例

    # 保存图表为图片文件
    if cal: 
        plt.savefig('ece_histogram_cal_TS.png') 
    else:  
        plt.savefig('ece_histogram.png')

    # 关闭绘图窗口
    plt.close()

class _ECELoss():
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
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]
        self.bin_numbers = {}
        

    def eval(self, confidences, accuracies, assign_index=False, cal=False, plot=False):
        confidence_list, accuracy_list, prop_list = [], [], []
        
        ece = np.zeros(1)
        sort_index = np.argsort(confidences)
        # if assign_value:
        #     confidences = confidences[sort_index]
        #     accuracies = accuracies[sort_index]
        bin_number_count = 0
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.__gt__(bin_lower) * confidences.__le__(bin_upper)
            if assign_index:
                self.bin_numbers[str(bin_lower)] = len(confidences[in_bin])
            prop_in_bin = len(confidences[in_bin])/len(confidences)
            if prop_in_bin > 0:
                accuracy_in_bin = accuracies[in_bin].astype(float).mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                # if assign_value:
                #     print(accuracy_in_bin)
                #     print(avg_confidence_in_bin)
                bin_ece_pre = np.abs(avg_confidence_in_bin - accuracy_in_bin)
                bin_ece =  bin_ece_pre * prop_in_bin
                ece += bin_ece
                confidence_list.append(avg_confidence_in_bin)
                accuracy_list.append(accuracy_in_bin)
                prop_list.append(prop_in_bin)
            else:
                confidence_list.append(0)
                accuracy_list.append(0)
                prop_list.append(0)
        if plot:
            plot_weighted_histogram(accuracy_list, confidence_list, cal)

        return ece