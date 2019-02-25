import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, brier_score_loss
from sklearn.calibration import calibration_curve
from sklearn.metrics.classification import _check_binary_probabilistic_predictions
from sklearn.utils import column_or_1d
from statsmodels.nonparametric.smoothers_lowess import lowess
from matplotlib.lines import Line2D
from sklearn.metrics.ranking import _binary_clf_curve


############################## General utils ##############################
    
def possibly_values(possible_pandas):
    """
    Turns pandas Series/Dataframe into a numpy array, and leaves numpy arrays intact.
        
    Parameters
    ----------
    possible_pandas: data in either pandas or numpy format
    
    Returns
    -------
    possible_pandas.values
    
    Examples of usage:
    -----------------   
    possibly_values(dataframe)
    
    """
    if isinstance(possible_pandas, (pd.DataFrame, pd.Series)):
        return(possible_pandas.values)
    else:
        return(possible_pandas)

    
def moving_average(a, n=3) :
    """
    from: https://stackoverflow.com/questions/14313510/how-to-calculate-moving-average-using-numpy
    """
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n
    
    
############################## ROC ##############################

def plot_blank_roc(ax=None, sz=18):
    
    if ax is None:
        fig, ax = plt.subplots(1,1,figsize=(6,5));
    
    ax.plot([0, 1], [0, 1], 'k--', lw=2, alpha=0.7);
    ax.set_xlabel('False positive rate (1-Specifity)', size=sz);
    ax.set_ylabel('True positive rate (Sensitivity)', size=sz);
    ax.set_title('ROC curve', size=sz+2);
    ax.set_aspect('equal');
    xticklabels = np.round(np.arange(0,1.01,0.1),1)
    ax.set_xticks(xticklabels)
    ax.set_xticklabels(xticklabels, fontdict={'fontsize':sz//1.4})    
    yticklabels = np.round(np.arange(0,1.01,0.1),1)
    ax.set_yticks(yticklabels)
    ax.set_yticklabels(yticklabels, fontdict={'fontsize':sz//1.4}) 
    ax.set_xlim(-0.04,1.04);
    ax.set_ylim(-0.04,1.04);
    
    
def plot_roc_curve(y_true, y_pred, ax=None, label_head=None, fill=False,
                   color='b', sz=18, plot_thresholds=False, return_vals=False):
    
    y_true, y_pred = possibly_values(y_true), possibly_values(y_pred)
    
    if ax is None:
        fig, ax = plt.subplots(1,1,figsize=(6,5));
    
    plot_blank_roc(ax)
    
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    
    if label_head is None:
        label = 'auROC={:0.3f}'.format(roc_auc)
    else:
        label = label_head + (': auROC={:0.3f}').format(roc_auc)
        
    ax.plot(fpr, tpr, label=label, alpha=0.7, color=color);
    ax.legend(loc='lower right', fontsize=sz//1.2);
                
    if fill:
        ax.fill_between(fpr, tpr, step='post', alpha=0.05, color=color);
        
    if plot_thresholds:
        plot_roc_thresholds(ax, fpr, tpr, thresholds,thresh_markers=[],text_positions=[])
        
    if return_vals==True:
        return fpr, tpr, thresholds
        

def plot_roc_thresholds(ax, fpr, tpr, thresholds,
                        thresh_markers=[0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
                        text_positions=[(0.02, -0.02), (0.02, -0.02), (0.02, -0.02), (0.02, -0.03),
                                        (-0.02, +0.04), (-0.02, +0.04), (-0.02, +0.03), (-0.02, +0.03), (0, +0.02)], 
                        plot_grid=False):
    
    for thresh in np.arange(0.1,0.91, 0.1):
        idx = (np.abs(thresholds - thresh)).argmin()
        x,y = fpr[idx], tpr[idx]
        ax.plot(x,y,'.', color='b', markersize=12);
        
    for thresh, text_pos in zip(thresh_markers, text_positions):
        idx = (np.abs(thresholds - thresh)).argmin()
        x,y = fpr[idx], tpr[idx]
        ax.plot(x,y,'.', color='b', markersize=12);
        ax.annotate(str(int(thresh*100))+'%', xy=(x, y), xytext=(x+text_pos[0], y+text_pos[1]), fontsize=14);
    
        if plot_grid:
            ax.axvline(x=x, color='k', ls='--', alpha=0.2);
            ax.axhline(y=y, color='k', ls='--', alpha=0.2);

            
############################## Precision Recall ##############################

def plot_blank_pr(y_true, ax=None, sz=18):
    
    if ax is None:
        fig, ax = plt.subplots(1,1,figsize=(6,5));
    
    # plot baseline - given by Precision = Prevalence:
    prevalence = np.mean(y_true)
    ax.plot([0, 1], [prevalence, prevalence], 'k--', lw=2, alpha=0.7);

    ax.set_xlabel('Recall (Sensitivity)', size=sz)
    ax.set_ylabel('Precision (PPV)', size=sz)
    ax.set_title('Precision-Recall curve', size=sz+2)
    ax.set_aspect('equal');
    xticklabels = np.round(np.arange(0,1.01,0.1),1)
    ax.set_xticks(xticklabels)
    ax.set_xticklabels(xticklabels, fontdict={'fontsize':sz//1.4})    
    yticklabels = np.round(np.arange(0,1.01,0.1),1)
    ax.set_yticks(yticklabels)
    ax.set_yticklabels(yticklabels, fontdict={'fontsize':sz//1.4})    
    ax.set_xlim(-0.04,1.04);
    ax.set_ylim(-0.04,1.04);
    
    
def plot_pr_curve(y_true, y_pred, ax=None, label_head=None, fill=False,
                   color='b', sz=18, plot_thresholds=False, return_vals=False):

    y_true, y_pred = possibly_values(y_true), possibly_values(y_pred)
    
    if ax is None:
        fig, ax = plt.subplots(1,1,figsize=(6,5))
    
    plot_blank_pr(y_true, ax)
    
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
    pr_auc = average_precision_score(y_true, y_pred)
    if label_head is None:
        label = 'auPRC={:0.3f}'.format(pr_auc)
    else:
        label = label_head + (': auPRC={:0.3f}').format(pr_auc)
    
    ax.step(recall, precision, color=color, alpha=0.7, where='post', label=label)
    
    if fill:
        ax.fill_between(recall, precision, step='post', alpha=0.05, color=color);
        
    if plot_thresholds:
        plot_pr_thresholds(ax, fpr, tpr, thresholds,thresh_markers=[],text_positions=[])
        
    ax.legend(loc='lower right', fontsize=sz//1.2)
    
    if return_vals:
        return precision, recall, thresholds

    
def plot_pr_thresholds(ax, precision, recall, thresholds,
                        thresh_markers=[0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
                        text_positions=[(0.02, -0.02), (0.02, -0.02), (0.02, -0.02), (0.02, -0.03),
                                        (-0.02, +0.04), (-0.02, +0.04), (-0.02, +0.03), (-0.02, +0.03), (0, +0.02)], 
                        plot_grid=False):
    
    for thresh in np.arange(0.1,0.91, 0.1):
        idx = (np.abs(thresholds - thresh)).argmin()
        x,y = recall[idx], precision[idx]
        ax.plot(x,y,'.', color='b', markersize=12);
        
    for thresh, text_pos in zip(thresh_markers, text_positions):
        idx = (np.abs(thresholds - thresh)).argmin()
        x,y = recall[idx], precision[idx]
        ax.plot(x,y,'.', color='b', markersize=8);
        ax.annotate(str(int(thresh*100))+'%', xy=(x, y), xytext=(x+text_pos[0], y+text_pos[1]), fontsize=14);
        
        if plot_grid:
            ax.axvline(x=x, color='k', ls='--', alpha=0.2);
            ax.axhline(y=y, color='k', ls='--', alpha=0.2);
            
            
############################## Calibration ##############################
def _calibration_curve_percentiles(y_true, y_prob, normalize=False, n_bins=5):
    """
    Copied from the original sklearn function, with alterations to use percentiles in the binning
    """
    y_true = column_or_1d(y_true)
    y_prob = column_or_1d(y_prob)

    if normalize:  # Normalize predicted values into interval [0, 1]
        y_prob = (y_prob - y_prob.min()) / (y_prob.max() - y_prob.min())
    elif y_prob.min() < 0 or y_prob.max() > 1:
        raise ValueError("y_prob has values outside [0, 1] and normalize is "
                         "set to False.")

    y_true = _check_binary_probabilistic_predictions(y_true, y_prob)

    bins = np.linspace(0., 100., n_bins + 1)
    bins = np.round(np.percentile(y_prob,bins),5)
    bins[-1] = bins[-1]+1e-6
    binids = np.digitize(y_prob, bins) - 1

    bin_sums = np.bincount(binids, weights=y_prob, minlength=len(bins))
    bin_true = np.bincount(binids, weights=y_true, minlength=len(bins))
    bin_total = np.bincount(binids, minlength=len(bins))

    nonzero = bin_total != 0
    prob_true = (bin_true[nonzero] / bin_total[nonzero])
    prob_pred = (bin_sums[nonzero] / bin_total[nonzero])

    return prob_true, prob_pred


def _calibration_curve_hist(y_true, probs, hist_bins=None, hist_colors=['r','g'],
                            ax=None, ax_hist=None, as_subplot=False, sz=18, y_lim_multiplier=3):
    """
    Standard positive/negative histogram for calibration curves
    """
    ids_positive = np.where((y_true==1))[0]
    ids_negative = np.where((y_true==0))[0]
    probs_pos = probs[ids_positive]
    probs_neg = probs[ids_negative]
    weights=[np.ones_like(probs_pos)/len(probs), np.ones_like(probs_neg)/len(probs)]
    
    if as_subplot==False:
        # if not drawn seperately, then get ax
        if ax is None:
            fig, ax_hist = plt.subplots(1,1,figsize=(6,3))
        else:
            ax_hist = ax.twinx()
    
    if hist_bins is None:
        hist_bins = np.histogram_bin_edges(probs, bins='auto', range=(0, 1))
        
    ax_hist.hist([probs_pos, probs_neg],  bins=hist_bins, weights=weights,
            color=hist_colors, alpha=0.4, histtype='bar', stacked=True);
    
    ax_hist.get_yaxis().set_visible(False)
    
    custom_lines = [Line2D([0], [0], color=hist_colors[0], lw=3, alpha=0.5),
                    Line2D([0], [0], color=hist_colors[1], lw=3, alpha=0.5)]
    
    if (as_subplot) | (ax is None):
        ax_hist.set_xlabel('Predicted probability', fontsize=sz)
        ax_hist.set_xticks(np.arange(0,1.01,0.1))
        xticklabels = [str(i)+'%' for  i in np.arange(0,101,10)] 
        ax_hist.set_xticklabels(xticklabels, fontdict={'fontsize':sz//1.4})
        ax_hist.grid()
        ax_hist.legend(custom_lines, ['Case', 'Control'], loc='upper right', fontsize=sz//1.6);
    else:
        ax_hist.set_ylim(0,y_lim_multiplier*ax_hist.get_ylim()[1])
    if ax is not None:
            ax_hist.set_xlim(ax.get_xlim())
    


def plot_calibration_curve(y_true, probs, n_bins=10, ax=None, bins_by_percentile = True, hist=_calibration_curve_hist,
                           normalize=False, color='b', show_metrics=False, plot_lowess=False,
                           sz=18, as_subplot=False, **hist_params):
    """
    Plot a calibration curve
    Taken from Andreas Muller's course: http://www.cs.columbia.edu/~amueller/comsw4995s18/

    Parameters
    ----------
    y_true : True labels
    probs: probabilites for the positive class
    n_bins: number of bins for calibration curve
    ax: axis
    bins_by_percentile: True will generate curve bins by percentiles instead of values
    hist: function with inputs (ytest, ypred, ax, **hist_params), use None to avoid plotting histogram
    normalize: True will normalize predicted values into interval [0, 1]
    color: graph color
    show_metrics: True will add a panel with some metrics. Unusable with abnormal figure sizes.
    plot_lowess: calculates a moving average and plots smooth curve by lowess method
    sz: Fontsize for plots
    as_subplot: plots clibration curve and under it a histogram as subplots

    Returns
    -------
    curve: axes of calibration curve
    brier_score: brier score result

    """
    y_true, probs = possibly_values(y_true), possibly_values(probs)
    
    if (ax is None) & (as_subplot==False):
        fig, ax = plt.subplots(1,1,figsize=(6,5));
                
    if normalize:  # Normalize predicted values into interval [0, 1]
        probs = (probs - probs.min()) / (probs.max() - probs.min())
    
    if bins_by_percentile:
        prob_true, prob_pred = _calibration_curve_percentiles(y_true, probs, n_bins=n_bins, normalize=False)
        x_lbl = 'Predicted probability'
    else:
        prob_true, prob_pred = calibration_curve(y_true, probs, n_bins=n_bins, normalize=False)
        x_lbl = 'Predicted probability (#bins='+str(n_bins)+')'
    
    if as_subplot:
        # plot calibration curve and hist as subplots
        fig, axes = plt.subplots(2,1,figsize=(6,7), sharex=False, gridspec_kw={'height_ratios':[4,1]});
        ax = axes[0]
        # plot hist
        ax_hist = axes[1]
        _calibration_curve_hist(y_true, probs,
                                ax=ax, ax_hist=ax_hist, as_subplot=True, sz=sz, y_lim_multiplier=1)
        hist=None
        fig.subplots_adjust(hspace=0.04)    
        
    if hist is not None:
        hist(y_true, probs,ax=ax, **hist_params)

    ax.plot([0, 1], [0, 1], ':', c='k', label='ideal')
    
    if plot_lowess:
        # win size is the size of bins
        win_size = len(probs)//n_bins
        # sort by probs
        sorted_inds = np.argsort(probs)
        # obtain moving aberages
        mean_x = moving_average(probs[sorted_inds], win_size)
        mean_y = moving_average(y_true[sorted_inds], win_size)
        # plot for debug purposes only
#         ax.plot(mean_x,mean_y, '.')
        # smoothe plot with lowess
        ax.plot(mean_x, lowess(mean_y,mean_x, frac=1/3)[:,1], color=color, label='non-parametric')

#         bins = np.linspace(0., 1., 1*n_bins + 1)
#         bins[-1] = bins[-1]+1e-6
#         bins_mid_x = np.array([(a+b)/2 for a,b, in zip(bins[:-1], bins[1:])])
#         # find closest bin middle and transform x
#         mean_x = pd.Series(mean_x).apply(lambda xx: bins_mid_x[np.abs(xx-bins_mid_x).argmin()]).values
#         sns.lineplot(mean_x, mean_y)

    ax.plot(prob_pred, prob_true, ls='', marker="d", markersize=10, color=color, label='grouped patients')
    
    ax.set_xlabel(x_lbl, fontsize=sz)
    ax.set_ylabel('Fraction of positive samples', fontsize=sz)
    ax.set_title('Calibration curve', fontsize=sz+2)

    # Visuals
    ax.grid()
    ax.legend(fontsize=sz//1.4)
    ax.set_xticks(np.arange(0,1.01,0.1))
    if as_subplot:
        ax.set_xlabel('')
        xticklabels = ['' for  i in np.arange(0,101,10)] 
    else:
        xticklabels = [str(i)+'%' for  i in np.arange(0,101,10)] 
    ax.set_xticklabels(xticklabels, fontdict={'fontsize':sz//1.4})
    yticklabels = np.round(np.arange(0,1.01,0.1),1)
    ax.set_yticks(yticklabels)
    ax.set_yticklabels(yticklabels, fontdict={'fontsize':sz//1.4})    
#     ax.set_aspect('equal');
    ax.set_xlim(0,1.0);
    ax.set_ylim(0,1.0);

    brier_score = brier_score_loss(y_true, probs)
    if show_metrics:
        intercept = np.mean(y_pred) - np.mean(y_true)
        fpr, tpr, roc_thresholds = roc_curve(y_true, probs)
        roc_auc = auc(fpr, tpr)

        label = "\nintercept         {:.3f}".format(slope) \
            + "\nC (ROC)     {:.3f}".format(roc_auc) \
            + "\nBrier          {:.3f}".format(brier_score) \
            
        ax.text(x=0.05, y=0.68, s=label,
                fontsize=11,
               bbox={'facecolor': 'white', 'alpha':0.7, 'pad':10})

    return(ax, brier_score)


############################## Decision curves ##############################
def _get_net_benefit_curve(y_true, probs):
    """
    Wrapper function for sklearn's _binary_clf_curve
    """
    fps, tps, thresholds = _binary_clf_curve(y_true, probs)
    n = len(probs)
    net_benefits = (tps/n) - (thresholds/(1-thresholds))*(fps/n)
    return net_benefits, thresholds


def plot_decision_curve(y_true,probs,label=None,add_AUC=False, ax=None, sz=18,
                        color='b',lw=2,linestyle='-', alpha=0.5, draw_diag=True, show_legend=True):
    """
    Plots a decision curve,
    see reference: https://www.bmj.com/content/352/bmj.i6
    
    Parameters
    ----------
    ytest: true labels
    ypred_probs: calibrated predictions
    add_AUC: adds metrics panel
    sz: fontsize
    color,lw,linestyle,alpha: line parameters
    draw_diag: True will draw the standard treat all/none/perfect performance
    show_legend: True will add a legend to the figure
    
    """
    y_true, probs = possibly_values(y_true), possibly_values(probs)
    
    if ax is None:
        fig, ax = plt.subplots(1,1,figsize=(6,5));
        
    prevalence = (np.sum(y_true==1)/len(y_true))
    
    if draw_diag:
         ## DCA of perfect
        
        ax.plot([0,1], [prevalence,prevalence],
                color='navy', ls='-.', lw=lw, alpha=alpha,
                label='Perfect hypothetical predictor')

        ## DCA of treat none
        ax.plot([0,1], [0,0],
                color='navy', ls=':', lw=lw, alpha=alpha,
                label='Treat None')

        ## DCA of treat all
        thresholds = np.sort(probs)[::-1]
        y_const = np.ones_like(y_true)
        n = len(y_true)
        tps = sum(y_const == y_true)
        fps = n - tps
        net_benefits = np.sort((tps/n) - (thresholds/(1-thresholds))*(fps/n))
        idx = np.where((net_benefits<0.6)&(net_benefits>-0.1))
        ax.plot(thresholds[idx], net_benefits[idx],
                 color='navy', ls='--', lw=lw, alpha=alpha,
                 label='Treat All')
    
    ## DCA of predictor
    net_benefit, thresholds = _get_net_benefit_curve(y_true, probs)
    ax.plot(thresholds, net_benefit,
            color=color, linestyle=linestyle, lw=lw, alpha=1,
            label=label)    
        
    ax.set_xticks(np.arange(0,1.01,0.1))
    tick_labels = [str(p)+'%' for p in np.arange(0,101,10)]
    ax.set_xticklabels(tick_labels, fontdict={'fontsize':sz//1.4})
    ax.set_xlim(0,1)
    ax.set_ylim(-prevalence/6,1.05*prevalence)

    ax.set_xlabel('Threshold probability', fontsize=sz)
    ax.set_ylabel('Net benefit', fontsize=sz)
    ax.set_title('Decision curve', fontsize=sz+2)
    ax.set_aspect('auto')
    
    if show_legend:
        ax.legend(loc='upper right', fontsize=sz//1.3);