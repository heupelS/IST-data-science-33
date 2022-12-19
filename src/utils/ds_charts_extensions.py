import numpy as np
from numpy import ndarray
from ds_charts import plot_confusion_matrix, multiple_bar_chart, HEIGHT
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score,precision_score

from matplotlib.pyplot import Axes, gca, figure, savefig, subplots, imshow, imread, axis

def plot_evaluation_results_multi_label(labels: ndarray, trn_y, prd_trn, tst_y, prd_tst):
    cnf_mtx_trn = confusion_matrix(trn_y, prd_trn, labels=labels)
    fp_trn,fn_trn,tp_trn,tn_trn = cnf_result_multilabel(cnf_mtx_trn)

    cnf_mtx_tst = confusion_matrix(tst_y, prd_tst, labels=labels)
    fp_tst,fn_tst,tp_tst,tn_tst = cnf_result_multilabel(cnf_mtx_tst)

    evaluation = {
        'Accuracy': [accuracy_score(trn_y,prd_trn), accuracy_score(tst_y,prd_tst)],
        'Recall': [recall_score(trn_y,prd_trn,labels=labels,average='macro'), recall_score(tst_y,prd_tst,labels=labels,average='macro')],
        'F1-Score': [(2 * tp_trn)/(2 * tp_trn + fp_trn + fn_trn),(2 * tp_tst)/(2 * tp_tst + fp_tst + fn_tst)],
        'Precision': [precision_score(trn_y,prd_trn,labels=labels,average='macro'), precision_score(tst_y,prd_tst,labels=labels,average='macro')]}
        
    _, axs = subplots(1, 2, figsize=(2 * HEIGHT, HEIGHT))
    multiple_bar_chart(['Train', 'Test'], evaluation, ax=axs[0], title="Model's performance over Train and Test sets", percentage=True)
    plot_confusion_matrix(cnf_mtx_tst, labels, ax=axs[1], title='Test')

def cnf_result_multilabel(cnf_mtx_trn):
    fp = np.sum(cnf_mtx_trn.sum(axis=0) - np.diag(cnf_mtx_trn)    )
    fn = np.sum(cnf_mtx_trn.sum(axis=1) - np.diag(cnf_mtx_trn)    )
    tp = np.sum(np.diag(cnf_mtx_trn)   )
    tn = np.sum(np.sum(cnf_mtx_trn) - (fp + fn + tp))
    return fp,fn,tp,tn