import os, sys

from numpy import ndarray
from numpy import argsort, arange, std
from pandas import DataFrame, read_csv, unique

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, subplots, savefig, show
from matplotlib.pyplot import Axes

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

sys.path.append( os.path.join(os.path.dirname(__file__),'..') )
from ds_charts import plot_evaluation_results, multiple_line_chart, HEIGHT
from ds_charts import horizontal_bar_chart
from ds_charts import plot_overfitting_study
from ds_charts_extensions import plot_evaluation_results_multi_label
from general_utils import get_plot_folder_path


###########################################
## Select if plots show up of just saved ##
SHOW_PLOTS = False
###########################################

def RF(df: DataFrame, target_name: str, save_file_name: str):

    print('-- Calculate RF')

    y: ndarray = df.pop(target_name).values
    X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2, random_state=0)

    labels = unique(y_train)
    labels.sort()

    n_estimators = [5, 10, 25, 50, 75, 100, 200, 300, 400]
    max_depths = [5, 10, 25]
    max_features = [.3, .5, .7, 1]
    best = ('', 0, 0)
    last_best = 0
    best_model = None

    cols = len(max_depths)

    figure()
    fig, axs = subplots(1, cols, figsize=(cols*HEIGHT, HEIGHT), squeeze=False)
    for k in range(len(max_depths)):
        d = max_depths[k]
        values = {}
        for f in max_features:
            yvalues = []
            for n in n_estimators:
                rf = RandomForestClassifier(n_estimators=n, max_depth=d, max_features=f)
                rf.fit(X_train, y_train)
                prdY = rf.predict(X_test)
                yvalues.append(accuracy_score(y_test, prdY))
                if yvalues[-1] > last_best:
                    best = (d, f, n)
                    last_best = yvalues[-1]
                    best_model = rf

            values[f] = yvalues
        multiple_line_chart(n_estimators, values, ax=axs[0, k], title=f'Random Forests with max_depth={d}',
                            xlabel='nr estimators', ylabel='accuracy', percentage=True)
        
    savefig(  os.path.join(get_plot_folder_path(), '%s_study'%save_file_name) )

    print('Best results with depth=%d, %1.2f features and %d estimators, with accuracy=%1.4f'%(best[0], best[1], best[2], last_best))

    evaluate_RF(df, save_file_name, labels, best_model, X_train, X_test, y_train, y_test)
    feature_relevance(df, save_file_name, labels, best_model)
    overfitting_study(df, save_file_name, X_train, X_test, y_train, y_test, n_estimators)

    if SHOW_PLOTS:
        show()


def evaluate_RF(df: DataFrame, save_file_name: str, labels, best_model, X_train, X_test, y_train, y_test):
    
    prd_trn = best_model.predict(X_train)
    prd_tst = best_model.predict(X_test)

    label_count = len(labels)

    if label_count == 2:
        plot_evaluation_results(labels, y_train, prd_trn, y_test, prd_tst)
    else: 
        plot_evaluation_results_multi_label(labels, y_train, prd_trn, y_test, prd_tst)

    savefig(  os.path.join(get_plot_folder_path(), '%s_best_eval' % save_file_name) )

    
def feature_relevance(df: DataFrame, save_file_name: str, labels, best_model):

    variables = df.columns
    importances = best_model.feature_importances_
    stdevs = std([tree.feature_importances_ for tree in best_model.estimators_], axis=0)
    indices = argsort(importances)[::-1]
    elems = []
    for f in range(len(variables)):
        elems += [variables[indices[f]]]
        print(f'{f+1}. feature {elems[f]} ({importances[indices[f]]})')

    figure()
    horizontal_bar_chart(elems, importances[indices], stdevs[indices], title='Random Forest Features importance', xlabel='importance', ylabel='variables')

    savefig(  os.path.join(get_plot_folder_path(), '%s_best_ranking' % save_file_name) )


def overfitting_study(df: DataFrame, save_file_name: str, X_train, X_test, y_train, y_test, n_estimators):

    f = 0.7
    max_depth = 25
    eval_metric = accuracy_score
    y_tst_values = []
    y_trn_values = []
    for n in n_estimators:
        rf = RandomForestClassifier(n_estimators=n, max_depth=max_depth, max_features=f)
        rf.fit(X_train, y_train)
        prd_tst_Y = rf.predict(X_test)
        prd_trn_Y = rf.predict(X_train)
        y_tst_values.append(eval_metric(y_test, prd_tst_Y))
        y_trn_values.append(eval_metric(y_train, prd_trn_Y))
    plot_overfitting_study(n_estimators, y_trn_values, y_tst_values, name=f'RF_depth={max_depth}_vars={f}', xlabel='nr_estimators', ylabel=str(eval_metric))
    savefig(  os.path.join(get_plot_folder_path(), '%s_best_overfitting' % save_file_name) )
   
