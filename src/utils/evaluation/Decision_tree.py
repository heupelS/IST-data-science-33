import os, sys

from numpy import ndarray
from numpy import argsort, arange
from pandas import DataFrame, read_csv, unique

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, subplots, savefig, show
from matplotlib.pyplot import Axes

from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

sys.path.append( os.path.join(os.path.dirname(__file__),'..') )
from ds_charts import plot_evaluation_results, multiple_line_chart
from ds_charts import horizontal_bar_chart
from ds_charts import plot_overfitting_study
from ds_charts_extensions import plot_evaluation_results_multi_label
from general_utils import get_plot_folder_path


###########################################
## Select if plots show up of just saved ##
SHOW_PLOTS = False
###########################################

def DT(df: DataFrame, target_name: str, save_file_name: str):

    print('-- Calculate DT')

    y: ndarray = df.pop(target_name).values
    X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2, random_state=0)

    labels = unique(y_train)
    labels.sort()

    min_impurity_decrease = [0.01, 0.005, 0.0025, 0.001, 0.0005]
    max_depths = [2, 5, 10, 15, 20, 25]
    criteria = ['entropy', 'gini']
    best = ('',  0, 0.0)
    last_best = 0
    best_model = None

    figure()
    fig, axs = subplots(1, 2, figsize=(16, 4), squeeze=False)
    for k in range(len(criteria)):
        f = criteria[k]
        values = {}
        for d in max_depths:
            yvalues = []
            for imp in min_impurity_decrease:
                tree = DecisionTreeClassifier(max_depth=d, criterion=f, min_impurity_decrease=imp)
                tree.fit(X_train, y_train)
                prdY = tree.predict(X_test)
                yvalues.append(accuracy_score(y_test, prdY))
                if yvalues[-1] > last_best:
                    best = (f, d, imp)
                    last_best = yvalues[-1]
                    best_model = tree

            values[d] = yvalues

        multiple_line_chart(min_impurity_decrease, values, ax=axs[0, k], title=f'Decision Trees with {f} criteria',
                            xlabel='min_impurity_decrease', ylabel='accuracy', percentage=True)
        
    savefig(  os.path.join(get_plot_folder_path(), '%s_study'%save_file_name) )

    print('Best results achieved with %s criteria, depth=%d and min_impurity_decrease=%1.4f ==> accuracy=%1.2f'%(best[0], best[1], best[2], last_best))

    plot_tree(df, save_file_name, labels, best_model)
    evaluate_DT(df, save_file_name, labels, best_model, X_train, X_test, y_train, y_test)
    feature_relevance(df, save_file_name, labels, best_model)
    overfitting_study_DT(df, save_file_name, X_train, X_test, y_train, y_test, max_depths)

    if SHOW_PLOTS:
        show()

def plot_tree(df: DataFrame, save_file_name: str, labels, best_model):

    labels = [str(value) for value in labels]
    tree.plot_tree(best_model, feature_names=df.columns, class_names=labels)
    savefig(  os.path.join(get_plot_folder_path(), '%s_best_tree'%save_file_name) )

def evaluate_DT(df: DataFrame, save_file_name: str, labels, best_model, X_train, X_test, y_train, y_test):
    
    prd_trn = best_model.predict(X_train)
    prd_tst = best_model.predict(X_test)
    plot_evaluation_results(labels, y_train, prd_trn, y_test, prd_tst)

    label_count = len(labels)

    if label_count == 2:
        plot_evaluation_results(labels, y_train, prd_trn, y_test, prd_tst)
    else: 
        plot_evaluation_results_multi_label(labels, y_train, prd_trn, y_test, prd_tst)

    savefig(  os.path.join(get_plot_folder_path(), '%s_best_eval' % save_file_name) )

    
def feature_relevance(df: DataFrame, save_file_name: str, labels, best_model):

    variables = df.columns
    importances = best_model.feature_importances_
    indices = argsort(importances)[::-1]
    elems = []
    imp_values = []
    for f in range(len(variables)):
        elems += [variables[indices[f]]]
        imp_values += [importances[indices[f]]]
        print(f'{f+1}. feature {elems[f]} ({importances[indices[f]]})')

    figure()
    horizontal_bar_chart(elems, imp_values, error=None, title='Decision Tree Features importance', xlabel='importance', ylabel='variables')

    savefig(  os.path.join(get_plot_folder_path(), '%s_best_ranking' % save_file_name) )


def overfitting_study_DT(df: DataFrame, save_file_name: str, X_train, X_test, y_train, y_test, max_depths):

    imp = 0.0005
    f = 'entropy'
    eval_metric = accuracy_score
    y_tst_values = []
    y_trn_values = []
    for d in max_depths:
        tree = DecisionTreeClassifier(max_depth=d, criterion=f, min_impurity_decrease=imp)
        tree.fit(X_train, y_train)
        prdY = tree.predict(X_test)
        prd_tst_Y = tree.predict(X_test)
        prd_trn_Y = tree.predict(X_train)
        y_tst_values.append(eval_metric(y_test, prd_tst_Y))
        y_trn_values.append(eval_metric(y_train, prd_trn_Y))
    plot_overfitting_study(max_depths, y_trn_values, y_tst_values, name=f'%s=imp{imp}_{f}'%save_file_name, xlabel='max_depth', ylabel=str(eval_metric))
    savefig(  os.path.join(get_plot_folder_path(), '%s_best_overfitting' % save_file_name) )
