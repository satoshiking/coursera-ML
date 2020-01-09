import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, precision_recall_curve


data = pd.read_csv('classification.csv')
X = data['true']
y = data['pred']

TP = data[(data['true'] == 1) & (data['pred'] == 1)].count()[0]
TN = data[(data['true'] == 0) & (data['pred'] == 0)].count()[0]
FP = data[(data['true'] == 0) & (data['pred'] == 1)].count()[0]
FN = data[(data['true'] == 1) & (data['pred'] == 0)].count()[0]
print("TP, FP, FN, TN = %s %s %s %s" % (TP, FP, FN, TN))

accuracy_score = accuracy_score(X, y)
precision_score = precision_score(X, y)
recall_score = recall_score(X, y)
f1_score = f1_score(X, y)
print("accuracy, precision, recall, f1 = {:.3} {:.3} {:.3} {:.3}".
      format(accuracy_score, precision_score, recall_score, f1_score))

df = pd.read_csv('scores.csv')
roc_auc_score_logreg = roc_auc_score(df['true'], df['score_logreg'])
roc_auc_score_svm = roc_auc_score(df['true'], df['score_svm'])
roc_auc_score_knn = roc_auc_score(df['true'], df['score_knn'])
roc_auc_score_tree = roc_auc_score(df['true'], df['score_tree'])
print("roc_auc_score:")
print("logreg, svm, knn, tree = {:.3} {:.3} {:.3} {:.3}".
      format(roc_auc_score_logreg, roc_auc_score_svm, roc_auc_score_knn, roc_auc_score_tree))


logreg_precision, logreg_recall, logreg_thresholds = precision_recall_curve(df['true'], df['score_logreg'])
zipped_list_logreg = list(zip(logreg_precision, logreg_recall, logreg_thresholds))
zipped_list_logreg_filtered = sorted([x for x in zipped_list_logreg if x[1] >= 0.7], key=lambda y: y[0])
print("logreg: best precision = {:.3f}".format(zipped_list_logreg_filtered[-1][0]))

svm_precision, svm_recall, svm_thresholds = precision_recall_curve(df['true'], df['score_svm'])
zipped_list_svm = list(zip(svm_precision, svm_recall, svm_thresholds))
zipped_list_svm_filtered = sorted([x for x in zipped_list_svm if x[1] >= 0.7], key=lambda y: y[0])
print("svm:    best precision = {:.3f}".format(zipped_list_svm_filtered[-1][0]))

knn_precision, knn_recall, knn_thresholds = precision_recall_curve(df['true'], df['score_knn'])
zipped_list_knn = list(zip(knn_precision, knn_recall, knn_thresholds))
zipped_list_knn_filtered = sorted([x for x in zipped_list_knn if x[1] >= 0.7], key=lambda y: y[0])
print("knn:    best precision = {:.3f}".format(zipped_list_knn_filtered[-1][0]))

tree_precision, tree_recall, tree_thresholds = precision_recall_curve(df['true'], df['score_tree'])
zipped_list_tree = list(zip(tree_precision, tree_recall, tree_thresholds))
zipped_list_tree_filtered = sorted([x for x in zipped_list_tree if x[1] >= 0.7], key=lambda y: y[0])
print("tree:   best precision = {:.3f}".format(zipped_list_tree_filtered[-1][0]))

print("Best precision with recall >= 0.7 among all 4 classifiers is {:.3f}".
      format(max(zipped_list_logreg_filtered[-1][0],
                 zipped_list_svm_filtered[-1][0],
                 zipped_list_knn_filtered[-1][0],
                 zipped_list_tree_filtered[-1][0])))
