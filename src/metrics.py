from sklearn.metrics import roc_curve, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef


# for logistic regression: auc = roc_auc_score(y_test, model.predict_proba(X_test)[:,1])
def roc_auc(y_true, y_scores):
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    auc = roc_auc_score(y_true, y_scores)
    return fpr, tpr, thresholds, auc


def calculate_metrics(y_true, y_scores, threshold=0.5):
    # 根据设定的阈值将概率转为预测标签
    y_pred = (y_scores >= threshold).astype(int)

    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)

    return precision, recall, f1, mcc
