import xgboost as xgb

xgb_cl = xgb.XGBClassifier()

def train_test_split(x, y, test_size):
    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size)
    return x_train, x_test, y_train, y_test

def shap_values(x_train, y_train, x_test, y_test):
    from shap import TreeExplainer
    explainer = TreeExplainer(xgb_cl)
    shap_values = explainer.shap_values(x_test)
    print(shap_values)



def xgboost_train(x_train, y_train, x_test, y_test):
    xgb_cl.fit(x_train, y_train)
    y_pred = xgb_cl.predict(x_test)
    print(y_pred)
    print(y_test)
    print(xgb_cl.score(x_test, y_test))

def xgboost_train_cv(x_train, y_train, x_test, y_test):
    from sklearn.model_selection import cross_val_score
    scores = cross_val_score(xgb_cl, x_train, y_train, cv=5)
    print(scores)

def xgboost_explain(x_train, y_train, x_test, y_test):
    from shap import TreeExplainer
    explainer = TreeExplainer(xgb_cl)
    shap_values = explainer.shap_values(x_test)
    print(shap_values)


def xgboost_graph(x_train, y_train, x_test, y_test):
    from shap import TreeExplainer
    explainer = TreeExplainer(xgb_cl)
    shap_values = explainer.shap_values(x_test)
    shap.summary_plot(shap_values, x_test)

def xgboost_gridsearch(x_train, y_train, x_test, y_test):
    from sklearn.model_selection import GridSearchCV
    param_grid = {
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.5],
        'n_estimators': [100, 200, 300]
    }
    gs = GridSearchCV(xgb_cl, param_grid, cv=5)
    gs.fit(x_train, y_train)
    print(gs.best_params_)
    print(gs.best_score_)

