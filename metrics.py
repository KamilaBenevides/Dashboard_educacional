from sklearn.metrics import mean_squared_error, max_error, mean_absolute_percentage_error, r2_score, mean_absolute_error

def error_metrics(y_true, y_pred):
    rmse = mean_squared_error(y_true, y_pred)
    me = max_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    
    print('RMSE: ', rmse)
    print('Max error: ', me)
    print('MAPE: ', mape)
    print('R2: ', r2)
    print('MAE: ', mae)
