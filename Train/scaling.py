import numpy as np

class scaled_into_trainset:
    def __init__(self, scaler):
        self.scaler = scaler
    def x_scaled(self, train, test, val = None):
        self.scaler.fit(train)
        trainscaled_x = self.scaler.transform(train)
        testscaled_x = self.scaler.transform(test)

        if val is None:
            return trainscaled_x, testscaled_x

        else:
            valscaled_x = self.scaler.transform(val)
            return trainscaled_x, testscaled_x, valscaled_x
        

    def y_scaled(self, train, test, val = None):
        train_y_ar = np.array(train).reshape(-1, 1)
        test_y_ar = np.array(test).reshape(-1, 1)
        self.scaler.fit(train_y_ar)
        trainscaled_y = self.scaler.transform(train_y_ar)
        test_y_ar = self.scaler.transform(test_y_ar)

        if val is None :
            return trainscaled_y, test_y_ar 
        
        else:
            val_y_ar = np.array(val).reshape(-1, 1)
            val_y_ar = self.scaler.transform(val_y_ar)

            return trainscaled_y, test_y_ar, val_y_ar               

    def scale_to_origin(self, pred, test):
        pred = pred.reshape(-1, 1)
        test = test.reshape(-1, 1)

        pred_origin = self.scaler.inverse_transform(pred).reshape(-1)
        return pred_origin, test
