import numpy as np
import pprint


def softmax(model, note):
   preds = model.lin_reg.predict([model.X_train[note]])
   preds_exp = np.exp(preds)
   preds_exp = preds_exp / np.sum(preds_exp)
   print("SOFTMAX RESULTS:")
   print(preds_exp)
   print("SUM_SOFTMAX: ", np.sum(preds_exp), end="\n")
   preds_pow = np.power(preds_exp, 3)
   normed_preds = [pred/sum(preds_pow[0]) for pred in preds_pow[0]]
   print("NORMALIZATION RESULTS:")
   pprint.pprint(normed_preds)
   print("SUM_NORM: ", sum(normed_preds))