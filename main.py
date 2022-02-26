from train_and_test import model_train, model_test
import conf

test_file = conf.test_file

model_train()
model_test(test_file)
