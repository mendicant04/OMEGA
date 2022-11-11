from my_models.basenet import *


def get_model(net_name, num_class=10, unit_size=2048, temp=0.05):
    model_g = ResBase(option=net_name, pret=True)
    model_c = ResClassifier_MME(num_classes=num_class, input_size=unit_size, temp=temp)
    return model_g, model_c
