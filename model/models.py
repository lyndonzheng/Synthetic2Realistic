
def create_model(opt):
    print(opt.model)
    if opt.model == 'wsupervised':
        from .T2model import T2NetModel
        model = T2NetModel()
    elif opt.model == 'supervised':
        from .TaskModel import TNetModel
        model = TNetModel()
    elif opt.model == 'test':
        from .test_model import TestModel
        model = TestModel()
    else:
        raise ValueError("Model [%s] not recognized." % opt.model)
    model.initialize(opt)
    print("model [%s] was created." % (model.name()))
    return model