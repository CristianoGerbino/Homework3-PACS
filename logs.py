import pickle

STANDARD_PREFIX = 'model_std_state_dict'
DANN_PREFIX = 'model_dann_state_dict'
LOG_PREFIX = 'log_stage'
TEST_LOG_PREFIX = 'test_log_stage'
DOM_ADAPT_PREFIX = 'dann_log_stage'

class Logger():
    def __init__(self, **params):
        self.params = params
        self.data = []
        self.step_data = []

    def add_epoch_data(self, epoch, acc, loss, dual_loss=None):
        if dual_loss is not None:
            self.data.append({epoch:(acc, loss, dual_loss)})
        else:
            self.data.append({epoch:(acc, loss)})
        
    def add_step_data(self, step, acc, loss):
        self.step_data.append({step:(acc, loss)})
    
    def save(self, path):
        with open(path, 'wb') as logfile:
            pickle.dump(self, logfile)
    
    @classmethod
    def load(cls, path):
        with open(path, 'rb') as logfile:
            new_instance = pickle.load(logfile)
        return new_instance

    
def generate_model_checkpoint_name(DANN = False, loss = None, optional=''):
    name = ""
    if not DANN: 
        name += STANDARD_PREFIX
    else:
        name += DANN_PREFIX

    if loss is not None:
            name += '_'+loss

    name += optional+".pth"
    
    return name


def generate_log_filenames(DANN = False, loss = None, optional=''):
    train = LOG_PREFIX 
    test = TEST_LOG_PREFIX
    dann = DOM_ADAPT_PREFIX 

    if DANN:
      train += "with_DANN_" + optional+".obj"
      test  += "with_DANN_" + optional+".obj"
      dann += "with_DANN_" + optional+".obj"
      return train, test, dann
    
    if loss is not None:
        train += '_'+loss
        test += '_'+loss
    train += optional+".obj"
    test  += optional+".obj"
    
    return train, test