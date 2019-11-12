import json

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

class Params():
    """Class that loads hyperparameters from a json file.

    Example:
        params = Params(json_path)
        print(params.learning_rate)
        params.learning_rate = 0.5  # change the value of learning_rate in params
    """

    def __init__(self, json_path):
        self.update(json_path)

    def save(self, json_path):
        """Saves parameters to json file"""
        with open(json_path, 'w') as f:
            json.dump(self.JSON, f, indent=4)

    def update(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            JSON = json.load(f)
            self.JSON = dotdict(JSON)

            #self.__dict__.update(self.params)

    def __repr__(self):
        return self.JSON

    def __str__(self) :
        return json.dumps(self.JSON)

    @property
    def dict(self) :
        """Gives dict-like access to Params instance by `params.dict['learning_rate']`"""
        return self.__dict__