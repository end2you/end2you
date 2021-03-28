import json


class Params:
    """ Helper class for the hyper-parameters.
        It can either take a dictionary or read from json file.
    """
    
    def __init__(self, 
                 dict_params:dict=None, 
                 json_path:str=None):
        
        if json_path:
            with open(json_path) as f:
                params = json.load(f)
                self.__dict__.update(params)
        else:
            self.__dict__.update(dict_params)
    
    def save_to_json(self, json_path:str):
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)
    
    def update(self, json_path:str):
        """Load parameters from json file and update dictionary."""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)
    
    def __getitem__(self, key:str):
        return self.__dict__[key]
    
    def __str__(self):
        return str(self.dict)

    @property
    def dict(self):
        return self.__dict__
