from tools.printer import Printer, bcolors


class Config_base:
    def __init__(self, raw_cfg, name):
        Printer.print_OK("* Reading cfg for {}...".format(name))
        Printer.add_tab()
        self.raw_cfg = raw_cfg
        self.class_name = name

    # designed to load rest of the item in raw_cfg into the object, typically only used
    # in the layer of complexit
    def load_rest(self):
        for k, v in self.raw_cfg.items():
            self.__dict__[k] = v

    # should be called in the child class right before exiting __init__()
    def clean_up(self, load_rest=True):
        self.load_rest() if load_rest else None
        # removing raw_cfg, since it is dead weight
        name = self.class_name
        del self.class_name
        del self.raw_cfg
        Printer.remove_tab()
        Printer.print_OK("* Reading cfg for {} is done".format(name))

    # designed to read a specific mandatory component
    # after reading the key/value pair in raw_cfg should be deleted since it no longer serves any purpose
    def read_mandatory(self, key):
        try:
            extracted = self.raw_cfg[key]
            del self.raw_cfg[key]
            return extracted

        except KeyError as error:
            Printer.print_FAIL(
                "Unable to find basic elements {} of {}. Are you sure the config.yaml is defined corretly?".format(
                    key, self.class_name
                )
            )
            raise (error)
            
    def read_non_mandatory(self, key):
        if key in self.raw_cfg:
            return self.read_mandatory(key)
        # basically no error thrown if the key cannot be found
        else:
            Printer.print_WARNING("Unable to find non-mandatory element {} of {}. skipped".format(key, self.class_name))
            return None
    # equivilent to ., but returns default when not found
    def get(self, key, default=None):
        if key in self.__dict__:
            return self.__dict__[key]
        return default
    
    def has(self, key):
        return key in self.__dict__

    def put(self, key, value):
        self.__dict__[key] = value

    # set requires key to exists first, then it will put it in, while put() just put the new key directly
    def set(self, key, value):
        if key in self.__dict__:
            self.put(key, value)
        else:
            Printer.print_WARNING(
                "Unable to call set on {} for key = {}, since key does not exists".format(self.class_name, key)
            )

    def __str__(self):
        s = ""
        for k, v in self.__dict__.items():
            # designed to format config_base into the correct format like a tree recursively
            if isinstance(v, Config_base):
                s_child = str(v).split("\n")[:-1]
                s += bcolors.OKGREEN + bcolors.UNDERLINE + str(k) + bcolors.ENDC + ": "
                for s_c in s_child:
                    s += "\n\t" + s_c
                s += "\n"
            else:
                s += bcolors.OKBLUE + str(k) + bcolors.ENDC + ": " + str(v) + "\n"

        return s
