import yaml

# this is needed to ensure correct import order for some simulators
import engines.engine_builder as engine_builder

from util.logger import Logger

def build_env(env_file, engine_file, num_envs, device, visualize, enable_cameras=False):
    env_config, engine_config = load_configs(env_file, engine_file)

    env_name = env_config["env_name"]
    Logger.print("Building {} env".format(env_name))
    
    if (env_name == "char"):
        import envs.char_env as char_env
        EnvClass = char_env.CharEnv
    elif (env_name == "deepmimic"):
        import envs.deepmimic_env as deepmimic_env
        EnvClass = deepmimic_env.DeepMimicEnv
    elif (env_name == "amp"):
        import envs.amp_env as amp_env
        EnvClass = amp_env.AMPEnv
    elif (env_name == "ase"):
        import envs.ase_env as ase_env
        EnvClass = ase_env.ASEEnv
    elif (env_name == "add"):
        import envs.add_env as add_env
        EnvClass = add_env.ADDEnv
    elif (env_name == "char_dof_test"):
        import envs.char_dof_test_env as char_dof_test_env
        EnvClass = char_dof_test_env.CharDofTestEnv
    elif (env_name == "view_motion"):
        import envs.view_motion_env as view_motion_env
        EnvClass = view_motion_env.ViewMotionEnv
    elif (env_name == "task_location"):
        import envs.task_location_env as task_location_env
        EnvClass = task_location_env.TaskLocationEnv
    elif (env_name == "task_steering"):
        import envs.task_steering_env as task_steering_env
        EnvClass = task_steering_env.TaskSteeringEnv
    elif (env_name == "static_objects"):
        import envs.static_objects_env as static_objects_env
        EnvClass = static_objects_env.StaticObjectsEnv
    else:
        assert(False), "Unsupported env: {}".format(env_name)

    env = EnvClass(env_config, engine_config, num_envs, device, visualize, enable_cameras)

    return env

def load_config(file):
    if (file is not None and file != ""):
        with open(file, "r") as stream:
            config = yaml.safe_load(stream)
    else:
        config = None
    return config

def load_configs(env_file, engine_file):
    env_config = load_config(env_file)
    engine_config = load_config(engine_file)

    if ("engine" in env_config):
        env_engine_config = env_config["engine"]
        engine_config = override_engine_config(env_engine_config, engine_config)

    return env_config, engine_config

def override_engine_config(env_engine_config, engine_config):
    Logger.print("Overriding Engine configs with parameters from the Environment:")
    
    if (engine_config is None):
        engine_config = env_engine_config
    else:
        engine_config = engine_config.copy()
        for key, val in env_engine_config.items():
            engine_config[key] = val
            Logger.print("\t{}: {}".format(key, val))

    Logger.print("")
    return engine_config