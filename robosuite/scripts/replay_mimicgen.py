import json

import h5py
import numpy as np
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.obs_utils as ObsUtils
import robosuite as suite
from robomimic.config import config_factory
import imageio

from mimicgen_envs.envs.robosuite.coffee import Coffee_D0

PATH = "/home/rthompson/mimicgen_forcelearning/datasets/core/square_d0_force.hdf5"
#SAVE_PATH = "/home/rthompson/mimicgen_forcelearning/datasets/core/square_d0.hdf5"
CONFIG_PATH = "/home/rthompson/mimicgen_forcelearning/mimicgen_envs/exps/paper/core/coffee_d0/image/bc_rnn.json"


def h5_tree(val, pre=''):
    items = len(val)
    for key, val in val.items():
        items -= 1
        if items == 0:
            # the last item
            if type(val) == h5py._hl.group.Group:
                print(pre + '└── ' + key)
                h5_tree(val, pre+'    ')
            else:
                try:
                    print(pre + '└── ' + key + ' (%d)' % len(val))
                except TypeError:
                    print(pre + '└── ' + key + ' (scalar)')
        else:
            if type(val) == h5py._hl.group.Group:
                print(pre + '├── ' + key)
                h5_tree(val, pre+'│   ')
            else:
                try:
                    print(pre + '├── ' + key + ' (%d)' % len(val))
                except TypeError:
                    print(pre + '├── ' + key + ' (scalar)')


def main():

    ext_cfg = json.load(open(CONFIG_PATH, 'r'))
    config = config_factory(ext_cfg["algo_name"])
    with config.values_unlocked():
        config.update(ext_cfg)
    ObsUtils.initialize_obs_utils_with_config(config)

    metadata = FileUtils.get_env_metadata_from_dataset(PATH)

    env = EnvUtils.create_env_from_metadata(
        env_meta=metadata,
        env_name=metadata["env_name"], 
        render=True, 
        render_offscreen=True,
        use_image_obs=False, 
    )
    env = EnvUtils.wrap_env_from_config(env, config=config)

    # env = suite.make(
    #     env_name="Coffee_D0", # try with other tasks like "Stack" and "Door"
    #     robots="Panda",  # try with other robots like "Sawyer" and "Jaco"
    #     has_renderer=True,
    #     has_offscreen_renderer=False,
    #     use_camera_obs=False,
    # )

    with h5py.File(PATH, "r") as f:

        #duplicate the thing
        # new_f = cp.deepcopy(f)

        episodes = f["data"]

        for key in episodes:

            video_writer = imageio.get_writer(f"{key}.mp4", fps=20)

            state = episodes[key]["states"][0][:]

            print(episodes[key]["obs"].keys())
            env.reset()
            env.env.sim.set_state_from_flattened(state)
            print("REDAY 1")
            exit(0)
            #env.render()

            print("READY TO RENDER")
            video_img = env.render(mode="rgb_array", height=512, width=512)
            video_writer.append_data(video_img)
            print("RENDERING")
            i = 0

            for action in episodes[key]["actions"][:]:
                env.step(action)
                #env.render()
                #print(i)

                video_img = env.render(mode="rgb_array", height=512, width=512)
                video_writer.append_data(video_img)
                i += 1

            video_writer.close()
        exit(0)

if __name__ == "__main__":
    main()