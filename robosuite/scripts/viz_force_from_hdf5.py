
import argparse
import json
import os
import random

import h5py
import numpy as np

from robosuite.controllers import load_controller_config
from robosuite.utils.input_utils import *

from robomimic.envs.wrappers import ForceBinningWrapper

import plotly
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from PIL import Image
import robosuite_task_zoo

import mimicgen_envs
# def choose_mimicgen_environment():
#     """
#     Prints out environment options, and returns the selected env_name choice

#     Returns:
#         str: Chosen environment name
#     """

#     # try to import robosuite task zoo to include those envs in the robosuite registry
#     try:
#         import robosuite_task_zoo
#     except ImportError:
#         pass

#     # all base robosuite environments (and maybe robosuite task zoo)
#     robosuite_envs = set(suite.ALL_ENVIRONMENTS)

#     # all environments including mimicgen environments
#     import mimicgen_envs
#     all_envs = set(suite.ALL_ENVIRONMENTS)

#     # get only mimicgen envs
#     only_mimicgen_envs = sorted(all_envs - robosuite_envs)

#     # keep only envs that correspond to the different reset distributions from the paper
#     envs = [x for x in only_mimicgen_envs if x[-1].isnumeric()]

#     # Select environment to run
#     print("Here is a list of environments in the suite:\n")

#     for k, env in enumerate(envs):
#         print("[{}] {}".format(k, env))
#     print()
#     try:
#         s = input("Choose an environment to run " + "(enter a number from 0 to {}): ".format(len(envs) - 1))
#         # parse input into a number within range
#         k = min(max(int(s), 0), len(envs))
#     except:
#         k = 0
#         print("Input is not valid. Use {} by default.\n".format(envs[k]))

#     # Return the chosen environment name
#     return envs[k]


def force_binning(force):
    if force > 1:
        return 1
    if force < -1:
        return -1
    return 0

def torque_binning(torque):
    if torque > 0.1:
        return 1
    if torque < -0.1:
        return -1
    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--folder",
        type=str,
        help="Path to your demonstration folder that contains the demo.hdf5 file, e.g.: "
        "'path_to_assets_dir/demonstrations/YOUR_DEMONSTRATION'",
    ),
    parser.add_argument(
        "--use-actions",
        action="store_true",
    )
    args = parser.parse_args()


    PATH = "/home/rthompson/mimicgen_forcelearning/datasets/core/square_d0_force.hdf5"
    demo_path = args.folder
    hdf5_path = args.folder#os.path.join(demo_path, "demo.hdf5")
    f = h5py.File(hdf5_path, "r")

    #print(dir(f))
    #print(f["data"].attrs["env_args"])
    #env_name = f["data"].attrs["env_args"]["env_name"]
    #print(f["data"].attrs.keys())
    env_info = json.loads(f["data"].attrs["env_args"])
    env_info["env_name"] = env_info["env_name"]#.split("_")[0]

    del(env_info["env_version"])
    del(env_info["type"])

    env_info["env_kwargs"]["has_renderer"] = True
    env = suite.make(
        env_name=env_info["env_name"],
        **env_info["env_kwargs"],
        #single_object_mode=2,
        #nut_type="round",
        # has_renderer=True,
        # has_offscreen_renderer=True,
        # ignore_done=True,
        # use_camera_obs=True,
        # reward_shaping=True,
        # control_freq=20,
    )

    env = ForceBinningWrapper(env)

    force_idx = list(env._observables.keys()).index('robot0_ee_force')
    torque_idx = list(env._observables.keys()).index('robot0_ee_torque')

    # list of all demonstrations episodes
    demos = list(f["data"].keys())

    num_images = 10
    was_success = False
    ep = random.choice(demos)
    while True:
        if was_success:
            print("SUCCEEDED NEW")
            ep = random.choice(demos)
        else: 
            ep = ep

        fig = make_subplots(rows=3, cols=1, subplot_titles=("EE Force", "Demo", "EE Torque"))

        forces = []
        torques = []
        images = []
        print("Playing back random episode... (press ESC to quit)")

        # select an episode randomly
        
        ep_length = len(f["data/{}/states".format(ep)][()])
        image_freq = int(ep_length/num_images)

        # read the model xml, using the metadata stored in the attribute for this episode
        model_xml = f["data/{}".format(ep)].attrs["model_file"]
        print(f["data/{}".format(ep)].attrs.keys())

        env.reset()
        xml = env.edit_model_xml(model_xml)
        env.reset_from_xml_string(xml)
        env.sim.reset()

        #env.viewer.set_camera(0)

        # load the flattened mujoco states


        # states = f["data/{}/states".format(ep)][()]
        # obs = f["data/{}/obs".format(ep)]
        # forces = obs["robot0_ee_force"]
        # for state, force in zip(states, forces):
        #     print(state)
        #     print(force)
        #     input("next?")


        if args.use_actions:

            # load the initial state
            states = f["data/{}/states".format(ep)][()]
            env.sim.set_state_from_flattened(states[0])
            force_inits =[]
            for i in range(10):
                observation = env.step([0,0,0,0,0,0,1])
                force_inits.append(env._get_observations()['robot0_ee_force'])
            
            initial_force = np.mean(np.array(force_inits), axis=0)#env._get_observations()['robot0_ee_force']
            print(initial_force)
            #exit(0)
            #initial_tor = env._get_observations()['robot0_ee_torque'])


            forces.append(np.array([0,0,0]))
            torques.append(np.array([0,0,0]))

            # load the actions and play them back open-loop
            actions = np.array(f["data/{}/actions".format(ep)][()])
            num_actions = actions.shape[0]
            #print(actions[0])
            #exit(0)

            for j, action in enumerate(actions):
                observation = env.step(action)
                forces.append(observation[0]['robot0_ee_force'])
                torques.append(observation[0]['robot0_ee_torque'])
                if j % image_freq == 0:
                    images.append(observation[0]['agentview_image'])

                env.render()

                if j < num_actions - 1:
                    # ensure that the actions deterministically lead to the same recorded states
                    state_playback = env.sim.get_state().flatten()
                    if not np.all(np.equal(states[j + 1], state_playback)):
                        err = np.linalg.norm(states[j + 1] - state_playback)
                        #print(f"[warning] playback diverged by {err:.2f} for ep {ep} at step {j}")
            was_success = env._check_success()
            if isinstance(was_success, dict):
                was_success = was_success['task']

        else:
            # force the sequence of internal mujoco states one by one
            states = f["data/{}/states".format(ep)][()]
            actions = np.array(f["data/{}/actions".format(ep)][()])
            num_actions = actions.shape[0]
            j = 0
            for state, action in zip(states, actions):
                states = f["data/{}/states".format(ep)][()]
                env.sim.set_state_from_flattened(state)
                observation = env.step(action)

                # print(state)
                # print()
                # print(observation[0]['robot0_ee_force'])
                # input("continue?")

                forces.append(observation[0]['robot0_ee_force'])
                torques.append(observation[0]['robot0_ee_torque'])
                if j % image_freq == 0:
                    images.append(observation[0]['agentview_image'])

                # env.sim.set_state_from_flattened(state)
                # env.sim.forward()
                env.render()
                j+=1 


        # force  = states[:, 32:35]
        # torque = states[:, 35:38]

        v_func_force = np.vectorize(force_binning)
        v_func_torque = np.vectorize(torque_binning)

        print(np.array(forces) - initial_force)
        forces=np.array(forces)#v_func_force(np.array(forces) - initial_force)
        torques=np.array(torques)#v_func_torque(np.array(torques))
        
        fig.add_trace(go.Scatter(x = np.arange(len(forces)), y= forces[:,0], name='X force'), row=1, col=1)
        fig.add_trace(go.Scatter(x = np.arange(len(forces)), y= forces[:,1], name='Y force'), row=1, col=1)
        fig.add_trace(go.Scatter(x = np.arange(len(forces)), y= forces[:,2], name='Z force'), row=1, col=1)

        #print(dir(go))
        #figm = px.imshow(images, binary_string=True, facet_col=0, facet_col_wrap=10)
        #fig.add_trace(figm.data[0], 2, 1)
        #print(len(images))
        for i, image in enumerate(images):
            fig.add_trace( go.Image(z=np.flipud(image), x0=i*len(image),), row=2, col=1)

        fig.add_trace(go.Scatter(x = np.arange(len(torques)), y= torques[:,0], name='R torque'), row=3, col=1)
        fig.add_trace(go.Scatter(x = np.arange(len(torques)), y= torques[:,1], name='P torque'), row=3, col=1)
        fig.add_trace(go.Scatter(x = np.arange(len(torques)), y= torques[:,2], name='Y torque'), row=3, col=1)

        # other_obs = f["data/{}/obs".format(ep)]
        # #print(other_obs['robot0_ee_force'])
        # other_forces = other_obs['robot0_ee_force']#[other_obs[i]["robot0_ee_force"] for i in range(len(other_obs))]
        # other_torques = other_obs['robot0_ee_torque']#[other_obs[i]["robot0_ee_torque"] for i in range(len(other_obs))]
        # fig.add_trace(go.Scatter(x = np.arange(len(other_forces)), y= other_forces[:,0], name='other X force'), row=1, col=1)
        # fig.add_trace(go.Scatter(x = np.arange(len(other_forces)), y= other_forces[:,1], name='other Y force'), row=1, col=1)
        # fig.add_trace(go.Scatter(x = np.arange(len(other_forces)), y= other_forces[:,2], name='other Z force'), row=1, col=1)

        # fig.add_trace(go.Scatter(x = np.arange(len(other_torques)), y= other_torques[:,0], name='other R torque'), row=3, col=1)
        # fig.add_trace(go.Scatter(x = np.arange(len(other_torques)), y= other_torques[:,1], name='other P torque'), row=3, col=1)
        # fig.add_trace(go.Scatter(x = np.arange(len(other_torques)), y= other_torques[:,2], name='other Y torque'), row=3, col=1)


        fig.show()

    #actions = np.array(f["data/{}/actions".format(ep)][()])
    exit(0)

