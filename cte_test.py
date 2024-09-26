
import torch
import gym
import gym_donkeycar
import time
import numpy as np
from newtrain import SimpleCNN
def select_action(env):
    return env.action_space.sample()  # taking random action from the action_space

import cv2
from noise import *
import os
from tqdm import tqdm


def function_switch(img, fun_name: str):
    if "salt" in fun_name:
        return add_salt_pepper_noise(img)
    elif "gaussian" in fun_name:
        return add_gaussian_noise(img)
    elif "shot" in fun_name:
        scale = 100.0
        return add_shot_noise(img, scale=scale)
    elif "rain" in fun_name:
        rain_type = "torrential"
        return add_rain(img, rain_type=rain_type)
    elif "snow" in fun_name:
        # 增加亮度係數
        brightness_coefficient = 2.5
        # 雪點閾值
        snow_point = 200  # 增加此值可增加雪點數量
        return add_snow(img, snow_point, brightness_coefficient)
    elif "more_bri" in fun_name:
        # between 0.5 and 1.5
        brightness_coefficient = 1.5
        return change_brightness(img, brightness_coefficient)
    elif "less_bri" in fun_name:
        brightness_coefficient = 0.1
        return change_brightness(img, brightness_coefficient)
    elif "fog" in fun_name:
        fog_coeff = 0.9
        return add_fog(img, fog_coeff)
    elif "more_con" in fun_name:
        alpha = 1.8  # Simple contrast control
        beta = 5  # Simple brightness control
        return adjust_contrast(img, alpha, beta)  # 調整對比度
    elif "defocus" in fun_name:
        kernel_size = 31  # 虛焦的核大小，越大虛焦效果越明顯
        return add_defocus(img, kernel_size)  # 添加虛焦效果
    elif "glass" in fun_name:
        return add_glass_blur(img, kernel_size=17)
    elif "vert_motion" in fun_name:
        kernel_size = 20  # 運動模糊核大小，越大模糊效果越明顯
        style = "vert"
        return add_motion_blur(img, kernel_size, style)
    elif "hori_motion" in fun_name:
        kernel_size = 20  # 運動模糊核大小，越大模糊效果越明顯
        style = "hori"
        return add_motion_blur(img, kernel_size, style)
    else:
        _msg = f"Function '{fun_name}' not found!"
        raise Exception(_msg)
if __name__ == '__main__':
    env = gym.make("donkey-generated-roads-v0")
    obs = env.reset()
    env.pause_env()  # freeze
    ctelist = []
    device='cuda'
    model = SimpleCNN().to(device)
    model.load_state_dict(torch.load("model_epoch_10.pth"))
    noise = 'less_bri'
    for i in range(500):
        "MAE: 0.02  MAE sam trained is 0.1"

        env.pause_env()  # unfreeze
        img = obs
        img = function_switch(img, noise)

        tensor = torch.from_numpy(img)

        # 调整维度，从 (120, 160, 3) 转换为 (3, 120, 160)
        tensor = tensor.permute(2, 0, 1)
        tensor = tensor.float()
        tensor = tensor/255
        tensor = tensor.to(device)
        action = model(tensor.unsqueeze(0))

        result = action.squeeze().tolist()

        obs, reward, done, info = env.step([result[1],result[0]])
        ctelist.append(np.abs(info['cte']+2))
        print(info['cte'])
        time.sleep(0.0167)

        env.pause_env()  # Freeze
        print(ctelist)
    print('mean')
    print(np.mean(ctelist))

#new = [4.213637999773229,4.171676910021984,2.636423412471515,1.3520262727552326,2.1887333605013555]
# rain = [25.793220801296616,58.275497797029864, ,74.01542967879784]
# snow = [130.86793826855205,139.54222226964717,101.38986001815456]
# night = [5.71902729856188,5.562620322908755,5.550844528526708,NaN]
#old[-0.41262078345508735,-2.2813784253009652,-3.5821639283100377,]
