
import pybullet as p

def getAddition(keys, scale):

    yaw_add = 0
    pitch_add = 0
    x_add = 0
    y_add = 0
    z_add = 0

    botton_add_dict = {"d" : yaw_add,
                        "r" : pitch_add,  
                        "k" : x_add,
                        "y" : y_add,
                        "u" : z_add}
    botton_minus_dict = {"f" : yaw_add,
                            "e" : pitch_add,
                            "h" : x_add,
                            "l" : y_add,
                            "j" : z_add}
    
    for botton in botton_add_dict.keys():
        if ord(botton) in keys and keys[ord(botton)] & p.KEY_WAS_TRIGGERED:
            botton_add_dict[botton] += 1
            # print("{} KEY_WAS_TRIGGERED".format(botton))
        elif ord(botton) in keys and keys[ord(botton)] & p.KEY_IS_DOWN:
            botton_add_dict[botton] += 1
            # print("{} KEY_IS_DOWN".format(botton))
        elif ord(botton) in keys and keys[ord(botton)] & p.KEY_WAS_RELEASED:
            botton_add_dict[botton] += 1
            # print("{} KEY_WAS_RELEASED".format(botton))

    for botton in botton_minus_dict.keys():
        if ord(botton) in keys and keys[ord(botton)] & p.KEY_WAS_TRIGGERED:
            botton_minus_dict[botton] -= 1
            # print("{} KEY_WAS_TRIGGERED".format(botton))
        elif ord(botton) in keys and keys[ord(botton)] & p.KEY_IS_DOWN:
            botton_minus_dict[botton] -= 1
            # print("{} KEY_IS_DOWN".format(botton))
        elif ord(botton) in keys and keys[ord(botton)] & p.KEY_WAS_RELEASED:
            botton_minus_dict[botton] -= 1
            # print("{} KEY_WAS_RELEASED".format(botton))

    yaw_add = botton_add_dict["d"] + botton_minus_dict["f"]
    pitch_add = botton_add_dict["r"] + botton_minus_dict["e"]
    x_add = (botton_add_dict["k"] + botton_minus_dict["h"]) * scale
    y_add = (botton_add_dict["y"] + botton_minus_dict["l"]) * scale
    z_add = (botton_add_dict["u"] + botton_minus_dict["j"]) * scale
    
    return yaw_add, pitch_add, x_add, y_add, z_add


def getAdditionPlain(keys, scale):

    yaw_add = 0
    pitch_add = 0
    x_add = 0
    y_add = 0
    z_add = 0

    if ord("u") in keys and keys[ord("u")] & p.KEY_WAS_TRIGGERED:
        z_add -= 1
        print("u KEY_WAS_TRIGGERED")
    elif ord("u") in keys and keys[ord("u")] & p.KEY_IS_DOWN:
        z_add -= 1
        print("u KEY_IS_DOWN")
    elif ord("u") in keys and keys[ord("u")] & p.KEY_WAS_RELEASED:
        z_add -= 1
        print("u KEY_WAS_RELEASED")
    if ord("o") in keys and keys[ord("o")] & p.KEY_WAS_TRIGGERED:
        z_add += 1
        print("o KEY_WAS_TRIGGERED")
    elif ord("o") in keys and keys[ord("o")] & p.KEY_IS_DOWN:
        z_add += 1
        print("o KEY_IS_DOWN")
    elif ord("o") in keys and keys[ord("o")] & p.KEY_WAS_RELEASED:
        z_add += 1
        print("o KEY_WAS_RELEASED")

    if ord("j") in keys and keys[ord("j")] & p.KEY_WAS_TRIGGERED:
        x_add -= 1
        print("j KEY_WAS_TRIGGERED")
    elif ord("j") in keys and keys[ord("j")] & p.KEY_IS_DOWN:
        x_add -= 1
        print("j KEY_IS_DOWN")
    elif ord("j") in keys and keys[ord("j")] & p.KEY_WAS_RELEASED:
        x_add -= 1
        print("j KEY_WAS_RELEASED")
    if ord("l") in keys and keys[ord("l")] & p.KEY_WAS_TRIGGERED:
        x_add += 1
        print("l KEY_WAS_TRIGGERED")
    elif ord("l") in keys and keys[ord("l")] & p.KEY_IS_DOWN:
        x_add += 1
        print("l KEY_IS_DOWN")
    elif ord("l") in keys and keys[ord("l")] & p.KEY_WAS_RELEASED:
        x_add += 1
        print("l KEY_WAS_RELEASED") 

    if ord("i") in keys and keys[ord("i")] & p.KEY_WAS_TRIGGERED:
        y_add += 1
        print("i KEY_WAS_TRIGGERED")
    elif ord("i") in keys and keys[ord("i")] & p.KEY_IS_DOWN:
        y_add += 1
        print("i KEY_IS_DOWN")
    elif ord("i") in keys and keys[ord("i")] & p.KEY_WAS_RELEASED:
        y_add += 1
        print("i KEY_WAS_RELEASED")
    if ord("k") in keys and keys[ord("k")] & p.KEY_WAS_TRIGGERED:
        y_add -= 1
        print("k KEY_WAS_TRIGGERED")
    elif ord("k") in keys and keys[ord("k")] & p.KEY_IS_DOWN:
        y_add -= 1
        print("k KEY_IS_DOWN")
    elif ord("k") in keys and keys[ord("k")] & p.KEY_WAS_RELEASED:
        y_add -= 1
        print("k KEY_WAS_RELEASED")

    if ord("f") in keys and keys[ord("f")] & p.KEY_WAS_TRIGGERED:
        yaw_add -= 1
        print("f KEY_WAS_TRIGGERED")
    elif ord("f") in keys and keys[ord("f")] & p.KEY_IS_DOWN:
        yaw_add -= 1
        print("f KEY_IS_DOWN")
    elif ord("f") in keys and keys[ord("f")] & p.KEY_WAS_RELEASED:
        yaw_add -= 1
        print("f KEY_WAS_RELEASED")
    if ord("d") in keys and keys[ord("d")] & p.KEY_WAS_TRIGGERED:
        yaw_add += 1
        print("d KEY_WAS_TRIGGERED")
    elif ord("d") in keys and keys[ord("d")] & p.KEY_IS_DOWN:
        yaw_add += 1
        print("d KEY_IS_DOWN")
    elif ord("d") in keys and keys[ord("d")] & p.KEY_WAS_RELEASED:
        yaw_add += 1
        print("d KEY_WAS_RELEASED")

    if ord("r") in keys and keys[ord("r")] & p.KEY_WAS_TRIGGERED:
        pitch_add += 1
        print("r KEY_WAS_TRIGGERED")
    elif ord("r") in keys and keys[ord("r")] & p.KEY_IS_DOWN:
        pitch_add += 1
        print("r KEY_IS_DOWN")
    elif ord("r") in keys and keys[ord("r")] & p.KEY_WAS_RELEASED:
        pitch_add += 1
        print("r KEY_WAS_RELEASED")
    if ord("e") in keys and keys[ord("e")] & p.KEY_WAS_TRIGGERED:
        pitch_add -= 1
        print("e KEY_WAS_TRIGGERED")
    elif ord("e") in keys and keys[ord("e")] & p.KEY_IS_DOWN:
        pitch_add -= 1
        print("e KEY_IS_DOWN")
    elif ord("e") in keys and keys[ord("e")] & p.KEY_WAS_RELEASED:
        pitch_add -= 1
        print("e KEY_WAS_RELEASED")
    
    yaw_add = yaw_add
    pitch_add = pitch_add
    x_add = x_add * scale
    y_add = y_add * scale
    z_add = z_add * scale
    
    return yaw_add, pitch_add, x_add, y_add, z_add


def getDirection(keys):

    botton_direction = {"u" : [1, 0, 0, 0, 0],
                        "h" : [0, 1, 0, 0, 0],  
                        "j" : [0, 0, 1, 0, 0],
                        "k" : [0, 0, 0, 1, 0]}
    
    for botton in botton_direction.keys():
        if ord(botton) in keys and keys[ord(botton)] & p.KEY_WAS_TRIGGERED:
            print("{} KEY_WAS_TRIGGERED".format(botton))
            return botton_direction[botton]
        elif ord(botton) in keys and keys[ord(botton)] & p.KEY_IS_DOWN:
            print("{} KEY_IS_DOWN".format(botton))
            return botton_direction[botton]
        elif ord(botton) in keys and keys[ord(botton)] & p.KEY_WAS_RELEASED:
            print("{} KEY_WAS_RELEASED".format(botton))
            return botton_direction[botton]
    
    return [0, 0, 0, 0, 1]