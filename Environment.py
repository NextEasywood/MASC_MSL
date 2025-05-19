import numpy as np
import pandas as pd
import gym
from Gurobi_up import ex_gurobi


class DataManager:
    def __init__(self):
        self.PV_Generation = []
        self.Electricity_Consumption = []
        self.WT_Generation = []
        self.HOT_Consumption = []

    def add_day_data(self, pv_data, electricity_data, wt_data, hot_data):
        self.PV_Generation.append(pv_data)
        self.Electricity_Consumption.append(electricity_data)
        self.WT_Generation.append(wt_data)
        self.HOT_Consumption.append(hot_data)

    def get_data(self, day):
        return self.PV_Generation[day], self.Electricity_Consumption[day], self.WT_Generation[day], self.HOT_Consumption[day]



class Node1:
    def __init__(self, node_id, pv_generation, wt_generation, load):
        self.node_id = node_id  # 节点ID

        self.pv_generation = pv_generation   # 光伏发电数据引用，类型为列表
        self.wt_generation = wt_generation
        self.load = load  # 电力负载数据引用，类型为列表
        self.modulus = [2.5,4.5,0.2]

        self.energy_change = 0  # 能量变化量初始化为0
        self.capacity = 500
        self.max_soc = 400
        self.min_soc = 100
        self.factor = 0.95  # 电池的退化系数，用于计算能量变化对成本的影响
        self.max_charge = 100  # 最大充电功率，单位为kW
        self.current_capacity = 100  # 初始化电池状态

        self.current_electrolyzer_workspeed = 0  # 初始化电解水状态
        self.electrolyzer_climb = 50 # 电解水的爬坡
        self.electrolyzer_worklimit = 100
        self.electrolyzer_factor = 0.8

        self.current_hydrogen = 150  # 储氢罐起始
        self.hydrogen_capacity = 500  # 储氢罐的总容量，单位为kWh
        self.max_hydrogen = 400  # 氢气上限
        self.min_hydrogen = 150  # 氢气下限

        self.current_fuel_cell_workspeed = 0  # 初始化燃料电池状态
        self.fuel_cell_climb = 50  # 燃料电池的爬坡
        self.fuel_cell_worklimit = 100
        self.fuel_cell_factor = 0.6

        self.buy_price = [0.3578, 0.3578, 0.3578, 0.3578, 0.3578, 0.3578,
                     0.3578, 0.3578, 0.8325, 0.8325, 0.8325, 0.3578,
                     0.3578, 0.8325, 0.8325, 0.8325, 0.8325, 0.8325,
                     0.8325, 1.2109, 1.2109, 0.8325, 0.3578, 0.3578, 0.3578]
        self.sell_price = [0.2, 0.2, 0.2, 0.2, 0.2, 0.2,
                  0.2, 0.2, 0.4125, 0.4125, 0.4125, 0.2,
                  0.2, 0.4125, 0.4125, 0.4125, 0.4125, 0.4125,
                  0.4125, 0.2, 0.2, 0.2, 0.2, 0.2]

        self.NODE1 = [92.22610988, 79.13110834, 121.1913795, 65.47463633, 86.47475944, 108.88683,
                      99.08583381, 194.8780421 - 50, 212.3303298 - 50, 203.9702702 - 50, 194.186895, 184.9074134,
                      218.0964762, 190.6869309, 114.0898789, 60.04401912, 109.9482686, 59.63049216,
                      37.11368379, 48.416151, 77.69792905, 76.78159562, 86.06799968, 92.7216716]
    def step(self, action, time):
        new_action = []
        next_state = []

        # action[0]是正的，表示充电，是负的，表示放电
        if time == 23:
            if self.current_capacity > self.min_soc:
                action[0] = -1  # 矫正动作为最小值，使电池放电
        if action[0] > 0:
            # 充电，应用系数0.95
            energy = action[0] * self.max_charge * self.factor
        else:
            # 放电，应用系数1/0.95
            energy = action[0] * self.max_charge / self.factor  # 计算动作对应的能量变化，action_battery是一个在[-1, 1]范围内的值
        updated_capacity = max(self.min_soc, min(self.max_soc, (self.current_capacity + energy)))  # 计算能量变化量  不是soc
        self.energy_change = updated_capacity - self.current_capacity  # 计算能量变化量  不是soc
        if self.energy_change > 0:
            # 充电，应用系数/0.95
            corrected_action0 = self.energy_change / self.max_charge / self.factor
        else:
            # 放电，应用系数*0.95
            corrected_action0 = self.energy_change / self.max_charge * self.factor
        self.current_capacity = updated_capacity  # 更新电池的当前容量 不是soc

        action[1] = round(action[1], 4)
        action[2] = round(action[2], 4)
        save_a2 = action[2]
        save_a2_sign = False
        save_a1 = action[1]
        save_a1_sign = False

        if round(action[1], 4) != 0:
            if round(self.current_electrolyzer_workspeed, 4) != 0:
                action[2] = 0
                save_a2_sign = True
            else:
                if round(self.current_fuel_cell_workspeed, 4) == 0:
                    action[2] = 0
                    save_a2_sign = True
                else:
                    action[1] = 0
                    save_a1_sign = True
        else:
            if round(self.current_electrolyzer_workspeed, 4) != 0:
                action[2] = 0
                save_a2_sign = True
            else:
                action[1] = 0
                save_a1_sign = True
        corrected_action1 = action[1]
        corrected_action2 = action[2]

        if round(action[2], 4) == 0:
            # 输入爬坡，剩余路程，当前工作速度
            b = count_interval(self.electrolyzer_climb,
                               (self.max_hydrogen - self.current_hydrogen) / self.electrolyzer_factor,
                               self.current_electrolyzer_workspeed)
            corrected_action1 = max(-self.electrolyzer_climb, -self.current_electrolyzer_workspeed,
                                    min(b, action[1] * self.electrolyzer_climb,
                                        self.electrolyzer_worklimit - self.current_electrolyzer_workspeed)) / self.electrolyzer_climb
            current_electrolyzer_workspeed = self.current_electrolyzer_workspeed + self.electrolyzer_climb * corrected_action1
            self.current_electrolyzer_workspeed = round(current_electrolyzer_workspeed, 4)

            if round(current_electrolyzer_workspeed, 4) == 0 and save_a2_sign == True:
                action[2] = save_a2
                b2 = count_interval(self.fuel_cell_climb,
                                    (self.current_hydrogen - self.min_hydrogen) * self.fuel_cell_factor,
                                    self.current_fuel_cell_workspeed)
                corrected_action2 = max(-self.fuel_cell_climb, -self.current_fuel_cell_workspeed,
                                        min(b2, action[2] * self.fuel_cell_climb,
                                            self.fuel_cell_worklimit - self.current_fuel_cell_workspeed)) / self.fuel_cell_climb
                current_fuel_cell_speed = corrected_action2 * self.fuel_cell_climb + self.current_fuel_cell_workspeed
                self.current_fuel_cell_workspeed = round(current_fuel_cell_speed,4)

            update_hydrogen = self.current_hydrogen + self.current_electrolyzer_workspeed * self.electrolyzer_factor - self.current_fuel_cell_workspeed / self.fuel_cell_factor

            self.current_hydrogen = round(update_hydrogen,4)
        else:
            b2 = count_interval(self.fuel_cell_climb,
                                (self.current_hydrogen - self.min_hydrogen) * self.fuel_cell_factor,
                                self.current_fuel_cell_workspeed)
            corrected_action2 = max(-self.fuel_cell_climb, -self.current_fuel_cell_workspeed,
                                    min(b2, action[2] * self.fuel_cell_climb,
                                        self.fuel_cell_worklimit - self.current_fuel_cell_workspeed)) / self.fuel_cell_climb
            current_fuel_cell_speed = corrected_action2 * self.fuel_cell_climb + self.current_fuel_cell_workspeed
            self.current_fuel_cell_workspeed = round(current_fuel_cell_speed, 4)

            if round(current_fuel_cell_speed, 4) == 0 and save_a1_sign == True:
                action[1] = save_a1
                b = count_interval(self.electrolyzer_climb,
                                   (self.max_hydrogen - self.current_hydrogen) / self.electrolyzer_factor,
                                   self.current_electrolyzer_workspeed)
                corrected_action1 = max(-self.electrolyzer_climb, -self.current_electrolyzer_workspeed,
                                        min(b, action[1] * self.electrolyzer_climb,
                                            self.electrolyzer_worklimit - self.current_electrolyzer_workspeed)) / self.electrolyzer_climb
                current_electrolyzer_workspeed = self.current_electrolyzer_workspeed + self.electrolyzer_climb * corrected_action1

                self.current_electrolyzer_workspeed = round(current_electrolyzer_workspeed, 4)

            update_hydrogen2 = self.current_hydrogen - current_fuel_cell_speed / self.fuel_cell_factor + self.current_electrolyzer_workspeed * self.electrolyzer_factor
            self.current_hydrogen = round(update_hydrogen2,4)

        if self.energy_change >= 0:
            unbalance = (self.pv_generation[time] *self.modulus[0] + self.wt_generation[time] * self.modulus[1] - self.load[time] * self.modulus[2] + self.NODE1[time])/2 - self.current_electrolyzer_workspeed + self.current_fuel_cell_workspeed - self.energy_change / 0.95
        else:
            unbalance = (self.pv_generation[time] *self.modulus[0] + self.wt_generation[time] * self.modulus[1] - self.load[time] * self.modulus[2] + self.NODE1[time])/2 - self.current_electrolyzer_workspeed + self.current_fuel_cell_workspeed - self.energy_change * 0.95
        state_sum = self.pv_generation[time] * self.modulus[0] + self.wt_generation[time] * self.modulus[1] - self.load[
            time] * self.modulus[2]
        reward1 = (self.current_electrolyzer_workspeed * 0.16 -
                   (0.008 * self.current_electrolyzer_workspeed + 0.00009 * self.current_electrolyzer_workspeed * self.current_electrolyzer_workspeed) -
                   (0.009 * self.current_fuel_cell_workspeed + 0.00011 * self.current_fuel_cell_workspeed * self.current_fuel_cell_workspeed))

        # 电池、电解水改变、燃料电池改变
        new_action.append(corrected_action0)
        new_action.append(corrected_action1)
        new_action.append(corrected_action2)

        # 时间，净负荷，电池soc，氢气容量，电解水工作速度，燃料电池工作速度
        next_state.append(round(((time + 1)%24) / 23,4))
        next_state.append((self.pv_generation[(time + 1) % 24]*self.modulus[0] + self.wt_generation[(time + 1) % 24] *self.modulus[1]- self.load[(time + 1) % 24] *self.modulus[2] + self.NODE1[(time + 1) % 24])/500)
        next_state.append(self.current_capacity / self.capacity)
        next_state.append(self.current_hydrogen / self.hydrogen_capacity)
        next_state.append(self.current_electrolyzer_workspeed / self.electrolyzer_worklimit)
        next_state.append(self.current_fuel_cell_workspeed / self.fuel_cell_worklimit)
        next_state.append(self.sell_price[(time + 1) % 24])
        next_state.append(self.buy_price[(time + 1) % 24])

        return new_action, next_state, unbalance, reward1,state_sum

    def reset(self,pv_data,wt_data,load_data):  # 重置电池状态
        self.energy_change = 0  # 能量变化量初始化为0
        self.current_capacity = 100  # 初始化电池状态
        self.current_electrolyzer_workspeed = 0  # 初始化电解水状态
        self.current_hydrogen = 150  # 储氢罐起始
        self.current_fuel_cell_workspeed = 0  # 初始化燃料电池状态
        self.pv_generation = pv_data  # 光伏发电数据引用，类型为列表
        self.wt_generation = wt_data
        self.load = load_data

    def get_state(self, time):
        state = []
        state.append(round(time / 23,4))
        state.append((
            self.pv_generation[time % 24]*self.modulus[0] + self.wt_generation[time % 24] * self.modulus[1] - self.load[time  % 24]*self.modulus[2] + self.NODE1[time % 24])/500)
        state.append(self.current_capacity / self.capacity)
        state.append(self.current_hydrogen / self.hydrogen_capacity)
        state.append(self.current_electrolyzer_workspeed / self.electrolyzer_worklimit)
        state.append(self.current_fuel_cell_workspeed / self.fuel_cell_worklimit)
        state.append(self.sell_price[time % 24])
        state.append(self.buy_price[time % 24])
        return state


class Node2:
    def __init__(self, node_id, pv_generation, wt_generation, load):
        self.node_id = node_id  # 节点ID

        self.pv_generation = pv_generation  # 光伏发电数据引用，类型为列表
        self.wt_generation = wt_generation
        self.load = load  # 电力负载数据引用，类型为列表
        self.modulus = [2,2,1]

        self.energy_change = 0  # 能量变化量初始化为0
        self.capacity = 250  # 电池的总容量，单位为kWh
        self.max_soc = 200
        self.min_soc = 50
        self.factor = 0.95
        self.max_charge = 50
        self.current_capacity = 50  # 初始化电池状态
        self.soc = self.current_capacity / self.capacity  # 初始化节点的SOC（电池的当前容量），并确保为浮点值

        # 发电机
        self.current_diesel_engine_workspeed = 0  # 初始化发电机状态
        self.diesel_engine_climb = 50 # 发电机的爬坡
        self.diesel_engine_worklimit = 100

        self.buy_gas_limit = 50
        # 储气罐
        self.current_hydrogen = 150  # 储氢罐起始
        self.hydrogen_capacity = 500  # 电池的总容量，单位为kWh
        self.max_hydrogen = 400  # 氢气上限
        self.min_hydrogen = 150  # 氢气下限
        # 燃料电池
        self.current_fuel_cell_workspeed = 0  # 初始化燃料电池
        self.fuel_cell_climb = 75  # 燃料电池的爬坡
        self.fuel_cell_worklimit = 150 # 燃料电池的工作限制
        self.fuel_cell_factor = 0.7 # 能量转换效率
        self.buy_price = [0.3578, 0.3578, 0.3578, 0.3578, 0.3578, 0.3578,
                          0.3578, 0.3578, 0.8325, 0.8325, 0.8325, 0.3578,
                          0.3578, 0.8325, 0.8325, 0.8325, 0.8325, 0.8325,
                          0.8325, 1.2109, 1.2109, 0.8325, 0.3578, 0.3578, 0.3578]
        self.sell_price = [0.2, 0.2, 0.2, 0.2, 0.2, 0.2,
                  0.2, 0.2, 0.4125, 0.4125, 0.4125, 0.2,
                  0.2, 0.4125, 0.4125, 0.4125, 0.4125, 0.4125,
                  0.4125, 0.2, 0.2, 0.2, 0.2, 0.2]

        self.NODE2 = [-27.42786139, -37.46380191, -29.08689146, -39.98063555, -39.95162173, -22.9110332,
                 -29.89239206, 14.69925088, -85.7622598, -128.266853, -153.3504564, -180.4597494,
                 -170.228744, -167.3553783, -131.0734733, -149.1190553, -189.7732588, -206.1887842,
                 -239.6008373, -136.0302747, -89.08933104, -61.34851929, -31.67918964, -14.34519281]

    def step(self, action, time):  # 根据传入的动作更新电池状态
        new_action = []
        next_state = []

        # action[0]是正的，表示充电，是负的，表示放电
        if time == 23:
            if self.current_capacity > self.min_soc:
                action[0] = -1  # 矫正动作为最小值，使电池放电
        if action[0] > 0:
            # 充电，应用系数0.95
            energy = action[0] * self.max_charge * self.factor
        else:
            # 放电，应用系数1/0.95
            energy = action[0] * self.max_charge / self.factor  # 计算动作对应的能量变化，action_battery是一个在[-1, 1]范围内的值
        updated_capacity = max(self.min_soc, min(self.max_soc, (self.current_capacity + energy)))  # 计算能量变化量  不是soc
        self.energy_change = updated_capacity - self.current_capacity  # 计算能量变化量  不是soc
        if self.energy_change > 0:
            # 充电，应用系数/0.95
            corrected_action0 = self.energy_change / self.max_charge / self.factor
        else:
            # 放电，应用系数*0.95
            corrected_action0 = self.energy_change / self.max_charge * self.factor
        self.current_capacity = updated_capacity  # 更新电池的当前容量 不是soc

        action[1] = max(action[1], 0)
        action[1] = round(action[1], 4)
        action[2] = round(action[2], 4)
        action[3] = round(action[3], 4)
        save_a1 = action[1]
        save_a1_sign = False
        save_a2 = action[2]
        save_a2_sign = False
        if action[1] != 0:
            if round(self.current_fuel_cell_workspeed, 4) == 0:
                action[2] = 0
                save_a2_sign = True
            else:
                action[1] = 0
                save_a1_sign = True

        corrected_action1 = action[1]
        corrected_action2 = action[2]
        if action[2] == 0:
            buygas = action[1] * self.buy_gas_limit
            updated_hydrogen_capacity = max(self.min_hydrogen, min(self.max_hydrogen, self.current_hydrogen + buygas))
            gas_change = updated_hydrogen_capacity - self.current_hydrogen

            corrected_action1 = gas_change / self.buy_gas_limit

            if round(gas_change, 4) == 0 and save_a2_sign == True:
                action[2] = save_a2
                b2 = count_interval(self.fuel_cell_climb,
                                    (self.current_hydrogen - self.min_hydrogen) * self.fuel_cell_factor,
                                    self.current_fuel_cell_workspeed)
                corrected_action2 = max(-self.fuel_cell_climb, -self.current_fuel_cell_workspeed,
                                        min(b2, action[2] * self.fuel_cell_climb,
                                            self.fuel_cell_worklimit - self.current_fuel_cell_workspeed)) / self.fuel_cell_climb
                current_fuel_cell_speed = corrected_action2 * self.fuel_cell_climb + self.current_fuel_cell_workspeed
                self.current_fuel_cell_workspeed = round(current_fuel_cell_speed, 4)

            updated_hydrogen = self.current_hydrogen + gas_change - self.current_fuel_cell_workspeed / self.fuel_cell_factor
            self.current_hydrogen = round(updated_hydrogen,4)

        else:
            b2 = count_interval(self.fuel_cell_climb,
                                (self.current_hydrogen - self.min_hydrogen) * self.fuel_cell_factor,
                                self.current_fuel_cell_workspeed)

            corrected_action2 = max(-self.fuel_cell_climb, -self.current_fuel_cell_workspeed,
                                    min(b2, action[2] * self.fuel_cell_climb,
                                        self.fuel_cell_worklimit - self.current_fuel_cell_workspeed)) / self.fuel_cell_climb
            current_fuel_cell_speed = corrected_action2 * self.fuel_cell_climb + self.current_fuel_cell_workspeed
            self.current_fuel_cell_workspeed = round(current_fuel_cell_speed, 4)

            if round(current_fuel_cell_speed, 4) == 0 and save_a1_sign == True:
                action[1] = save_a1
                buygas = action[1] * self.buy_gas_limit  # 计算动作对应的能量变化，action_battery是一个在[-1, 1]范围内的值
                updated_hydrogen3 = max(self.min_hydrogen,
                                        min(self.max_hydrogen, self.current_hydrogen + buygas))  # 计算能量变化量  不是soc
                updated_hydrogen_capacity = updated_hydrogen3 - self.current_hydrogen  # 计算能量变化量  不是soc
                corrected_action1 = updated_hydrogen_capacity / self.buy_gas_limit

            updated_hydrogen_capacity = self.current_hydrogen - self.current_fuel_cell_workspeed / self.fuel_cell_factor + corrected_action1 * self.buy_gas_limit
            self.current_hydrogen = round(updated_hydrogen_capacity, 4)

        # action[3] 是发电机动作
        corrected_diesel_engine_action = max(0, min(self.diesel_engine_worklimit, action[
            3] * self.diesel_engine_climb + self.current_diesel_engine_workspeed))
        corrected_action3 = (corrected_diesel_engine_action - self.current_diesel_engine_workspeed) / self.diesel_engine_climb
        self.current_diesel_engine_workspeed = round(corrected_diesel_engine_action,4)

        if self.energy_change >= 0:
            unbalance = (self.pv_generation[time] *self.modulus[0] + self.wt_generation[time] *self.modulus[1] - self.load[time] *self.modulus[2] + self.NODE2[time]) / 2+ self.current_diesel_engine_workspeed + self.current_fuel_cell_workspeed - self.energy_change / 0.95
        else:
            unbalance = (self.pv_generation[time] *self.modulus[0] + self.wt_generation[time] *self.modulus[1] - self.load[time] *self.modulus[2] + self.NODE2[time]) / 2 + self.current_diesel_engine_workspeed + self.current_fuel_cell_workspeed - self.energy_change * 0.95
        state_sum = self.pv_generation[time] * self.modulus[0] + self.wt_generation[time] * self.modulus[1] - self.load[
            time] * self.modulus[2]
        reward2 = (- (0.011 * self.current_fuel_cell_workspeed + 0.00009 * self.current_fuel_cell_workspeed * self.current_fuel_cell_workspeed)
                   - (0.08 * self.current_diesel_engine_workspeed + 0.002 * self.current_diesel_engine_workspeed * self.current_diesel_engine_workspeed)
                   - corrected_action1 * self.buy_gas_limit * 0.25 - 0.0316 * (0.76572 * self.current_diesel_engine_workspeed + 0.202 * corrected_action1 * self.buy_gas_limit))

        new_action.append(corrected_action0)
        new_action.append(corrected_action1)
        new_action.append(corrected_action2)
        new_action.append(corrected_action3)


        next_state.append(round(((time + 1)%24) / 23,4))
        next_state.append((self.pv_generation[(time + 1) % 24] * self.modulus[0] + self.wt_generation[(time + 1) % 24] * self.modulus[1] - self.load[(time + 1) % 24] * self.modulus[2] + self.NODE2[(time + 1) % 24])/500)
        next_state.append(self.current_capacity / self.capacity)
        next_state.append(self.current_hydrogen / self.hydrogen_capacity)
        next_state.append(self.current_fuel_cell_workspeed / self.fuel_cell_worklimit)
        next_state.append(self.current_diesel_engine_workspeed / self.diesel_engine_worklimit)
        next_state.append(self.sell_price[(time + 1) % 24])
        next_state.append(self.buy_price[(time + 1) % 24])

        return new_action, next_state, unbalance, reward2,state_sum

    def reset(self,pv_data,wt_data,load_data):  # 重置电池状态
        self.energy_change = 0  # 能量变化量初始化为0
        self.current_capacity = 50  # 初始化电池状态
        # 发电机
        self.current_diesel_engine_workspeed = 0  # 初始化发电机状态
        # 储气罐
        self.current_hydrogen = 150  # 储氢罐起始
        # 燃料电池
        self.current_fuel_cell_workspeed = 0  # 初始化燃料电池
        self.pv_generation = pv_data  # 光伏发电数据引用，类型为列表
        self.wt_generation = wt_data
        self.load = load_data

    def get_state(self, time):
        state = []
        state.append(round(time / 23,4))
        state.append((self.pv_generation[time % 24] * self.modulus[0] + self.wt_generation[time % 24] * self.modulus[1] - self.load[time  % 24] * self.modulus[2] + self.NODE2[time % 24])/500)
        state.append(self.current_capacity / self.capacity)
        state.append(self.current_hydrogen / self.hydrogen_capacity)
        state.append(self.current_fuel_cell_workspeed / self.fuel_cell_worklimit)
        state.append(self.current_diesel_engine_workspeed / self.diesel_engine_worklimit)
        state.append(self.sell_price[time % 24])
        state.append(self.buy_price[time % 24])

        return state

class Node3:
    def __init__(self, node_id, pv_generation, wt_generation, load, hot_load):
        self.node_id = node_id  # 节点ID

        self.pv_generation = pv_generation   # 光伏发电数据引用，类型为列表
        self.wt_generation = wt_generation
        self.load = load  # 电力负载数据引用，类型为列表
        self.thermal_load = hot_load
        self.modulus = [3,1,0.5,0.3]

        self.energy_change = 0  # 能量变化量初始化为0
        self.capacity = 1000  # 电池的总容量，单位为kWh
        self.max_soc = 800  # 最大充电状态，即电池最大可以充到总容量的80%
        self.min_soc = 200  # 最小充电状态，即电池最小可以充到总容量的20%
        self.factor = 0.95  # 电池的退化系数，用于计算能量变化对成本的影响
        self.max_charge = 100  # 最大充电功率，单位为kW
        self.current_capacity = 200  # 初始化电池状态


        self.boiler_worklimit = 100
        self.boiler_factor = 0.95

        self.current_chp_workspeed = 0  # 初始化热电联产购气状态
        self.chp_worklimit = 100
        self.chp_energy_factor = 0.35
        self.chp_thermal_factor = 0.5
        self.buy_price = [0.3578, 0.3578, 0.3578, 0.3578, 0.3578, 0.3578,
                          0.3578, 0.3578, 0.8325, 0.8325, 0.8325, 0.3578,
                          0.3578, 0.8325, 0.8325, 0.8325, 0.8325, 0.8325,
                          0.8325, 1.2109, 1.2109, 0.8325, 0.3578, 0.3578, 0.3578]
        self.sell_price = [0.2, 0.2, 0.2, 0.2, 0.2, 0.2,
                  0.2, 0.2, 0.4125, 0.4125, 0.4125, 0.2,
                  0.2, 0.4125, 0.4125, 0.4125, 0.4125, 0.4125,
                  0.4125, 0.2, 0.2, 0.2, 0.2, 0.2]

        self.NODE3 = [-13.71393069, -18.73190095, -14.54344573, -19.99031777, -19.97581087, -11.4555166,
                 -14.94619603, 82.2892145, 60.09545002, 47.88859845, 39.65494082, 79.7678586,
                 100.7682293, 82.28832041, 37.5464732, -15.12914594, -53.02769366, -103.0943921,
                 - 119.8004186, -68.01513734, -44.54466552, -30.67425965, -15.83959482, -7.172596406]

    def step(self, action, time):  # 根据传入的动作更新电池状态
        new_action = []
        next_state = []
        # action[0]是正的，表示充电，是负的，表示放电
        if time == 23:
            if self.current_capacity > self.min_soc:
                action[0] = -1  # 矫正动作为最小值，使电池放电

        if action[0] > 0:
            # 充电，应用系数0.95
            energy = action[0] * self.max_charge * self.factor
        else:
            # 放电，应用系数1/0.95
            energy = action[0] * self.max_charge / self.factor  # 计算动作对应的能量变化，action_battery是一个在[-1, 1]范围内的值
        updated_capacity = max(self.min_soc, min(self.max_soc, (self.current_capacity + energy)))  # 计算能量变化量  不是soc
        self.energy_change = updated_capacity - self.current_capacity  # 计算能量变化量  不是soc
        if self.energy_change > 0:
            # 充电，应用系数/0.95
            corrected_action0 = self.energy_change / self.max_charge / self.factor
        else:
            # 放电，应用系数*0.95
            corrected_action0 = self.energy_change / self.max_charge * self.factor
        self.current_capacity = updated_capacity  # 更新电池的当前容量 不是soc

        action[1] = round(action[1], 4)

        corrected_boiler_action = max(0,min(self.boiler_worklimit * action[1] * self.boiler_factor,self.thermal_load[time] * self.modulus[3]))

        corrected_action1 = corrected_boiler_action / self.boiler_worklimit / self.boiler_factor
        # 由电锅炉计算出chp动作（气）
        self.current_chp_workspeed = round((self.thermal_load[time] * self.modulus[3] - corrected_boiler_action) / self.chp_thermal_factor ,4)

        if self.energy_change >= 0:
            unbalance = (self.pv_generation[time] * self.modulus[0] + self.wt_generation[time] * self.modulus[1] - self.load[time] * self.modulus[2] + self.NODE3[time])/2 - corrected_boiler_action / self.boiler_factor  + self.current_chp_workspeed * self.chp_energy_factor - self.energy_change / 0.95  # + self.current_diesel_engine_workspeed
        else:
            unbalance = (self.pv_generation[time] * self.modulus[0] + self.wt_generation[time] * self.modulus[1] - self.load[time] * self.modulus[2] + self.NODE3[time])/2 - corrected_boiler_action / self.boiler_factor + self.current_chp_workspeed * self.chp_energy_factor - self.energy_change * 0.95 # + self.current_diesel_engine_workspeed
        state_sum = self.pv_generation[time] * self.modulus[0] + self.wt_generation[time] * self.modulus[1] - self.load[
            time] * self.modulus[2]
        # 电池、发电机、电锅炉
        new_action.append(corrected_action0)
        new_action.append(corrected_action1)
        # new_action.append(corrected_action2)
        # (0.1 * self.current_diesel_engine_workspeed + 0.001 * self.current_diesel_engine_workspeed * self.current_diesel_engine_workspeed) -
        reward3 = (- (0.1 * self.current_chp_workspeed * 0.35 + 0.002 * 0.35 * 0.35 * self.current_chp_workspeed * self.current_chp_workspeed) - self.current_chp_workspeed * 0.25
                   - (0.011 * corrected_boiler_action / self.boiler_factor + 0.0008 * corrected_boiler_action * corrected_boiler_action / self.boiler_factor / self.boiler_factor) - 0.0316 * (
                           self.current_chp_workspeed * 0.202))  # 0.76572 * self.current_diesel_engine_workspeed +

        # 时间，净负荷，电池soc，发电机速度/200，热负荷，
        next_state.append(round(((time+1)%24)/23,4))
        next_state.append((self.pv_generation[(time+1)%24] * self.modulus[0] + self.wt_generation[(time+1)%24] * self.modulus[1] - self.load[(time+1)%24] *self.modulus[2]+ self.NODE3[(time + 1) % 24])/500)
        next_state.append(self.current_capacity/self.capacity)
        # next_state.append(self.current_diesel_engine_workspeed / self.diesel_engine_worklimit)
        next_state.append(self.thermal_load[(time+1)%24] * self.modulus[3]/250)
        next_state.append(self.sell_price[(time + 1) % 24])
        next_state.append(self.buy_price[(time + 1) % 24])

        return new_action, next_state, unbalance, reward3,state_sum

    def get_state(self, time):
        state = []
        state.append(round(time / 23,4))
        state.append((self.pv_generation[time % 24] *self.modulus[0] + self.wt_generation[time % 24] * self.modulus[1] - self.load[time % 24] * self.modulus[2]+ self.NODE3[time % 24])/500)
        state.append(self.current_capacity / self.capacity)
        # state.append(self.current_diesel_engine_workspeed / self.diesel_engine_worklimit)
        state.append(self.thermal_load[time % 24] * self.modulus[3]/250)
        state.append(self.sell_price[time % 24])
        state.append(self.buy_price[time % 24])

        return state

    def reset(self,pv_data,wt_data,load_data,hot_load):
        self.energy_change = 0  # 能量变化量初始化为0
        self.current_capacity = 200  # 初始化电池状态
        # 柴油发电机
        # self.current_diesel_engine_workspeed = 0  # 初始化电池状态
        # 热电联产
        self.current_chp_workspeed = 0  # 初始化热电联产购气状态
        self.pv_generation = pv_data  # 光伏发电数据引用，类型为列表
        self.wt_generation = wt_data
        self.load = load_data
        self.thermal_load = hot_load


class Node4:
    def __init__(self, node_id, pv_generation, wt_generation, load):
        self.node_id = node_id  # 节点ID

        self.pv_generation = pv_generation   # 光伏发电数据引用，类型为列表
        self.wt_generation = wt_generation
        self.load = load  # 电力负载数据引用，类型为列表
        self.modulus = [3.5,3.5,0.5]

        self.energy_change = 0  # 能量变化量初始化为0
        self.capacity = 500  # 电池的总容量，单位为kWh
        self.max_soc = 400  # 最大充电状态，即电池最大可以充到总容量的80%
        self.min_soc = 100  # 最小充电状态，即电池最小可以充到总容量的20%
        self.factor = 0.95  # 电池的退化系数，用于计算能量变化对成本的影响
        self.max_charge = 100  # 最大充电功率，单位为kW
        self.current_capacity = 100  # 初始化电池状态

        self.current_electrolyzer_workspeed = 0  # 初始化电解水状态
        self.electrolyzer_climb = 50 # 电解水的爬坡
        self.electrolyzer_worklimit = 100
        self.electrolyzer_factor = 0.8

        self.current_hydrogen = 150  # 储氢罐起始
        self.hydrogen_capacity = 500  # 储氢罐的总容量，单位为kWh
        self.max_hydrogen = 400  # 氢气上限
        self.min_hydrogen = 150  # 氢气下限

        self.current_fuel_cell_workspeed = 0  # 初始化燃料电池状态
        self.fuel_cell_climb = 50  # 燃料电池的爬坡
        self.fuel_cell_worklimit = 100
        self.fuel_cell_factor = 0.6
        self.buy_price = [0.3578, 0.3578, 0.3578, 0.3578, 0.3578, 0.3578,
                          0.3578, 0.3578, 0.8325, 0.8325, 0.8325, 0.3578,
                          0.3578, 0.8325, 0.8325, 0.8325, 0.8325, 0.8325,
                          0.8325, 1.2109, 1.2109, 0.8325, 0.3578, 0.3578, 0.3578]
        self.sell_price = [0.2, 0.2, 0.2, 0.2, 0.2, 0.2,
                  0.2, 0.2, 0.4125, 0.4125, 0.4125, 0.2,
                  0.2, 0.4125, 0.4125, 0.4125, 0.4125, 0.4125,
                  0.4125, 0.2, 0.2, 0.2, 0.2, 0.2]

        self.NODE4 = [102.6095005, 84.39175229, 136.6574564, 67.4748768, 92.48262222, 123.6266699,
                      110.1303661, 176.0742293, 148.1765763, 119.8771497, 98.22450158, 37.27117528,
                      66.79144004, 50.79943009, 19.27138591, - 14.97681998 + 50, 47.80079134, 16.98685672 + 50,
                      - 18.56964335 + 50, 22.0112983, 69.16461455, 75.33919212, 94.16497376, 106.6258681]
    def step(self, action, time):  # 根据传入的动作更新电池状态
        new_action = []
        next_state = []

        # action[0]是正的，表示充电，是负的，表示放电
        if time == 23:
            if self.current_capacity > self.min_soc:
                action[0] = -1  # 矫正动作为最小值，使电池放电
        if action[0] > 0:
            # 充电，应用系数0.95
            energy = action[0] * self.max_charge * self.factor
        else:
            # 放电，应用系数1/0.95
            energy = action[0] * self.max_charge / self.factor  # 计算动作对应的能量变化，action_battery是一个在[-1, 1]范围内的值
        updated_capacity = max(self.min_soc, min(self.max_soc, (self.current_capacity + energy)))  # 计算能量变化量  不是soc
        self.energy_change = updated_capacity - self.current_capacity  # 计算能量变化量  不是soc
        if self.energy_change > 0:
            # 充电，应用系数/0.95
            corrected_action0 = self.energy_change / self.max_charge / self.factor
        else:
            # 放电，应用系数*0.95
            corrected_action0 = self.energy_change / self.max_charge * self.factor
        self.current_capacity = updated_capacity  # 更新电池的当前容量 不是soc

        action[1] = round(action[1],4)
        action[2] = round(action[2], 4)
        save_a2 = action[2]
        save_a2_sign = False
        save_a1 = action[1]
        save_a1_sign = False

        if action[1] != 0:
            if round(self.current_electrolyzer_workspeed, 4) != 0:
                action[2] = 0
                save_a2_sign = True
            else:
                if round(self.current_fuel_cell_workspeed, 4) == 0:
                    action[2] = 0
                    save_a2_sign = True
                else:
                    action[1] = 0
                    save_a1_sign = True
        else:
            if round(self.current_electrolyzer_workspeed, 4) != 0:
                action[2] = 0
                save_a2_sign = True
            else:
                action[1] = 0
                save_a1_sign = True
        corrected_action1 = action[1]
        corrected_action2 = action[2]
        if round(action[2], 4) == 0:
            # 输入爬坡，剩余路程，当前工作速度
            b = count_interval(self.electrolyzer_climb,
                               (self.max_hydrogen - self.current_hydrogen) / self.electrolyzer_factor,
                               self.current_electrolyzer_workspeed)
            corrected_action1 = max(-self.electrolyzer_climb, -self.current_electrolyzer_workspeed,
                                    min(b, action[1] * self.electrolyzer_climb,
                                        self.electrolyzer_worklimit - self.current_electrolyzer_workspeed)) / self.electrolyzer_climb
            current_electrolyzer_workspeed = self.current_electrolyzer_workspeed + self.electrolyzer_climb * corrected_action1
            self.current_electrolyzer_workspeed = round(current_electrolyzer_workspeed,4)

            if round(current_electrolyzer_workspeed, 4) == 0 and save_a2_sign == True:
                action[2] = save_a2
                b2 = count_interval(self.fuel_cell_climb,
                                    (self.current_hydrogen - self.min_hydrogen) * self.fuel_cell_factor,
                                    self.current_fuel_cell_workspeed)
                corrected_action2 = max(-self.fuel_cell_climb, -self.current_fuel_cell_workspeed,
                                        min(b2, action[2] * self.fuel_cell_climb,
                                            self.fuel_cell_worklimit - self.current_fuel_cell_workspeed)) / self.fuel_cell_climb
                current_fuel_cell_speed = corrected_action2 * self.fuel_cell_climb + self.current_fuel_cell_workspeed
                self.current_fuel_cell_workspeed = round(current_fuel_cell_speed,4)

            update_hydrogen = self.current_hydrogen + self.current_electrolyzer_workspeed * self.electrolyzer_factor - self.current_fuel_cell_workspeed / self.fuel_cell_factor

            self.current_hydrogen = round(update_hydrogen,4)
        else:
            b2 = count_interval(self.fuel_cell_climb,
                                (self.current_hydrogen - self.min_hydrogen) * self.fuel_cell_factor,
                                self.current_fuel_cell_workspeed)
            corrected_action2 = max(-self.fuel_cell_climb, -self.current_fuel_cell_workspeed,
                                    min(b2, action[2] * self.fuel_cell_climb,
                                        self.fuel_cell_worklimit - self.current_fuel_cell_workspeed)) / self.fuel_cell_climb
            current_fuel_cell_speed = corrected_action2 * self.fuel_cell_climb + self.current_fuel_cell_workspeed
            self.current_fuel_cell_workspeed = round(current_fuel_cell_speed, 4)

            if round(current_fuel_cell_speed, 4) == 0 and save_a1_sign == True:
                action[1] = save_a1
                b = count_interval(self.electrolyzer_climb,
                                   (self.max_hydrogen - self.current_hydrogen) / self.electrolyzer_factor,
                                   self.current_electrolyzer_workspeed)
                corrected_action1 = max(-self.electrolyzer_climb, -self.current_electrolyzer_workspeed,
                                        min(b, action[1] * self.electrolyzer_climb,
                                            self.electrolyzer_worklimit - self.current_electrolyzer_workspeed)) / self.electrolyzer_climb
                current_electrolyzer_workspeed = self.current_electrolyzer_workspeed + self.electrolyzer_climb * corrected_action1

                self.current_electrolyzer_workspeed = round(current_electrolyzer_workspeed,4)

            update_hydrogen2 = self.current_hydrogen - current_fuel_cell_speed / self.fuel_cell_factor + self.current_electrolyzer_workspeed * self.electrolyzer_factor
            self.current_hydrogen = round(update_hydrogen2,4)

        if self.energy_change >= 0:
            unbalance = (self.pv_generation[time] *self.modulus[0] + self.wt_generation[time] * self.modulus[1] - self.load[time] * self.modulus[2] + self.NODE4[time])/2 - self.current_electrolyzer_workspeed + self.current_fuel_cell_workspeed - self.energy_change / 0.95
        else:
            unbalance = (self.pv_generation[time] *self.modulus[0] + self.wt_generation[time] * self.modulus[1] - self.load[time] * self.modulus[2] + self.NODE4[time])/2 - self.current_electrolyzer_workspeed + self.current_fuel_cell_workspeed - self.energy_change * 0.95
        state_sum = self.pv_generation[time] * self.modulus[0] + self.wt_generation[time] * self.modulus[1] - self.load[
            time] * self.modulus[2]
        reward4 = (self.current_electrolyzer_workspeed * 0.16 -
                   (0.01 * self.current_electrolyzer_workspeed + 0.0001 * self.current_electrolyzer_workspeed * self.current_electrolyzer_workspeed) -
                   (0.009 * self.current_fuel_cell_workspeed + 0.0001 * self.current_fuel_cell_workspeed * self.current_fuel_cell_workspeed))

        # 电池、电解水改变、燃料电池改变
        new_action.append(corrected_action0)
        new_action.append(corrected_action1)
        new_action.append(corrected_action2)

        # 时间，净负荷，电池soc，氢气容量，电解水工作速度，燃料电池工作速度
        next_state.append(round(((time + 1)%24) / 23,4))
        next_state.append((self.pv_generation[(time + 1) % 24]*self.modulus[0] + self.wt_generation[(time + 1) % 24] *self.modulus[1]- self.load[(time + 1) % 24] *self.modulus[2]+ self.NODE4[(time + 1) % 24])/500)
        next_state.append(self.current_capacity / self.capacity)
        next_state.append(self.current_hydrogen / self.hydrogen_capacity)
        next_state.append(self.current_electrolyzer_workspeed / self.electrolyzer_worklimit)
        next_state.append(self.current_fuel_cell_workspeed / self.fuel_cell_worklimit)
        next_state.append(self.sell_price[(time + 1) % 24])
        next_state.append(self.buy_price[(time + 1) % 24])

        return new_action, next_state, unbalance, reward4,state_sum

    def reset(self,pv_data,wt_data,load_data):  # 重置电池状态
        self.energy_change = 0  # 能量变化量初始化为0
        self.current_capacity = 100  # 初始化电池状态
        self.current_electrolyzer_workspeed = 0  # 初始化电解水状态
        self.current_hydrogen = 150  # 储氢罐起始
        self.current_fuel_cell_workspeed = 0  # 初始化燃料电池状态
        self.pv_generation = pv_data  # 光伏发电数据引用，类型为列表
        self.wt_generation = wt_data
        self.load = load_data

    def get_state(self, time):
        state = []
        state.append(round(time / 23,4))
        state.append((
            self.pv_generation[time % 24]*self.modulus[0] + self.wt_generation[time % 24] * self.modulus[1] - self.load[time % 24]*self.modulus[2] + self.NODE4[time % 24])/500)
        state.append(self.current_capacity / self.capacity)
        state.append(self.current_hydrogen / self.hydrogen_capacity)
        state.append(self.current_electrolyzer_workspeed / self.electrolyzer_worklimit)
        state.append(self.current_fuel_cell_workspeed / self.fuel_cell_worklimit)
        state.append(self.sell_price[time % 24])
        state.append(self.buy_price[time % 24])
        return state


class Node5:
    def __init__(self, node_id, pv_generation, wt_generation, load):
        self.node_id = node_id  # 节点ID

        self.pv_generation = pv_generation  # 光伏发电数据引用，类型为列表
        self.wt_generation = wt_generation
        self.load = load  # 电力负载数据引用，类型为列表
        self.modulus = [1.5,2.5,1]

        self.energy_change = 0  # 能量变化量初始化为0
        self.capacity = 250  # 电池的总容量，单位为kWh
        self.max_soc = 200  # 最大充电状态，即电池最大可以充到总容量的80%
        self.min_soc = 50  # 最小充电状态，即电池最小可以充到总容量的20%
        self.factor = 0.95  # 电池的退化系数，用于计算能量变化对成本的影响
        self.max_charge = 50  # 最大充电功率，单位为kW
        self.current_capacity = 50  # 初始化电池状态
        self.soc = self.current_capacity / self.capacity  # 初始化节点的SOC（电池的当前容量），并确保为浮点值

        # 发电机
        self.current_diesel_engine_workspeed = 0  # 初始化发电机状态
        self.diesel_engine_climb = 51 # 发电机的爬坡
        self.diesel_engine_worklimit = 100

        self.buy_gas_limit = 50
        # 储气罐
        self.current_hydrogen = 150  # 储氢罐起始
        self.hydrogen_capacity = 500  # 电池的总容量，单位为kWh
        self.max_hydrogen = 400  # 氢气上限
        self.min_hydrogen = 150  # 氢气下限
        # 燃料电池
        self.current_fuel_cell_workspeed = 0  # 初始化燃料电池
        self.fuel_cell_climb = 70  # 燃料电池的爬坡
        self.fuel_cell_worklimit = 150 # 燃料电池的工作限制
        self.fuel_cell_factor = 0.7 # 能量转换效率

        self.buy_price = [0.3578, 0.3578, 0.3578, 0.3578, 0.3578, 0.3578,
                          0.3578, 0.3578, 0.8325, 0.8325, 0.8325, 0.3578,
                          0.3578, 0.8325, 0.8325, 0.8325, 0.8325, 0.8325,
                          0.8325, 1.2109, 1.2109, 0.8325, 0.3578, 0.3578, 0.3578]
        self.sell_price = [0.2, 0.2, 0.2, 0.2, 0.2, 0.2,
                  0.2, 0.2, 0.4125, 0.4125, 0.4125, 0.2,
                  0.2, 0.4125, 0.4125, 0.4125, 0.4125, 0.4125,
                  0.4125, 0.2, 0.2, 0.2, 0.2, 0.2]

        self.NODE5 = [-4.163175157, - 16.83907126, 1.153288977, - 22.48759663, - 17.45993512, 4.105404099,
                 - 4.877079628, 18.46833602, - 88.74135054, - 136.2735478, - 164.9025781, - 222.9586327,
                 - 214.2006221, - 206.8463583, - 155.3451327, - 160.9746664, - 177.979349, - 182.1725344,
                 - 219.3546822, - 118.0249876, - 66.34747503, - 40.14582894, - 9.67827592, 8.414500085]

    def step(self, action, time):  # 根据传入的动作更新电池状态
        new_action = []
        next_state = []

        # action[0]是正的，表示充电，是负的，表示放电
        if time == 23:
            if self.current_capacity > self.min_soc:
                action[0] = -1  # 矫正动作为最小值，使电池放电
        if action[0] > 0:
            # 充电，应用系数0.95
            energy = action[0] * self.max_charge * self.factor
        else:
            # 放电，应用系数1/0.95
            energy = action[0] * self.max_charge / self.factor  # 计算动作对应的能量变化，action_battery是一个在[-1, 1]范围内的值
        updated_capacity = max(self.min_soc, min(self.max_soc, (self.current_capacity + energy)))  # 计算能量变化量  不是soc
        self.energy_change = updated_capacity - self.current_capacity  # 计算能量变化量  不是soc
        if self.energy_change > 0:
            # 充电，应用系数/0.95
            corrected_action0 = self.energy_change / self.max_charge / self.factor
        else:
            # 放电，应用系数*0.95
            corrected_action0 = self.energy_change / self.max_charge * self.factor
        self.current_capacity = updated_capacity  # 更新电池的当前容量 不是soc

        action[1] = max(action[1], 0)

        action[1] = round(action[1],4)
        action[2] = round(action[2], 4)
        action[3] = round(action[3], 4)
        save_a1 = action[1]
        save_a1_sign = False
        save_a2 = action[2]
        save_a2_sign = False
        if action[1] != 0:
            if round(self.current_fuel_cell_workspeed, 4) == 0:
                action[2] = 0
                save_a2_sign = True
            else:
                action[1] = 0
                save_a1_sign = True

        corrected_action1 = action[1]
        corrected_action2 = action[2]
        if round(action[2], 4) == 0:
            buygas = action[1] * self.buy_gas_limit  # 计算动作对应的能量变化，action_battery是一个在[-1, 1]范围内的值
            updated_hydrogen_capacity = max(self.min_hydrogen, min(self.max_hydrogen, self.current_hydrogen + buygas))
            # 计算能量变化量  不是soc
            gas_change = updated_hydrogen_capacity - self.current_hydrogen  # 计算能量变化量  不是soc

            corrected_action1 = gas_change / self.buy_gas_limit

            if round(gas_change, 4) == 0 and save_a2_sign == True:
                action[2] = save_a2
                b2 = count_interval(self.fuel_cell_climb,
                                    (self.current_hydrogen - self.min_hydrogen) * self.fuel_cell_factor,
                                    self.current_fuel_cell_workspeed)
                corrected_action2 = max(-self.fuel_cell_climb, -self.current_fuel_cell_workspeed,
                                        min(b2, action[2] * self.fuel_cell_climb,
                                            self.fuel_cell_worklimit - self.current_fuel_cell_workspeed)) / self.fuel_cell_climb
                current_fuel_cell_speed = corrected_action2 * self.fuel_cell_climb + self.current_fuel_cell_workspeed
                self.current_fuel_cell_workspeed = round(current_fuel_cell_speed, 4)

            updated_hydrogen = self.current_hydrogen + gas_change - self.current_fuel_cell_workspeed / self.fuel_cell_factor
            self.current_hydrogen = round(updated_hydrogen, 4)

        else:
            b2 = count_interval(self.fuel_cell_climb,
                                (self.current_hydrogen - self.min_hydrogen) * self.fuel_cell_factor,
                                self.current_fuel_cell_workspeed)

            corrected_action2 = max(-self.fuel_cell_climb, -self.current_fuel_cell_workspeed,
                                    min(b2, action[2] * self.fuel_cell_climb,
                                        self.fuel_cell_worklimit - self.current_fuel_cell_workspeed)) / self.fuel_cell_climb
            current_fuel_cell_speed = corrected_action2 * self.fuel_cell_climb + self.current_fuel_cell_workspeed
            self.current_fuel_cell_workspeed = round(current_fuel_cell_speed, 4)

            if round(current_fuel_cell_speed, 4) == 0 and save_a1_sign == True:
                action[1] = save_a1
                buygas = action[1] * self.buy_gas_limit  # 计算动作对应的能量变化，action_battery是一个在[-1, 1]范围内的值
                updated_hydrogen3 = max(self.min_hydrogen,
                                        min(self.max_hydrogen, self.current_hydrogen + buygas))  # 计算能量变化量  不是soc
                updated_hydrogen_capacity = updated_hydrogen3 - self.current_hydrogen  # 计算能量变化量  不是soc
                corrected_action1 = updated_hydrogen_capacity / self.buy_gas_limit

            updated_hydrogen_capacity = self.current_hydrogen - self.current_fuel_cell_workspeed / self.fuel_cell_factor + corrected_action1 * self.buy_gas_limit
            self.current_hydrogen = round(updated_hydrogen_capacity, 4)

        # action[3] 是发电机动作
        corrected_diesel_engine_action = max(0, min(self.diesel_engine_worklimit, action[
            3] * self.diesel_engine_climb + self.current_diesel_engine_workspeed))
        corrected_action3 = (
                                    corrected_diesel_engine_action - self.current_diesel_engine_workspeed) / self.diesel_engine_climb
        self.current_diesel_engine_workspeed = round(corrected_diesel_engine_action, 4)
        state_sum = self.pv_generation[time] *self.modulus[0] + self.wt_generation[time] *self.modulus[1] - self.load[time] *self.modulus[2]
        if self.energy_change >= 0:
            unbalance = (self.pv_generation[time] *self.modulus[0] + self.wt_generation[time] *self.modulus[1] - self.load[time] *self.modulus[2] + self.NODE5[time])/2 + self.current_diesel_engine_workspeed + self.current_fuel_cell_workspeed - self.energy_change / 0.95
        else:
            unbalance = (self.pv_generation[time] *self.modulus[0] + self.wt_generation[time] *self.modulus[1] - self.load[time] *self.modulus[2] + self.NODE5[time])/2 + self.current_diesel_engine_workspeed + self.current_fuel_cell_workspeed - self.energy_change * 0.95

        reward5 = (- (0.01 * self.current_fuel_cell_workspeed + 0.00008 * self.current_fuel_cell_workspeed * self.current_fuel_cell_workspeed)
                   - (0.08 * self.current_diesel_engine_workspeed + 0.002 * self.current_diesel_engine_workspeed * self.current_diesel_engine_workspeed)
                   - corrected_action1 * self.buy_gas_limit * 0.25 - 0.0316 * (0.76572 * self.current_diesel_engine_workspeed + 0.202 * corrected_action1 * self.buy_gas_limit))

        # 电池、电解水改变、燃料电池改变
        new_action.append(corrected_action0)
        new_action.append(corrected_action1)
        new_action.append(corrected_action2)
        new_action.append(corrected_action3)

        # 时间，净负荷，电池soc，氢气容量，电解水工作速度，燃料电池工作速度
        next_state.append(round(((time + 1) % 24) / 23,4))
        next_state.append((self.pv_generation[(time + 1) % 24] * self.modulus[0] + self.wt_generation[(time + 1) % 24] * self.modulus[1] - self.load[(time + 1) % 24] * self.modulus[2] + self.NODE5[(time + 1) % 24])/500)
        next_state.append(self.current_capacity / self.capacity)
        next_state.append(self.current_hydrogen / self.hydrogen_capacity)
        next_state.append(self.current_fuel_cell_workspeed / self.fuel_cell_worklimit)
        next_state.append(self.current_diesel_engine_workspeed / self.diesel_engine_worklimit)
        next_state.append(self.sell_price[(time + 1) % 24])
        next_state.append(self.buy_price[(time + 1) % 24])
        return new_action, next_state, unbalance, reward5,state_sum



    def reset(self,pv_data,wt_data,load_data):  # 重置电池状态
        self.energy_change = 0  # 能量变化量初始化为0
        self.current_capacity = 50  # 初始化电池状态
        # 发电机
        self.current_diesel_engine_workspeed = 0  # 初始化发电机状态
        # 储气罐
        self.current_hydrogen = 150  # 储氢罐起始
        # 燃料电池
        self.current_fuel_cell_workspeed = 0  # 初始化燃料电池
        self.pv_generation = pv_data  # 光伏发电数据引用，类型为列表
        self.wt_generation = wt_data
        self.load = load_data

    def get_state(self, time):
        state = []
        state.append(round(time / 23,4))
        state.append((self.pv_generation[time % 24] * self.modulus[0] + self.wt_generation[time % 24] * self.modulus[1] - self.load[time  % 24] * self.modulus[2] + self.NODE5[time % 24])/500)
        state.append(self.current_capacity / self.capacity)
        state.append(self.current_hydrogen / self.hydrogen_capacity)
        state.append(self.current_fuel_cell_workspeed / self.fuel_cell_worklimit)
        state.append(self.current_diesel_engine_workspeed / self.diesel_engine_worklimit)
        state.append(self.sell_price[time % 24])
        state.append(self.buy_price[time % 24])

        return state


def count_interval(a, y ,x):
    n = x // a
    if n == 0:
        if y <= a:
            b = y - x
        elif a < y <= 2 * x + a:
            b = (y + a) / 2 - x
        else:
            b = a
    else:
        if x % a == 0:
            if (n + 1) * n * a / 2 >= y >= n * (n-1) * a / 2:
                b = (y - (n - 1) * n *a / 2) / n -a
                # 区间就是【-a，b】
            elif (n+1)*(n+2)*a/2 >= y > (n+1)*n*a/2:
                b = (y - (n+1)*n*a/2)/(n+1)
                # 区间就是【-a，b】
            else:
                b = a
        else:
            if n*(n+1)*a/2 <= y <= (n+1)*(n+2)*a/2:
                b = n*a + (y-n*(n+1)*a/2)/(n+1) - x
            elif n * (n+1)*a/2 > y >= (n-1)*n*a/2 + n*(x-n*a)-0.01:
                b = (y -(n-1)*n*a/2)/n + (n-1)*a - x
            elif (n+1)*(n+2)*a/2 < y <= (x+a-a*(n+1)/2)*(n+2):
                b = (y - (n+1)*(n+2)*a/2)/(n+2) + (n+1)*a -x
            else:
                b = a
    return b



class ESSEnv(gym.Env):
    def __init__(self, **kwargs):
        super(ESSEnv, self).__init__()
        self.data_manager = DataManager()  # 初始化数据管理器
        self.current_day = 0  # 当前天数
        self._load_day_data()   # 加载每天的光伏发电和电力消耗数据

        self.episode_length = kwargs.get('episode_length', 24)  # 设置每集长度，默认为24小时

        self.current_time = None  # 当前时间步
        self.buy_price = [0.3578, 0.3578, 0.3578, 0.3578, 0.3578, 0.3578,
                          0.3578, 0.3578, 0.8325, 0.8325, 0.8325, 0.3578,
                          0.3578, 0.8325, 0.8325, 0.8325, 0.8325, 0.8325,
                          0.8325, 1.2109, 1.2109, 0.8325, 0.3578, 0.3578, 0.3578]
        self.sell_price = [0.2, 0.2, 0.2, 0.2, 0.2, 0.2,
                           0.2, 0.2, 0.4125, 0.4125, 0.4125, 0.2,
                           0.2, 0.4125, 0.4125, 0.4125, 0.4125, 0.4125,
                           0.4125, 0.2, 0.2, 0.2, 0.2, 0.2]

        # 假设每个节点都需要整个数据数组，这里需要调整以匹配具体的实现需求
        pv_data = self.data_manager.PV_Generation[self.current_day]  # 示例取第一天的数据
        wt_data = self.data_manager.WT_Generation[self.current_day]
        load_data = self.data_manager.Electricity_Consumption[self.current_day]
        hot_data = self.data_manager.HOT_Consumption[self.current_day]

        # 创建节点实例
        self.node1 = Node1("node1", pv_data, wt_data, load_data)
        self.node2 = Node2("node2", pv_data, wt_data, load_data)
        self.node3 = Node3("node3", pv_data, wt_data, load_data, hot_data)  # 假设Node3需要热负载，这里简单复用电负载
        self.node4 = Node4("node4", pv_data, wt_data, load_data)
        self.node5 = Node5("node5", pv_data, wt_data, load_data)

    def reset(self, is_test):
        # self.current_day = np.random.randint(0, len(self.data_manager.PV_Generation))
        if not is_test:
            self.current_day = (self.current_day) % 600
        self.current_time = 0

        pv_data = self.data_manager.PV_Generation[self.current_day]
        wt_data = self.data_manager.WT_Generation[self.current_day]
        load_data = self.data_manager.Electricity_Consumption[self.current_day]
        hot_data = self.data_manager.HOT_Consumption[self.current_day]

        # 重置节点状态
        self.node1.reset(pv_data,wt_data,load_data)
        self.node2.reset(pv_data,wt_data,load_data)
        self.node3.reset(pv_data,wt_data,load_data,hot_data)
        self.node4.reset(pv_data,wt_data,load_data)
        self.node5.reset(pv_data,wt_data,load_data)

        state = []
        state.append(self.node1.get_state(self.current_time))
        state.append(self.node2.get_state(self.current_time))
        state.append(self.node3.get_state(self.current_time))
        state.append(self.node4.get_state(self.current_time))
        state.append(self.node5.get_state(self.current_time))
        #
        return state

    def step(self, actions):
        # 构建当前状态
        timenow = self.current_time
        r = []
        next_states = []
        new_actions = []
        rewards = []

        new_action1, next_state1, unbalance1, reward1,s1 = self.node1.step(actions[0], timenow)
        new_action2, next_state2, unbalance2, reward2,s2 = self.node2.step(actions[1], timenow)
        new_action3, next_state3, unbalance3, reward3,s3 = self.node3.step(actions[2], timenow)
        new_action4, next_state4, unbalance4, reward4,s4 = self.node4.step(actions[3], timenow)
        new_action5, next_state5, unbalance5, reward5,s5 = self.node5.step(actions[4], timenow)

        next_states.append(next_state1)
        next_states.append(next_state2)
        next_states.append(next_state3)
        next_states.append(next_state4)
        next_states.append(next_state5)

        new_actions.append(new_action1)
        new_actions.append(new_action2)
        new_actions.append(new_action3)
        new_actions.append(new_action4)
        new_actions.append(new_action5)

        un_reward = ex_gurobi(unbalance1, unbalance2, unbalance3, unbalance4, unbalance5, self.sell_price[timenow],
                              self.buy_price[timenow])


        rewards = [(reward1 + un_reward) / 100,  # + comp_r
                  (reward2 ) / 100,
                  (reward3 ) / 100,
                   (reward4 ) / 100,
                  (reward5 ) / 100]

        # 时间步递增
        self.current_time += 1
        done = self.current_time == self.episode_length  # 检查是否达到一个episode的长度

        # 如果episode结束，重置环境到下一个天
        if done:
            self.current_day = (self.current_day + 1) % len(self.data_manager.PV_Generation)

        dones = [done] * 5  # 所有节点的done标志

        return next_states, rewards, dones, new_actions

    def get_reward_detail(self):
        reward_details = {
            'excess_penalty': [],
            'deficient_penalty': [],
            'sell_benefit': [],
            'buy_cost': []
        }
        for node in self.nodes:
            reward_details['excess_penalty'].append(node.excess_penalty)
            reward_details['deficient_penalty'].append(node.deficient_penalty)
            reward_details['sell_benefit'].append(node.sell_benefit)
            reward_details['buy_cost'].append(node.buy_cost)

        return reward_details

    def render(self, current_obs, next_obs, reward, done):
        print('day={}, hour={:2d}, state={}, next_state={}, reward={:.4f}, terminal={}\n'.format(
            self.current_day, self.current_time, current_obs, next_obs, reward, done))

    def _load_day_data(self):
        pv_df = pd.read_excel('DATA/PV.xlsx')
        wt_df = pd.read_excel('DATA/WT.xlsx')
        electricity_df = pd.read_excel('DATA/Load.xlsx')
        hot_df = pd.read_excel('DATA/H_Load.xlsx')
        for day in range(650):# len(pv_df)
            pv_data = pv_df.iloc[day].to_numpy(dtype=float).flatten()
            wt_data = wt_df.iloc[day].to_numpy(dtype=float).flatten()
            electricity_data = electricity_df.iloc[day].to_numpy(dtype=float).flatten()
            hot_data = hot_df.iloc[day].to_numpy(dtype=float).flatten()
            self.data_manager.add_day_data(pv_data, electricity_data, wt_data, hot_data)


