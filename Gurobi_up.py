from gurobipy import Model, GRB, quicksum


def ex_gurobi(node1, node2, node3, node4, node5, sell_price, buy_price):
    # 创建模型
    model = Model('microgrid_optimization')
    model.setParam('OutputFlag', 0)
    # 12,13,14,15,23,24,25,34,35,45
    LOSS = [6 * 0.005, 8 * 0.005, 13 * 0.005, 5 * 0.005, 10 * 0.005,
            5 * 0.005, 7 * 0.005, 7 * 0.006, 12 * 0.005, 4 * 0.005]

    # 定义节点1的变量
    P12 = model.addVar(name='P12', lb=0, ub=2000)
    P13 = model.addVar(name='P13', lb=0, ub=2000)
    P14 = model.addVar(name='P14', lb=0, ub=2000)
    P15 = model.addVar(name='P15', lb=0, ub=2000)
    P1buy = model.addVar(name='P1buy', lb=0, ub=2000)
    P1sell = model.addVar(name='P1sell', lb=0, ub=2000)

    P21 = model.addVar(name='P21', lb=0, ub=2000)
    P23 = model.addVar(name='P23', lb=0, ub=2000)
    P24 = model.addVar(name='P24', lb=0, ub=2000)
    P25 = model.addVar(name='P25', lb=0, ub=2000)
    P2buy = model.addVar(name='P2buy', lb=0, ub=2000)
    P2sell = model.addVar(name='P2sell', lb=0, ub=2000)

    P31 = model.addVar(name='P31', lb=0, ub=2000)
    P32 = model.addVar(name='P32', lb=0, ub=2000)
    P34 = model.addVar(name='P34', lb=0, ub=2000)
    P35 = model.addVar(name='P35', lb=0, ub=2000)
    P3buy = model.addVar(name='P3buy', lb=0, ub=2000)
    P3sell = model.addVar(name='P3sell', lb=0, ub=2000)

    P41 = model.addVar(name='P41', lb=0, ub=2000)
    P42 = model.addVar(name='P42', lb=0, ub=2000)
    P43 = model.addVar(name='P43', lb=0, ub=2000)
    P45 = model.addVar(name='P45', lb=0, ub=2000)
    P4buy = model.addVar(name='P4buy', lb=0, ub=2000)
    P4sell = model.addVar(name='P4sell', lb=0, ub=2000)

    P51 = model.addVar(name='P51', lb=0, ub=2000)
    P52 = model.addVar(name='P52', lb=0, ub=2000)
    P53 = model.addVar(name='P53', lb=0, ub=2000)
    P54 = model.addVar(name='P54', lb=0, ub=2000)
    P5buy = model.addVar(name='P5buy', lb=0, ub=2000)
    P5sell = model.addVar(name='P5sell', lb=0, ub=2000)

    # 12,13,14,15,23,24,25,34,35,45
    model.addConstr(P1sell - P1buy == node1 - P12 - P13 - P14 - P15 + (1 - LOSS[0]) * P21 + (1 - LOSS[1]) * P31 + (
                1 - LOSS[2]) * P41 + (1 - LOSS[3]) * P51)
    model.addConstr((P1sell + P12 + P13 + P14 + P15) * (P1buy + P21 + P31 + P41 + P51) == 0)

    model.addConstr(P2sell - P2buy == node2 - P21 - P23 - P24 - P25 + (1 - LOSS[0]) * P12 + (1 - LOSS[4]) * P32 + (
                1 - LOSS[5]) * P42 + (1 - LOSS[6]) * P52)
    model.addConstr((P2sell + P21 + P23 + P24 + P25) * (P2buy + P12 + P32 + P42 + P52) == 0)

    model.addConstr(P3sell - P3buy == node3 - P31 - P32 - P34 - P35 + (1 - LOSS[1]) * P13 + (1 - LOSS[4]) * P23 + (
                1 - LOSS[7]) * P43 + (1 - LOSS[8]) * P53)
    model.addConstr((P3sell + P31 + P32 + P34 + P35) * (P3buy + P13 + P23 + P43 + P53) == 0)

    model.addConstr(P4sell - P4buy == node4 - P41 - P42 - P43 - P45 + (1 - LOSS[2]) * P14 + (1 - LOSS[5]) * P24 + (
                1 - LOSS[7]) * P34 + (1 - LOSS[9]) * P54)
    model.addConstr((P4sell + P41 + P42 + P43 + P45) * (P4buy + P14 + P24 + P34 + P54) == 0)

    model.addConstr(P5sell - P5buy == node5 - P51 - P52 - P53 - P54 + (1 - LOSS[3]) * P15 + (1 - LOSS[6]) * P25 + (
                1 - LOSS[8]) * P35 + (1 - LOSS[9]) * P45)
    model.addConstr((P5sell + P51 + P52 + P53 + P54) * (P5buy + P15 + P25 + P35 + P45) == 0)

    reward1 = ((P1sell + P2sell + P3sell + P4sell + P5sell) * sell_price - (P1buy + P2buy + P3buy + P4buy + P5buy) * (buy_price + 0.0316 * 0.683))

    model.setObjective(reward1, GRB.MAXIMIZE)

    model.optimize()

    temp = model.objVal
    del model
    return temp




