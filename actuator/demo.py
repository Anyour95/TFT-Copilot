import json
from Keyboard_op import *
from Mouse_op import *
from Action_op import *
def parse_operation(ops_json_PATH,action):
    with open(ops_json_PATH,'r',encoding='utf-8') as ops_json:
        ops = json.load(ops_json)
    game_round = ops.get("game_round")
    operations = ops.get("operations", {})
    
    # 检查并执行每个操作
    if operations.get("refresh_shop") is not None:
        # 执行刷新商店
        print(f'DO:refresh_shop!\n')

    if operations.get("buy_units"):
        for buy in operations["buy_units"]:
            shop_idx = buy["shop_index"]
            print(f'DO:buy_units at idx={shop_idx}!\n')
            # 模拟鼠标操作
            action.mouse.move_to(x_percent=0.5,y_percent=0.5)
            action.mouse.click(button='left',clicks=2,interval=0.1)
            # 点击商店对应位置
            # ...
    if operations.get("sell_units"):
        for sell in operations["sell_units"]:
            location = sell["location"]
            idx = sell["index"]
            print(f'DO:sell_units at {location} idx={idx}!\n')
            # 模拟鼠标操作
            action.mouse.move_to(x_percent=0.3*idx,y_percent=0.3)
            action.mouse.click(button='left',clicks=2,interval=0.1)
            # 点击商店对应位置
            # ...
    # 其他操作类似...
    return True

def main():
    PATH = 'operation_protocol/game_op.json'
    keyboard = KeyboardController()
    mouse = MouseController()
    action = GameAction(mouse,keyboard)
    STATE = parse_operation(PATH,action)

if __name__ == "__main__":
    main()