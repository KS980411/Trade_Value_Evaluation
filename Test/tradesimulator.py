import pandas as pd
import os


os.chdir('/Users/choeunsol/Python/PAINS 방중 프로젝트/최종 분석 데이터셋/모델 예측치')

kbo_g = pd.read_excel('kbo major level trade value.xlsx')
kbo_ng = pd.read_excel('kbo minor level trade value.xlsx')

# 시뮬레이터

def trade_simulator(data, player_1, player_2):
    player1_value = 0
    player2_value = 0
    for i in range(len(player_1)):
        value_one = data[data['name'] == player_1[i]]['trade_value'].values
        player1_value += value_one
    for j in range(len(player_2)):
        value_two = data[data['name'] == player_2[j]]['trade_value'].values
        player2_value += value_two
    
    if (abs(player1_value - player2_value) <= 3):
        print_script = "트레이드 성사!"
    elif (abs(player1_value - player2_value) <= 7):
        print_script = "음.. 고려할 만은 하겠군요."
    else:
        print_script = "이 카드로는 받을 수 없습니다.."
    
    player1 = player1_value
    player2 = player2_value
    return player1, player2, print_script