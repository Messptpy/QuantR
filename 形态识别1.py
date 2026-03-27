import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
from scipy.ndimage import gaussian_filter1d
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ==============================================================================
# 1. 核回归
# ==============================================================================
def gaussian_kernel(x, h):
    return np.exp(-0.5 * (x / h) ** 2) / (h * np.sqrt(2 * np.pi))

def kernel_regression(y, h, window_size=35):
    n = len(y)
    smooth = np.zeros(n)
    for t in range(n):
        start = max(0, t - window_size)
        end = t
        incides = np.arange(start, end + 1)
        distances = t - incides
        weights = gaussian_kernel(distances, h)
        smooth[t] = np.sum(weights * y[incides]) / np.sum(weights)
    return smooth

def optimal_bandwidth(y):
    def loocv_mse(h):
        n = len(y)
        mse = 0
        for i in range(n):
            y_train = np.delete(y, i)
            u = (np.arange(n-1) - i) / h
            weights = gaussian_kernel(u, 1)
            weights /= np.sum(weights)
            y_pred = np.sum(weights * y_train)
            mse += (y[i] - y_pred) ** 2
        return mse / n
    res = minimize_scalar(loocv_mse, bounds=(0.1, 10), method='bounded')
    return res.x * 0.3

# ==============================================================================
# 2. 极值点
# ==============================================================================
def get_alternating_extrema(smooth_price, raw_price, min_sep=2):
    dx = np.gradient(smooth_price)
    dx_smooth = gaussian_filter1d(dx, sigma=1)
    sign_change = np.where(np.diff(np.sign(dx_smooth)) != 0)[0]
    if len(sign_change) < 5:
        return np.array([]), np.array([]), np.array([])

    extrema_pos = []
    extrema_price = []
    extrema_type = []
    prev_type = None

    for pos in sign_change:
        if dx_smooth[pos] > 0 and dx_smooth[pos+1] < 0:
            curr_type = 'max'
        elif dx_smooth[pos] < 0 and dx_smooth[pos+1] > 0:
            curr_type = 'min'
        else:
            continue

        if prev_type is None or (curr_type != prev_type and pos - extrema_pos[-1] >= min_sep):
            extrema_pos.append(pos)
            extrema_price.append(raw_price[pos])
            extrema_type.append(curr_type)
            prev_type = curr_type
    return np.array(extrema_pos), np.array(extrema_price), np.array(extrema_type)

# ==============================================================================
# 3. 形态识别
# ==============================================================================
def detect_patterns(extrema_pos, extrema_price, extrema_type):
    patterns = []
    n_extrema = len(extrema_pos)
    if n_extrema < 3:
        return patterns

    tol_hs = 0.015
    tol_other = 0.0075
    used = set()  # 防止重复识别

    # --------------------------
    # 1. 5点形态：头肩、矩形（只识别一次）
    # --------------------------
    if n_extrema >= 5:
        for i in range(n_extrema - 4):
            if i in used:
                continue
                
            pos = extrema_pos[i:i+5]
            p = extrema_price[i:i+5]
            typ = extrema_type[i:i+5]
            if pos[-1] - pos[0] > 35:
                continue

            p_mean_04 = (p[0] + p[4]) / 2 
            p_mean_13 = (p[1] + p[3]) / 2 
            p_mean_top = (p[0] + p[2] + p[4]) / 3 
            p_mean_bot = (p[1] + p[3]) / 2 

            # 头肩顶
            if (typ == ['max','min','max','min','max']).all():
                if (p[2] > p[0] and p[2] > p[4] and
                    abs(p[0]-p[4]) / (p_mean_04*2) < tol_hs and  # |E1-均值|/均值 < 1.5% 化简得
                    abs(p[1]-p[3]) / (p_mean_13*2) < tol_hs):
                    patterns.append(('HS', pos))
                    used.add(i)    # 标记已使用的起点索引，后续极值仍可构成形态
                    continue

            # 头肩底
            if (typ == ['min','max','min','max','min']).all():
                if (p[2] < p[0] and p[2] < p[4] and
                    abs(p[0]-p[4])/(p_mean_04*2) < tol_hs and
                    abs(p[1]-p[3])/(p_mean_13*2) < tol_hs):
                    patterns.append(('IHS', pos))
                    used.add(i)
                    continue

            # 矩形顶
            if (typ == ['max','min','max','min','max']).all():
                t1 = abs(p[0]-p_mean_top)/p_mean_top < tol_other
                t2 = abs(p[2]-p_mean_top)/p_mean_top < tol_other
                t3 = abs(p[4]-p_mean_top)/p_mean_top < tol_other
                b1 = abs(p[1]-p_mean_bot)/p_mean_bot < tol_other
                b2 = abs(p[3]-p_mean_bot)/p_mean_bot < tol_other
                sep = np.min([p[0],p[2],p[4]]) > np.max([p[1],p[3]])
                if t1 and t2 and t3 and b1 and b2 and sep:
                    patterns.append(('RTOP', pos))
                    used.add(i)
                    continue

            # 矩形底
            if (typ == ['min','max','min','max','min']).all():
                b1 = abs(p[0]-p_mean_bot)/p_mean_bot < tol_other
                b2 = abs(p[2]-p_mean_bot)/p_mean_bot < tol_other
                b3 = abs(p[4]-p_mean_bot)/p_mean_bot < tol_other
                t1 = abs(p[1]-p_mean_top)/p_mean_top < tol_other
                t2 = abs(p[3]-p_mean_top)/p_mean_top < tol_other
                sep = np.max([p[1],p[3]]) > np.min([p[0],p[2],p[4]])
                if b1 and b2 and b3 and t1 and t2 and sep:
                    patterns.append(('RBOT', pos))
                    used.add(i)
                    continue

    # --------------------------
    # 2. 双顶 / 双底（3点形态）
    # --------------------------
    if n_extrema >= 3:
        for i in range(n_extrema - 2):
            if i in used:
                continue

            pos = extrema_pos[i:i+3]
            p = extrema_price[i:i+3]
            typ = extrema_type[i:i+3]
        
            # 双顶：
            if (typ == ['max', 'min', 'max']).all():
                mean_p = (p[0] + p[2]) / 2  # 两个峰的均值
                cond1 = abs(p[0] - mean_p) / mean_p < 0.0075
                cond2 = abs(p[2] - mean_p) / mean_p < 0.0075
                if pos[2] - pos[0] > 22 and cond1 and cond2:
                    patterns.append(('DTOP', pos))
                    used.add(i) 
                    


            # 双底：
            if (typ == ['min', 'max', 'min']).all():
                mean_p = (p[0] + p[2]) / 2 
                # 论文公式：每个点都在均值 ±0.75% 内
                cond1 = abs(p[0] - mean_p) / mean_p < 0.0075
                cond2 = abs(p[2] - mean_p) / mean_p < 0.0075
                if pos[2] - pos[0] > 22 and cond1 and cond2:
                    patterns.append(('DBOT', pos))
                    used.add(i)                      

    return patterns
# ==============================================================================
# 4. 回测
# ==============================================================================
def backtest(price, signals, sig_map, hold=5):
    logret = np.log(pd.Series(price)/pd.Series(price).shift(1)).fillna(0).values
    n = len(price)
    pos = np.zeros(n)
    stats = {}

    for t in range(n):
        if signals[t]==0 or t+hold>=n : 
            continue
    # 累计持有 hold 天的收益
        ret = np.sum(logret[t+1:t+1+hold])
    # 做空信号
        if signals[t]==-1: 
            ret = -ret
    # 按形态分类保存收益
        pat = sig_map[t]
        if pat not in stats: 
            stats[pat] = []
        stats[pat].append(ret)
    # 多头=1, 空头=-1
        if signals[t] == 1:   
            pos[t : t+hold] = 1
        elif signals[t] == -1:
            pos[t : t+hold] = -1

    res = {
        p:{
            'count':len(r),
            'win':np.mean(np.array(r)>0),
            'ret':np.mean(r)
        } 
        for p,r in stats.items()
    }
    cost = np.where(np.diff(pos, prepend=0) !=0, 0.0005, 0)
    return np.exp(np.cumsum(pos*logret-cost)), np.exp(np.cumsum(logret)), res

# ==============================================================================
# 5. 主程序（35日窗口 + 3天滞后 + 无未来函数）
# ==============================================================================
if __name__ == "__main__":
    data = pd.read_excel(r"C:\Users\LEGION\Desktop\NYSE NASDAQ.xlsx", skiprows=5, parse_dates=['Date'], index_col='Date')
    price = data['close.1'].values     #close是纳斯达克指数，close.1是标普500指数
    dates = data.index
    n_total = len(price)

    WINDOW = 35
    LAG_DAYS = 3
    HOLD_DAYS = 3    #持有日期可改

    signals = np.zeros(n_total)
    sig_map = {}
    triggered = set()  # 全局去重：记录已经发过信号的K线位置

    print("识别形态中...")
    for i in range(WINDOW + LAG_DAYS, n_total):
        current_idx = i - LAG_DAYS
        wp = price[current_idx - WINDOW : current_idx]

        h = optimal_bandwidth(wp)
        sp = kernel_regression(wp, h)
        ep, px, et = get_alternating_extrema(sp, wp)
        pats = detect_patterns(ep, px, et)

        for typ, p5_pos in pats:
            # 真正发信号的K线位置
            signal_day = current_idx - WINDOW +p5_pos[-1] + 1

            # --------------------------
            # 这个位置已经发过信号 跳过
            # --------------------------
            if signal_day in triggered:
                continue

            # 底部形态  做多
            if typ in ['IHS','DBOT','RBOT']:
                signals[signal_day] = 1
                sig_map[signal_day] = typ
                triggered.add(signal_day)  

            # 顶部形态  做空
            if typ in ['HS','DTOP','RTOP']:
                signals[signal_day] = -1
                sig_map[signal_day] = typ
                triggered.add(signal_day)  

    # 回测
    cum_strat, cum_bh, stats = backtest(price, signals, sig_map, HOLD_DAYS)
    # ===================== 输出结果 =====================
    print("\n" + "="*70)
    print("          形态统计结果（持有" + str(HOLD_DAYS) + "天）      ")
    print("="*70)
    print(f"{'形态':<10}{'出现次数':<10}{'胜率':<10}{'平均收益':<12}")
    print("-"*70)
    for pat in ['IHS','HS','DBOT','DTOP','RBOT','RTOP']:
        v = stats.get(pat, {'count':0,'win':0,'ret':0})
        print(f"{pat:<10}{v['count']:<10}{v['win']:<10.1%}{v['ret']:<12.2%}")

    print("\n【最终收益】")
    print(f"策略收益：{cum_strat[-1]-1:.2%}")
    print(f"买入持有：{cum_bh[-1]-1:.2%}")

    # ===================== 绘图 =====================
    plt.figure(figsize=(16,10))
    plt.subplot(221)
    plt.plot(dates, price, c='gray', lw=1.2, label='Close')
    plt.scatter(dates[signals==1], price[signals==1], c='r', s=70, label='Buy(IHS/DBOT/RBOT)')
    plt.scatter(dates[signals==-1], price[signals==-1], c='g', s=70, label='Sell(HS/DTOP/RTOP)')
    plt.title('Price & Signals (HS/IHS as Main)', fontsize=14)
    plt.legend()
    plt.grid(alpha=0.3)

    plt.subplot(222)
    plt.plot(dates, cum_strat, c='r', lw=2.5, label='Strategy')
    plt.plot(dates, cum_bh, c='b', lw=1.5, label='Buy&Hold')
    plt.title('Return Curve')
    plt.legend()
    plt.grid(alpha=0.3)

    plt.subplot(223)
    names = list(stats.keys())
    plt.bar(names, [stats[k]['count'] for k in names], color='orange')
    plt.title('Pattern Count')

    plt.subplot(224)
    plt.bar(names, [stats[k]['ret'] for k in names], color='green')
    plt.title('Return')

    plt.tight_layout()
    plt.show()