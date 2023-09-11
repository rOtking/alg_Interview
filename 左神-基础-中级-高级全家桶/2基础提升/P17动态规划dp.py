''''''
# ---------------------------- 1.之前的抽牌问题 ------------------ #
'''
先后手抽牌问题

递归
'''
def getMax(arr):
    # l,r上先手
    def f1(arr, l, r):
        if l == r:
            return arr[l]
        # todo 很巧妙！
        return max(arr[l] + f2(arr, l + 1, r), arr[r] + f2(arr, l, r - 1))


    # 后手
    def f2(arr, l, r):
        if l == r:
            return 0

        # todo 让接下来别人的先手最大值 最小
        return min(f1(arr, l + 1, r), f1(arr, l, r - 1))

    return max(f1(arr, 0, len(arr) - 1), f2(arr, 0, len(arr) - 1))

'''
memo1与mome2两个缓存即可，不赘述。

            DP改"范围上尝试的递归"
            
（1）f1的参数范围都是0～N-1，正方形；f2也是正方形。一共两张表dp1 dp2
（2）所求 max(dp1[0][N-1], dp2[0][N-1])，右上角的值。
（3）范围尝试的隐藏条件：范围尝试的左边界一定小于等于右边界，所以正方的左下半区是没用的！（特征）
（4）basecase l==r时
（5）普遍，dp1依赖在dp2对称点的左边与下面的值，dp2一样
（6）顺序：交替利用对角线，或者从下向上、从左到右

范围尝试往往先填对角线
'''
def gameDp(arr):
    dp1 = [[0] * len(arr) for _ in range(len(arr))]
    dp2 = [[0] * len(arr) for _ in range(len(arr))]

    # 初始化
    for i in range(len(arr)):
        dp1[i][i] = arr[i]
        dp2[i][i] = 0

    # 从下向上，从左到右
    for i in range(len(arr) - 2, -1, -1):
        for j in range(i + 1, len(arr[0])):
            dp1[i][j] = max(arr[i] + dp2[i + 1][j], arr[j] + dp2[i][j - 1])
            dp2[i][j] = min(dp1[i + 1][j], dp1[i][j - 1])

    return max(dp1[0][len(arr) - 1], dp2[0][len(arr) - 1])



# ---------------------------- 2.高维dp，象棋问题 ------------------ #
'''
10*9的棋盘，马在00位置，走k步到(a,b)的方法数。

process(x,y,step)    从00出发step到xy位置的方法数。
       
       . . x . x . . .
       . x . . . x . .
       . . . o . . . .
       . x . . . x . .
       . . x . x . . .
共8个位置一步到.
'''

def ways(a, b, k):
    def process(x, y, step):
        if x < 0 or x > 8 or y < 0 or y > 9:
            return 0

        # 留在原地是1种，否则没有
        if step == 0:
            return 1 if x == 0 and y == 0 else 0

        return process(x - 1, y - 2, step - 1) + \
               process(x - 2, y - 1, step - 1) + \
               process(x - 2, y + 1, step - 1) + \
               process(x - 1, y + 2, step - 1) + \
               process(x + 1, y + 2, step - 1) + \
               process(x + 2, y + 1, step - 1) + \
               process(x + 2, y - 1, step - 1) + \
               process(x + 1, y - 2, step - 1)
    return process(a, b, k)

'''
            dp
(1)三个变量是个立体表x:[0,8],y[0,9],step[0,k]
(2)目标dp[a][b][k]
(3)初始立方体之外都是0，step=0的底面有
(4)每层只依赖下一层
'''
def waysDp(a, b, k):
    dp = [[[0] * (k + 1) for _ in range(10)] for _ in range(9)]

    dp[0][0][0] = 1 # 初始

    for h in range(1, k + 1):     # 高
        for y in range(10):
            for x in range(9):
                dp[x][y][h] += getValue(dp, x - 1, y - 2, h - 1)
                dp[x][y][h] += getValue(dp, x - 2, y - 1, h - 1)
                dp[x][y][h] += getValue(dp, x - 2, y + 1, h - 1)
                dp[x][y][h] += getValue(dp, x - 1, y + 2, h - 1)
                dp[x][y][h] += getValue(dp, x + 1, y + 2, h - 1)
                dp[x][y][h] += getValue(dp, x + 2, y + 1, h - 1)
                dp[x][y][h] += getValue(dp, x + 2, y - 1, h - 1)
                dp[x][y][h] += getValue(dp, x + 1, y - 2, h - 1)

    return dp[a][b][k]


# 控制输出边界
def getValue(dp, x, y, h):
    if x < 0 or x > 8 or y < 0 or y > 9:
        return 0
    return dp[x][y][h]


# ---------------------------- 3.高维dp，生存概率问题 ------------------ #
'''
问题：N*M的格子，Bob在初始位置(a,b)，每次可向上下左右4个方向任走一步，越界就死，问走k步生存的概率？

p = 活下来的方法数 / 总方法

总方法 = 4^k


process(N, M, row, col, k)：当前在row，col，走k步的方法
'''

def bobWays(N, M, a, b, k):
    def process(N, M, row, col, k):
        if row < 0 or row > N or col < 0 or col > M:
            return 0

        # 走完时没越界，就找到一种
        if k == 0:
            return 1

        return process(N, M, row - 1, col, k - 1) + \
               process(N, M, row, col - 1, k - 1) + \
               process(N, M, row + 1, col, k - 1) + \
               process(N, M, row, col + 1, k)
    return process(N, M, a, b, k)

'''
（1）可变参数row,col,k
（2）目标dp[a][b][k]
（3）初始：底面都是1，立方体外都是0
（4）依赖下面的值
（5）从下向上


类似，就不具体写了。
'''


# ---------------------------- 4.货币组成方法数 ------------------ #
'''
问题：arr元素代表coin面值，不重复，可以任意张，问组成aim的方法数？

背包问题的从左到右尝试，包含枚举。
'''
# todo 找到之后rest是否为0是很好的0-1终止条件


def getWays(arr, aim):

    # 自由使用arr[index...]所有面值，能搞定rest的方法数
    def process(arr, index, rest):
        # 一般basecase两种，有结果与无结果
        # todo 超
        if index == len(arr):
            return 1 if rest == 0 else 0

        i = 0
        res = 0
        while(arr[index] * i <= rest):
            res += process(arr, index + 1, rest - arr[index] * i)
            i += 1

        return res

    return process(arr, 0, aim)

'''
            改DP
（1）参数：2维，index[0,len(arr)],rest[0,aim]
（2）目标：dp[0][aim]
（3）初始：dp[len(arr)][:]
（4）普遍：依赖下一行的某些列
（5）顺序：从下到上，从左到右

'''

def getWaysDp(arr, aim):
    dp = [[0] * (aim + 1) for _ in range(len(arr) + 1)]       #  (N + 1) * aim

    dp[len(arr)][0] = 1

    for row in range(len(arr) - 1, -1, -1):
        for col in range(aim + 1):
            # 枚举
            i = 0
            while(arr[row] * i <= col):
                dp[row][col] += dp[row + 1][col - arr[row] * i]
                i += 1

    return dp[0][aim]



'''
                复杂度分析
                
表大小 O(N*aim)，但是计算每个格子存在枚举行为，最差的复杂度为O(aim)，就是面值为1的时候。总时间O(aim^2 * N)

三个for循环。


枚举行为没有必要！通过观察！！！！

设第i位置的面值为3  

  row    . . . . . . . . x . . ? . .
         . . d . . c . . b . . a          
    
dp[row][?] = dp[row][x] + a = dp[row][? - arr[row]] + dp[row + 1][?]

可有本行左边间隔arr[row]位置的值 + 下行本列的值 直接得到，不用重复枚举。太多重复值了！！


存在枚举行为的dp是可以优化的。注意不要越界，限制不能丢。

            这就是"斜率优化"！！！！！
            
    填表时有枚举行为，通过观察，看看邻接位置能不能替代枚举行为！！！！
'''
def getWaysDpPlus(arr, aim):
    dp = [[0] * (aim + 1) for _ in range(len(arr) + 1)]       #  (N + 1) * aim

    dp[len(arr)][0] = 1

    for row in range(len(arr) - 1, -1, -1):
        for col in range(aim + 1):
            # todo 注意不要越界，限制不能丢。
            # dp[row][col] = dp[row][row - arr[row]] + dp[row + 1][col]
            dp[row][col] = dp[row + 1][col]
            if row - arr[row] >= 0:
                dp[row][col] += dp[row][row - arr[row]]

    return dp[0][aim]


#--------------------------------------#
'''

            DP总结
1.尝试：从左到右，范围尝试，能搞定70%的题；
2.memo
3.推依赖关系，搭积木：严格表结构DP；-------->  4.精致DP

        评价尝试的好坏：
1.可变参数自己的变化范围，一般都是int，如果是list很复杂，一般不出现；2.参数个数，决定是几维表。

越少越好，表越小，复杂度越低。


'''





