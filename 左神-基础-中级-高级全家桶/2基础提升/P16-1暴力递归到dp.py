''''''

'''
整体思路：
    
    （1）尝试的暴力递归：最难最关键的部分，根据题意尝试 
             |
             v
    （2）记忆搜索的DP（貌似就是带备忘录的递归）
             |
             v
    （3）有严格表结构的dp
             |
             v
    （4）更精致优化的dp
        
（2）（3）都是可以不需要知道题意，就能从递归版本改写出来的。

（2）（3）多数复杂度是一样的，但（2）没有严格的位置依赖，在（3）的基础上是有可能写出更优化的dp的，从（2）是不能到（4）的。


'''
# --------------------------- 1.阿里的机器人运动问题 --------------#
'''
题意：1...N共N个位置，机器人初始在S位置[1,N]，终止位置E [1,N]，机器人每次可向左或向右走一步。
        要求从S经过k步到E，有多少走法。

    因为只能左或右，两种。
    
'''

def walkWays1(N, E, S, K):
    # 从cur位置到E 走剩余rest不，共多少走法。
    def process(N, E, cur, rest):
        # basecase
        if rest == 0:
            return 1 if cur == E else 0

        if cur == 1:
            return process(N, E, cur + 1, rest - 1)

        if cur == N:
            return process(N, E, cur - 1, rest - 1)

        return process(N, E, cur - 1, rest - 1) + process(N, E, cur + 1, rest - 1)

    return process(N, E, S, K)
'''
暴力姐：会重复算很多函数，前面用的时候去算，算晚就完了，后面有掉相同的参数重新算。

    可以存下来，下次遇到直接用，用空间换时间。
    
所以加缓存----->记忆化搜索

用arr与dict都行，用dict给自己降低难度，因为不用考虑arr的维度大小。

核心：就是在返回结果前，先存进dict；在调用递归前先检查存储。很无脑的。memo是个全局能访问的值，也可以作为一个参数，递归函数传递的时候的带着。

搞清楚可变参数都是谁，有几个，他就是你memo_arr的维度与范围，memo_dict的话可变参数的联合就是你的key!
'''
def walkWays2(N, E, S, K):
    # 从cur位置到E 走剩余rest不，共多少走法。
    def process(N, E, cur, rest):
        # todo 先查缓存
        if (cur, rest) in memo:
            return memo[(cur, rest)]
        # todo 所有分支，加缓存的方式都是一样的。所以可写到一个if else里 统一输出。
        #  不然全都是  return memo[(cur, rest)]   很麻烦
        # basecase
        if rest == 0:
            memo[(cur, rest)] =  1 if cur == E else 0
        elif cur == 1:
            memo[(cur, rest)] =  process(N, E, cur + 1, rest - 1)
        elif cur == N:
            memo[(cur, rest)] = process(N, E, cur - 1, rest - 1)
        else:
            memo[(cur, rest)] = process(N, E, cur - 1, rest - 1) + process(N, E, cur + 1, rest - 1)
        return memo[(cur, rest)]
    # 傻缓存
    memo = {}
    return process(N, E, S, K)

'''
        复杂度分析

暴力姐：每步左右两个选择，k步，就是高度为k的二叉树，最差情况的复杂度是   O(2^k)

memo:  memo表是cur*rest = N * k的规模，就是看可变参数的变化范围。求每个各自的代价是O(1)，求所有各自就是O(N*k)

        注意求每个各自是没有枚举行为的。枚举行为是什么在之后的dp中会继续。
---------------------------------------------------------    
'''
'''
dp动态规划的改写：
（1）几个可变参数就是几维的dp表，这里就是cur*rest = （N+1） * （k+1）的二维表，画出来；
        多一个为了表达方便对应，0可以不用

假设    1 2 3 4 5     初始在2位置，E=4，k=4，则cur=N=5，dp是6行5列
               rest            
            0 1 2 3 4
          0 - - - - -
          1 
     cur  2         * 
          3 
          4 
          5 


（2）标记目标所求，即dp[4][0]位置的值 *

（3）由basecase得到初始值，即rest=0时，cur=E为1，其他都是0；

               rest            
            0 1 2 3 4
          0 - - - - -
          1 0
     cur  2 0 a     * 
          3 0   ？
          4 1 b
          5 0
          
（4）选定任意位置，推到他的涞源。看真正使用递归的地方:
        
        <1>cur=1,  dp[1][rest] = dp[2][rest - 1]-->dp[1][1]=dp[2][0]=0,dp[1][2]=dp[2][1]，即第一行的值是左下角的值；
        <2>cur=5， 第5行是左上角的值；
        <3>中间位置：左上角 + 左下角的和
（5）确定顺序，保证每次需要的值之前都得到了
        for rest:
            for cur:
    即可
'''
# --------------------------------------------------- #

# --------------------------- 2.硬币组成面额问题 --------------#
'''
问题：arr是整数coin值，一个数代表一枚coin，可重复，aim目标，求组成aim最少要多少硬币。
        如 arr = [2 7 3 5 3]   aim=10

从左到右的尝试方法：i位置到结束的结果 = 选i + 不选i
'''

def minCoins(arr, aim):
    # 从i开始选，得到aim最少数；就是依次选不选
    def process(arr, index, aim):
        # 组成不了
        if aim < 0:
            return -1
        # 不需要硬币了
        if aim == 0:
            return 0

        # aim>0 有硬币还能继续，没硬币了也组成不了
        if index >= len(arr):
            return -1

        # aim>0还有coin,选择当前 + 不选
        # todo -1这种无效解会干扰，要处理一下
        # res = min(process(arr, index + 1, aim - arr[index]) + 1,
        #           process(arr, index + 1, aim))
        res1 = process(arr, index + 1, aim - arr[index])
        res2 = process(arr, index + 1, aim)

        if res1 == -1 and res2 == -1:
            return -1

        elif res1 == -1:
            return res2
        elif res1 == -1:
            return res1 + 1
        else:
            return min(res1 + 1, res2)

    return process(arr, 0, aim)
'''
memo的方法就不谈了，直接上dp。

(1)index与rest两个可变参数,index是[0,len(arr)]，rest是0-aim,
(2)目标 dp[0][aim]
(3)边界，rest=0的列全0，index=len(arr)的行全-1
(4)普遍位置，递归:从下到上，从左到右
'''
def minCoinsDp(arr, aim):

    dp = [[0] * (aim + 1) for _ in range(len(arr) + 1)]

    # 初始化
    for i in range(len(arr) + 1):
        dp[i][0] = 0
    for i in range(1, aim + 1):
        dp[len(arr)][i] = -1

    for row in range(len(arr) - 1, -1, -1):
        for col in range(1, aim):
            res1 = dp[row + 1][col - arr[row]] if col - arr[row] >= 0 else -1
            res2 = dp[row + 1][col]
            res = -1
            if res1 == -1 and res2 == -1:
                res = -1
            elif res1 == -1:
                res =  res2
            elif res1 == -1:
                res =  res1 + 1
            else:
                res = min(res1 + 1, res2)
            dp[row][col] = res

    return dp[0][aim]



# --------------------------------------------------- #
'''
        递归改DP的技术总结
    完全是手工活，不依赖技术与题意！
（1）确定可变参数、维度；
（2）确定最终所求！
（3）basecase，dp表的初始值；
（4）普遍递推关系；
（5）dp递推顺序。

'''