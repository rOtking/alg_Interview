''''''
'''
滑动窗口，窗口内最大值最小值更新结构。
'''

# ---------------------- 1.窗口 --------------------- #
'''
窗口的运动原则：
1.LR初始在arr最左，-1的位置。
2.LR都只能向右，不能向左。
3.L不能超过R。

R向右动就是有个数进到窗口中，L向右动就有数出窗口。


        最大值最小值更新结构
不遍历窗口，用很小的代价，随时获取LR的窗口内的最大值与最小值。


利用双端队列，左右两端都能进出数字。

队列中放下标不是数，这样既能直到位置，又能知道数字是多少。

1.最大值结构

dq中保证其代表的数字从大到小。

如    
     位置    0 1 2 3 4 5 6 7 8
     arr    3 2 4 6 3 7 5 3 5       

流程：
（1）R+1，dq从右边进一个当前数的位置；如果满足进的数小于队尾位置的数，则加入；否则，小于当前数的依次从右边出去，当前数进入；
        
        <1>R=0，dq空，直接进，dq=[0];
        <2>R=1,arr[1]<arr[0],进,dq=[0,1];
        <3>R=2,arr[2]=4>arr[0]与arr[1]，dq=[0,1]-->dq=[]-->dq=[2];

（2）L+1，看出去的L位置是不是dq的队首放的位置，是则dq队首位置从左边出去；不是，说明之前被淘汰了，它不会作为最大值，直接跳过。
        <4>L=0,0位置的3出去，此时最大值是2位置的4，3之前被pk掉了，也就是L不是dq队首的2，跳过；
        
结论：此结构可以保证任何时刻，窗口最大值都是dq的头结点！

dq的含义：假如R不动了，L动，谁会依次成为最大值。
    
    更新时，如果后面的点大于等于前面的，那前面的点永远不会是最大，因为后面的又大出的又晚。

            
                复杂度
dq更新总代价O(N)，单次更新的平均为O(N)/N = O(1)


最小值就是dq是从小到大。
'''
# ----------------------------------------- #

# ---------------------- 2.求窗口最大值 --------------------- #
'''
arr=[4,3,5,4,3,3,6,7],窗口w=3，从左到右，窗口最大值。

固定窗口，上面是不固定，完美使用，降维打击。就是每次LR都动1下即可。


LC239  暴力姐超时的。滑窗已解决。
'''


class Solution:
    # def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
    #     if len(nums) < k:
    #         return []
    #
    #     L = 0
    #     R = k - 1
    #     res = []
    #     while (R < len(nums)):
    #         a = max(nums[L: R + 1])
    #         res.append(a)
    #         L += 1
    #         R += 1
    #
    #     return res
    def maxSlidingWindow(self, nums, k):
        if len(nums) < k:
            return []

        dq = []
        L = -1
        R = -1

        # 右端加入k次
        for i in range(k):
            self.addFromRight(dq, nums, i)
            R += 1
        res = []
        res.append(nums[dq[0]])  # 先加第一个位置
        # LR一起动
        # todo 很关键！因为在内部R+=1，所以在len(nums) - 2的位置就是最后一次！
        while(R < len(nums) - 1):
            L += 1
            self.deleteFromLeft(dq, L)
            R += 1
            self.addFromRight(dq, nums, R)
            res.append(nums[dq[0]])
        return res

    def addFromRight(self, dq, nums, i):
        if len(dq) == 0:
            dq.append(i)

        else:
            while(len(dq) != 0 and nums[dq[-1]] <= nums[i]):
                dq.pop()

            dq.append(i)

        return

    def deleteFromLeft(self, dq, L):
        if L == dq[0]:
            dq.pop(0)
        else:
            pass

        return

sol = Solution()
res = sol.maxSlidingWindow([1,3,-1,-3,5,3,6,7], 3)
print(res)

# todo ok
# ----------------------------------------- #