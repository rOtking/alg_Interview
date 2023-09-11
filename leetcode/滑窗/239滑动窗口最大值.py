class Solution:
    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
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

# todo ok！见左神P13-2