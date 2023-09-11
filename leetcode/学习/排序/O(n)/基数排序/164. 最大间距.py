# 要求线性时间与空间，只能基数了


# todo 有点复杂 先略  不太重要 没必要！
class Solution:
    def maximumGap(self, nums) -> int:
        pass


    def radixSort(self, arr):
        help = [[]] * 10
        radix = 0
        # 计算位数
        max_v = max(arr)
        while(max_v > 0):
            radix += 1
            max_v = max_v // 10


        return radix

if __name__ == '__main__':
    sol = Solution()
    a = sol.radixSort(arr=[1,198])
    print(a)

